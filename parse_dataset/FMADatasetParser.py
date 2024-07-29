import pandas as pd
import utils
import os
import wave
import torch
import numpy as np
from msclap import CLAP
import librosa
import config
from pathos.multiprocessing import ProcessingPool as Pool
from datasets import DatasetDict
from datasets import Audio
from transformers import AutoModelForCausalLM, AutoTokenizer


class FMADatasetParser:
    def __init__(self, dataset_path, fma_data_path, fma_songs_path, train_dataset_path, genres_of_interest, generate_clap_features):
        self.dataset_path = dataset_path
        self.fma_data_path = fma_data_path
        self.fma_songs_path = fma_songs_path
        self.train_dataset_path = train_dataset_path
        self.genres_of_interest = genres_of_interest

        if generate_clap_features:
            self.clap_model = CLAP(version="2023", use_cuda=True)


    def compute_songs_features(self):
        # Load echonest data
        echonest = utils.load(os.path.join(self.fma_data_path, "fma_metadata", "echonest.csv"))

        # Reset index and sort by track_id
        echonest = echonest.reset_index()
        echonest = echonest.sort_values(by="track_id")

        # Create features dataframe
        features = pd.DataFrame()
        features["track_id"] = echonest["track_id"]

        # Flatten audio features and merge with features dataframe
        features = pd.concat([features, echonest["echonest", "audio_features"]], axis=1)

        # Load tracks info and filter by genres of interest
        tracks_info = utils.load(os.path.join(self.fma_data_path, "fma_metadata", "tracks.csv"))
        tracks_info = tracks_info[tracks_info["track"]["genre_top"].notna()]
        tracks_info = tracks_info[tracks_info["track"]["genre_top"].isin(self.genres_of_interest)]

        # Create tracks genres dataframe and merge with features dataframe
        tracks_genres = tracks_info["track"]["genre_top"].to_frame()
        tracks_genres = tracks_genres.reset_index()
        tracks_genres = tracks_genres.sort_values(by="track_id")
        features = features.merge(tracks_genres, on="track_id")

        # Save features to csv
        features.to_csv(os.path.join(self.train_dataset_path, "songs_features.csv"), index=False)


    def copy_songs_files(self):
        # Load songs features
        songs_features = pd.read_csv(os.path.join(self.train_dataset_path, "songs_features.csv"))
        song_ids = set(songs_features["track_id"])

        # Copy songs files
        for current_dir in os.walk(self.fma_songs_path):
            if len(current_dir[1]) == 0:
                files = [os.path.join(current_dir[0], file) for file in current_dir[2] if file.endswith(".mp3")]

                for file in files:
                    if int(file.split("/")[-1].split(".")[0]) in song_ids:
                        # Copy and convert mp3 to WAV
                        wav_file = os.path.join(self.train_dataset_path, file.split("/")[-1].replace(".mp3", ".wav"))
                        cmd = f"ffmpeg -i {file} {wav_file} -y"
                        os.system(cmd)

                        with wave.open(wav_file, "rb") as wav:
                            length = wav.getnframes() / wav.getframerate()

                            # some files are corrupted so we remove them
                            if length <= config.MIN_SONG_DURATION:
                                os.remove(wav_file)


    def compute_instrument(self, audio_embeddings: torch.Tensor) -> str:
        """Compute instrument using CLAP model"""
        instrument_embeddings = self.clap_model.get_text_embeddings(config.instrument_classes)
        instrument = self.clap_model.compute_similarity(
            audio_embeddings, instrument_embeddings
        ).argmax(dim=1)[0]
        return config.instrument_classes[instrument]
    

    def compute_mood(self, audio_embeddings: torch.Tensor) -> str:
        """Compute mood using CLAP model"""
        mood_embeddings = self.clap_model.get_text_embeddings(config.mood_theme_classes)
        mood = self.clap_model.compute_similarity(
            audio_embeddings, mood_embeddings
        ).argmax(dim=1)[0]
        return config.mood_theme_classes[mood]


    def compute_key(self, audio: np.ndarray, sampling_rate: int) -> str:
        """Compute key of the audio"""
        chroma = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)
        key = np.argmax(np.sum(chroma, axis=1))
        return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][key]


    def generate_clap_features_for_song(self, csv_row):
        track_id = self.songs_features.iloc[csv_row]["track_id"]
        track_id = str(track_id).zfill(6)

        wav_file = os.path.join(self.train_dataset_path, f"{track_id}.wav")

        if os.path.exists(wav_file):
            audio, sampling_rate = librosa.load(wav_file)
            audio_embeddings = self.clap_model.get_audio_embeddings([wav_file])

            instrument = self.compute_instrument(audio_embeddings)
            mood = self.compute_mood(audio_embeddings)
            key = self.compute_key(audio, sampling_rate)

            self.songs_features.at[csv_row, "instrument"] = instrument
            self.songs_features.at[csv_row, "mood"] = mood
            self.songs_features.at[csv_row, "key"] = key


    def generate_clap_features(self):
        # load song features

        self.songs_features = pd.read_csv(os.path.join(self.train_dataset_path, "songs_features.csv"))
        self.songs_features["instrument"] = None
        self.songs_features["mood"] = None
        self.songs_features["key"] = None

        def file_exists(track_id, base_path):
            file_path = os.path.join(base_path, f"{str(track_id).zfill(6)}.wav")
            return os.path.exists(file_path)

        self.songs_features['file_exists'] = self.songs_features['track_id'].apply(lambda x: file_exists(x, config.TRAINING_DATASET_PATH))
        self.songs_features = self.songs_features[self.songs_features['file_exists']]
        self.songs_features = self.songs_features.drop(columns=['file_exists'], )

        self.songs_features = self.songs_features.head(5)
        self.songs_features.reset_index(inplace=True)

        # parallelize
        for row in range(len(self.songs_features)):
            self.generate_clap_features_for_song(row)

        self.songs_features.to_csv(os.path.join(self.train_dataset_path, "full_songs_features.csv"))

        del self.songs_features

    
    def generate_caption(self, features):
        metadata_str = "\n".join([f"{key}: {value}" for key, value in features.items()])
        metadata_str = """
            Based on the following metadata, generate a description for the song:\n
        """ + metadata_str

        model = AutoModelForCausalLM.from_pretrained(config.CAPTION_GENERATOR_NAME, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(config.CAPTION_GENERATOR_NAME)

        inputs = tokenizer(metadata_str, return_tensors="pt")
        inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=config.MAX_CAPTION_LENGTH)
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return caption


    def build_dataset(self):
        songs_features = pd.read_csv(os.path.join(self.train_dataset_path, "full_songs_features.csv"))
        songs_features = songs_features[songs_features.notna()]

        csv_dataset = pd.DataFrame({"audio": [], "caption": [], "genre": []})

        # parallelize
        for _, row in songs_features.iterrows():
            track_id = row["track_id"]
            genre = row["genre_top"]
            features = songs_features[songs_features["track_id"] == track_id].reset_index(drop=True)
            features = features.drop("track_id", axis=1).to_dict()
    
            full_track_id = str(track_id).zfill(6)

            audio_file = os.path.join(self.train_dataset_path, f"{full_track_id}.wav")
            caption = self.generate_caption(features)

            csv_dataset = pd.concat([csv_dataset, pd.DataFrame({"audio": audio_file, "caption": caption, "genre": genre}, index=[0])], ignore_index=True)

            if _ == 2:
                break

        csv_dataset.to_csv(os.path.join(self.train_dataset_path, "dataset.csv"), index=False)

        dataset = DatasetDict.from_csv({"train": os.path.join(self.train_dataset_path, "dataset.csv")})
        dataset = dataset.cast_column("audio", Audio())

        dataset.save_to_disk(os.path.join(self.dataset_path, "hf_dataset"))