import pandas as pd
import os
from datasets import DatasetDict
from datasets import Audio
import re


class SongDescriberDatasetParser:
    def __init__(self, dataset_path, song_describer_data_path, song_describer_songs_path, validation_dataset_path, genres_of_interest):
        self.dataset_path = dataset_path
        self.song_describer_data_path = song_describer_data_path
        self.song_describer_songs_path = song_describer_songs_path
        self.validation_dataset_path = validation_dataset_path
        self.genres = genres_of_interest


    def filter_relevant_songs(self):
        song_describer_data = pd.read_csv(os.path.join(self.song_describer_data_path, "song_describer.csv"))
        song_describer_data = song_describer_data[~song_describer_data["is_valid_subset"].isna()]

        track_ids = [int(file.split(".")[0]) for file in os.listdir(self.validation_dataset_path) if file.endswith(".wav")]

        genres = []
        valid_track_ids = []
        captions = []

        for track_id in track_ids:
            caption = song_describer_data[song_describer_data["track_id"] == track_id]["caption"].values
            
            if len(caption) == 0:
                continue

            caption = " ".join(caption)

            found_genre = False
            for genre in self.genres:
                if re.search(genre, caption):
                    found_genre = True
                    break

            if not found_genre:
                os.remove(os.path.join(self.validation_dataset_path, f"{track_id}.wav"))
            else:
                genres.append(genre)
                valid_track_ids.append(track_id)
                captions.append(caption)

        new_song_describer_data = pd.DataFrame({"track_id": valid_track_ids, "caption": captions, "genre": genres})
        new_song_describer_data.to_csv(os.path.join(self.validation_dataset_path, "songs.csv"))


    def move_files(self):
        for current_dir in os.walk(self.song_describer_songs_path):
            if len(current_dir[1]) == 0:
                files = [os.path.join(current_dir[0], file) for file in current_dir[2] if file.endswith(".mp3")]

                for file in files:
                    # Copy and convert mp3 to WAV
                    trimmed_filename = ".".join(file.split("/")[-1].replace(".mp3", ".wav").split(".")[::2])
                    
                    wav_file = os.path.join(self.validation_dataset_path, trimmed_filename)
                    cmd = f"ffmpeg -i {file} {wav_file} -y"
                    os.system(cmd)


    def build_dataset(self):
        songs = pd.read_csv(os.path.join(self.validation_dataset_path, "songs.csv"))
        csv_dataset = pd.DataFrame({"audio": [], "caption": []})

        csv_dataset["audio"] = [f"{os.path.join(self.validation_dataset_path, str(track_id))}.wav" for track_id in songs["track_id"]]
        csv_dataset["caption"] = songs["caption"]
        csv_dataset["genre"] = songs["genre"]

        csv_dataset.to_csv(os.path.join(self.validation_dataset_path, "dataset.csv"), index=False)

        dataset = DatasetDict.from_csv({"test": os.path.join(self.validation_dataset_path, "dataset.csv")})
        dataset = dataset.cast_column("audio", Audio())
        
        train_dataset = DatasetDict.load_from_disk(os.path.join(self.dataset_path, "hf_dataset"))
        dataset["train"] = train_dataset["train"]

        dataset.save_to_disk(os.path.join(self.validation_dataset_path, "hf_dataset"))
