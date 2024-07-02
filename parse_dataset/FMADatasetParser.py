import pandas as pd
import utils
import os
import wave


class FMADatasetParser:
    def __init__(self, fma_data_path, fma_songs_path, dataset_path, genres_of_interest):
        self.fma_data_path = fma_data_path
        self.fma_songs_path = fma_songs_path
        self.dataset_path = dataset_path
        self.genres_of_interest = genres_of_interest

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
        features.to_csv(os.path.join(self.dataset_path, "songs_features.csv"), index=False)

    def copy_songs_files(self):
        # Load songs features
        songs_features = pd.read_csv(os.path.join(self.dataset_path, "songs_features.csv"))
        song_ids = set(songs_features["track_id"])

        # Copy songs files
        for current_dir in os.walk(self.fma_songs_path):
            if len(current_dir[1]) == 0:
                files = [os.path.join(current_dir[0], file) for file in current_dir[2] if file.endswith(".mp3")]

                for file in files:
                    if int(file.split("/")[-1].split(".")[0]) in song_ids:
                        # Copy and convert mp3 to WAV
                        wav_file = os.path.join(self.dataset_path, file.split("/")[-1].replace(".mp3", ".wav"))
                        cmd = f"ffmpeg -i {file} {wav_file} -y"
                        os.system(cmd)

                        with wave.open(wav_file, "rb") as wav:
                            length = wav.getnframes() / wav.getframerate()

                            if length <= 25:
                                os.remove(wav_file)
