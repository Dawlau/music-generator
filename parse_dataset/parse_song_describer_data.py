import os
import argparse
import config

from SongDescriberDatasetParser import SongDescriberDatasetParser
from datasets import DatasetDict

def make_args():
    def list_of_str(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default=config.DATASET_PATH)    
    parser.add_argument("--song_describer_data_path", type=str, default=config.SONG_DESCRIBER_DATA_PATH)
    parser.add_argument("--song_describer_songs_path", type=str, default=config.SONG_DESCRIBER_DATA_SONGS_PATH)
    parser.add_argument("--validation_dataset_path", type=str, default=config.VALIDATION_DATASET_PATH)
    parser.add_argument("--genres_of_interest", type=list_of_str, default=config.GENRES_OF_INTEREST)
    parser.add_argument("--move_songs_files", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--filter_relevant_songs", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--build_dataset", type=bool, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    return args


def main():
    if not os.path.exists(config.DATASET_PATH):
        os.makedirs(config.DATASET_PATH)

    if not os.path.exists(config.VALIDATION_DATASET_PATH):
        os.makedirs(config.VALIDATION_DATASET_PATH)

    args = make_args()

    song_describer_dataset_parser = SongDescriberDatasetParser(
        args.dataset_path,
        args.song_describer_data_path,
        args.song_describer_songs_path,
        args.validation_dataset_path,
        args.genres_of_interest,
    )

    if args.move_songs_files is not None:
        song_describer_dataset_parser.move_files()

    if args.filter_relevant_songs is not None:
        song_describer_dataset_parser.filter_relevant_songs()

    if args.build_dataset is not None:
        song_describer_dataset_parser.build_dataset()

if __name__ == "__main__":
    main()