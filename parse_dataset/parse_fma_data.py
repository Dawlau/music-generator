import os
import argparse

from FMADatasetParser import FMADatasetParser
import config

def make_args():
    def list_of_str(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default=config.DATASET_PATH)
    parser.add_argument("--fma_data_path", type=str, default=config.FMA_DATA_PATH)
    parser.add_argument("--fma_songs_path", type=str, default=config.FMA_SONGS_PATH)
    parser.add_argument("--train_dataset_path", type=str, default=config.TRAINING_DATASET_PATH)
    parser.add_argument("--genres_of_interest", type=list_of_str, default=config.GENRES_OF_INTEREST)
    parser.add_argument("--compute_songs_features", type=bool, default=True)
    parser.add_argument("--move_songs_files", type=bool, default=True)
    parser.add_argument("--generate_clap_features", type=bool, default=True)
    parser.add_argument("--build_dataset", type=bool, default=True)

    args = parser.parse_args()

    return args


def main():
    if not os.path.exists(config.DATASET_PATH):
        os.makedirs(config.DATASET_PATH)

    if not os.path.exists(config.TRAINING_DATASET_PATH):
        os.makedirs(config.TRAINING_DATASET_PATH)

    args = make_args()

    fma_parser = FMADatasetParser(
        args.dataset_path,
        args.fma_data_path,
        args.fma_songs_path,
        args.train_dataset_path,
        args.genres_of_interest,
        args.generate_clap_features
    )

    if args.compute_songs_features:
        fma_parser.compute_songs_features()

    if args.move_songs_files:
        fma_parser.copy_songs_files()

    if args.generate_clap_features:
        fma_parser.generate_clap_features()

    if args.build_dataset:
        fma_parser.build_dataset()

if __name__ == "__main__":
    main()