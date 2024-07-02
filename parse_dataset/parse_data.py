import os
import argparse

from FMADatasetParser import FMADatasetParser
from config import *


def make_args():
    def list_of_str(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()

    parser.add_argument("--fma_data_path", type=str, default=FMA_DATA_PATH)
    parser.add_argument("--fma_songs_path", type=str, default=FMA_SONGS_PATH)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--genres_of_interest", type=list_of_str, default=GENRES_OF_INTEREST)
    parser.add_argument("--compute_songs_features", type=bool, default=True)
    parser.add_argument("--move_songs_files", type=bool, default=True)

    args = parser.parse_args()

    return args


def main():
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    args = make_args()

    fma_parser = FMADatasetParser(
        args.fma_data_path,
        args.fma_songs_path,
        args.dataset_path,
        args.genres_of_interest
    )

    if args.compute_songs_features:
        fma_parser.compute_songs_features()

    if args.move_songs_files:
        fma_parser.copy_songs_files()

if __name__ == "__main__":
    main()