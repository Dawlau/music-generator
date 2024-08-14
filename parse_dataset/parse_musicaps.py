import os
import argparse
import config
from datasets import load_dataset, Audio
from pathlib import Path
import subprocess

def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir="/tmp/musiccaps",
    num_attempts=5,
    url_base="https://www.youtube.com/watch?v="
):
    status = False

    command = f"""
        yt-dlp --quiet --force-keyframes-at-cuts --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" "{url_base}{video_identifier}"
    """.strip()

    print(command)

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, "Downloaded"



def process(example):
    outfile_path = os.path.join(config.TEST_DATASET_PATH, f"{example['ytid']}.wav")
    status = True

    if not os.path.exists(outfile_path):
        status = False
        status, log = download_clip(
            example["ytid"],
            outfile_path,
            example["start_s"],
            example["end_s"],
        )

    example["audio"] = outfile_path
    example["download_status"] = status
    return example


def main():
    if not os.path.exists(config.TEST_DATASET_PATH):
        os.makedirs(config.TEST_DATASET_PATH)

    ds = load_dataset("google/MusicCaps", split="train")

    ds = ds.map(
        process,
        num_proc=5,
        writer_batch_size=1000,
        keep_in_memory=False
    ).cast_column("audio", Audio())

    ds.save_to_disk(config.TEST_DATASET_PATH)


    

if __name__ == "__main__":
    main()