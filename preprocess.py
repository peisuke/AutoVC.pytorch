import argparse
import math
import json
import os
import glob
import sys
import tqdm
import numpy as np
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wav-dir', type=str, required=True, help='The directory that contains wave files [./wavs]')
    parser.add_argument('--output-dir', default='./data', type=str, help='Output dir [./data]')
    args = parser.parse_args()

    wav_dir = args.wav_dir
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    speakers = sorted(glob.glob(os.path.join(wav_dir, '*')))
    speakers = list(filter(lambda x: os.path.isdir(x), speakers))

    random.seed(1234)
    random.shuffle(speakers)
    
    nb_speakers = len(speakers)
    nb_unseen_speakers = nb_speakers // 5
    nb_seen_speakers = nb_speakers - nb_unseen_speakers

    seen_speakers = speakers[:nb_seen_speakers]
    unseen_speakers = speakers[nb_seen_speakers:]
    
    train_data = {}
    seen_test_data = {}
    unseen_test_data = {}

    for s in seen_speakers:
        speaker = os.path.basename(s)
        wavfiles = sorted(glob.glob(os.path.join(s, '*.wav')))
        nb_files = len(wavfiles)
        nb_test = nb_files // 10

        trainfiles = wavfiles[:-nb_test]
        testfiles = wavfiles[-nb_test:]

        train_data[speaker] = trainfiles
        seen_test_data[speaker] = testfiles

    for s in unseen_speakers:
        speaker = os.path.basename(s)
        wavfiles = sorted(glob.glob(os.path.join(s, '*.wav')))
        unseen_test_data[speaker] = wavfiles

    with open(os.path.join(output_dir, 'train_files.json'), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(output_dir, 'seen_test_files.json'), 'w') as f:
        json.dump(seen_test_data, f)

    with open(os.path.join(output_dir, 'unseen_test_files.json'), 'w') as f:
        json.dump(unseen_test_data, f)