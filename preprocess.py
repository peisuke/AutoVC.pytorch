import argparse
import math
import json
import tqdm
import os
import glob
import sys
import tqdm
import numpy as np
import librosa
import random
from hparams import hparams as hp
from resemblyzer import VoiceEncoder, preprocess_wav

def compute_embed(files, encoder):
    emb = []
    
    files = random.sample(files, min(len(files), 20))
    for f in files:
        wav = preprocess_wav(f)
        e = encoder.embed_utterance(wav)
        emb.append(e)
    emb = np.array(emb)
    emb = emb.mean(axis=0)
    return emb

def create_data(data, output_dir, phase, seq_len=None):
    ret = []

    for k, v in tqdm.tqdm(data.items()):
        speaker_dir = os.path.join(output_dir, phase, k)
        os.makedirs(speaker_dir, exist_ok=True)
        
        emb_file = os.path.join(speaker_dir, 'emb.npy')
        np.save(emb_file, v['emb'])

        for fname in v['files']:
            n = os.path.splitext(os.path.basename(fname))[0]
            target_dir = os.path.join(speaker_dir, n)
            os.makedirs(target_dir, exist_ok=True)
            x, _ = librosa.load(fname, sr=hp.sample_rate)
            
            files = []
            if seq_len is not None:
                for i, p in enumerate(range(0, len(x), seq_len)):
                    wav = x[p:p+seq_len*2]
                    if len(wav) < seq_len:
                        wav = np.pad(wav, (0, seq_len - len(wav)), mode='constant')

                    filename = os.path.join(target_dir, '{0:04d}.wav'.format(i))                                        
                    librosa.output.write_wav(filename, wav, hp.sample_rate)
                    files.append(filename)
            else:
                filename = os.path.join(target_dir, '{0:04d}.wav'.format(0))                                        
                librosa.output.write_wav(filename, x, hp.sample_rate)
                files.append(filename)

            ret.append({'wav': files, 'emb': emb_file})
            
    return ret
            
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

    #### DEBUG
    speakers = speakers[:5]

    random.seed(1234)
    random.shuffle(speakers)
    
    nb_speakers = len(speakers)
    nb_unseen_speakers = nb_speakers // 5
    nb_seen_speakers = nb_speakers - nb_unseen_speakers

    seen_speakers = speakers[:nb_seen_speakers]
    unseen_speakers = speakers[nb_seen_speakers:]
    
    encoder = VoiceEncoder()
   
    train_data = {}
    test_data = {}
    unseen_data = {}

    for s in seen_speakers:
        speaker = os.path.basename(s)
        wavfiles = sorted(glob.glob(os.path.join(s, '*.wav')))

        #### DEBUG
        wavfiles = wavfiles[:10]
        
        emb = compute_embed(wavfiles, encoder)

        nb_files = len(wavfiles)
        nb_test = nb_files // 10

        trainfiles = wavfiles[:-nb_test]
        testfiles = wavfiles[-nb_test:]

        train_data[speaker] = {'files': trainfiles, 'emb': emb}
        test_data[speaker] = {'files': testfiles, 'emb': emb}

    for s in unseen_speakers:
        speaker = os.path.basename(s)
        wavfiles = sorted(glob.glob(os.path.join(s, '*.wav')))

        emb = compute_embed(wavfiles, encoder)
        
        #### DEBUG
        wavfiles = wavfiles[:10]

        unseen_data[speaker] = {'files': wavfiles, 'emb': emb}

    train_data = create_data(train_data, output_dir, 'train', hp.seq_len)
    test_data = create_data(test_data, output_dir, 'test')
    unseen_data = create_data(unseen_data, output_dir, 'unseen')

    with open(os.path.join(output_dir, 'train_data.json'), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(output_dir, 'test_data.json'), 'w') as f:
        json.dump(test_data, f)

    with open(os.path.join(output_dir, 'unseen_data.json'), 'w') as f:
        json.dump(unseen_data, f)
