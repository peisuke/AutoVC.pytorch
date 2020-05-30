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

def compute_embed(files):
    encoder = VoiceEncoder()
    emb = []
    
    files = random.sample(files, 20)
    for f in tqdm.tqdm(files):
        wav = preprocess_wav(f)
        e = encoder.embed_utterance(wav)
        emb.append(e)
    emb = np.array(emb)
    emb = emb.mean(axis=0)
    return emb

def create_data(data, phase):
    ret = []
     
    for k, v in data.items():
        for fname in tqdm.tqdm(v):
            n = os.path.splitext(os.path.basename(fname))[0]
            target_dir = os.path.join(output_dir, '{}/{}/{}'.format(phase, k, n))
            os.makedirs(target_dir, exist_ok=True)
            x, _ = librosa.load(fname, sr=hp.sample_rate)
            
            files = []
            for i, p in enumerate(range(0, len(x), hp.seq_len)):
                wav = x[p:p+hp.seq_len*2]
                if len(wav) < hp.seq_len:
                    wav = np.pad(wav, (0, hp.seq_len - len(wav)), mode='constant')
                
                filename = os.path.join(target_dir, '{0:04d}.wav'.format(i))                                        
                librosa.output.write_wav(filename, wav, hp.sample_rate)
                files.append(filename)
            
            ret.append({'files': files, 'speaker': k})
            
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

    random.seed(1234)
    random.shuffle(speakers)
    
    nb_speakers = len(speakers)
    nb_unseen_speakers = nb_speakers // 5
    nb_seen_speakers = nb_speakers - nb_unseen_speakers

    seen_speakers = speakers[:nb_seen_speakers]
    unseen_speakers = speakers[nb_seen_speakers:]
    
    speaker_emb = {}
    for s in speakers:
        speaker = os.path.basename(s)
        wavfiles = sorted(glob.glob(os.path.join(s, '*.wav')))
        emb = compute_embed(wavfiles)
        speaker_emb[speaker] = emb.tolist()
   
    '''
    train_data = {}
    test_data = {}
    unseen_data = {}
    
    for s in seen_speakers:
        speaker = os.path.basename(s)
        wavfiles = sorted(glob.glob(os.path.join(s, '*.wav')))
        nb_files = len(wavfiles)
        nb_test = nb_files // 10

        trainfiles = wavfiles[:-nb_test]
        testfiles = wavfiles[-nb_test:]

        train_data[speaker] = trainfiles
        test_data[speaker] = testfiles

    for s in unseen_speakers:
        speaker = os.path.basename(s)
        wavfiles = sorted(glob.glob(os.path.join(s, '*.wav')))
        unseen_data[speaker] = wavfiles

    train_data = create_data(train_data, 'train')
    test_data = create_data(test_data, 'test')
    unseen_data = create_data(unseen_data, 'unseen')
    '''

    with open(os.path.join(output_dir, 'speaker_emb.json'), 'w') as f:
        json.dump(speaker_emb, f)
   
    '''
    with open(os.path.join(output_dir, 'train_data.json'), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(output_dir, 'test_data.json'), 'w') as f:
        json.dump(test_data, f)

    with open(os.path.join(output_dir, 'unseen_data.json'), 'w') as f:
        json.dump(unseen_data, f)
    '''
