import os
import math
import random
import json
import torch
import torch.utils.data
import numpy as np
from spec_augment import spec_augment

from hparams import hparams as hp
from utils.dsp import load_wav
from utils.dsp import melspectrogram
from utils.dsp import pitch

class AudiobookDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, train=False):
        self.data = input_data

    def __getitem__(self, index):
        p = self.data[index]
        fs = p['wav']
        e = p['emb']
        
        f = random.choice(fs)
        wav = load_wav(f)
        emb = np.load(e)

        if len(wav) < hp.seq_len:
            wav = np.pad(wav, (0, hp.seq_len - len(wav)), mode='constant')
           
        return wav, emb, f

    def __len__(self):
        return len(self.data)

def pad_seq(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[1])/base))
    len_pad = len_out - x.shape[1]
    assert len_pad >= 0
    return np.pad(x, ((0,0), (0,len_pad)), 'constant'), len_pad

def pad_pitch(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad)), 'constant'), len_pad

def train_collate(batch):
    mel_win = hp.seq_len // hp.hop_length
    
    max_offsets = [x[0].shape[-1] - hp.seq_len + 1 for x in batch]

    sig_offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wav = [x[0][sig_offsets[i]:sig_offsets[i] + hp.seq_len] \
              for i, x in enumerate(batch)]
    
    # volume augmentation
    wav = [w * 2 ** (np.random.rand() * 2 - 1) for w in wav]
    
    mels = [melspectrogram(w[:-1]) for w in wav]
    ps = [pitch(w[:-1]) for w in wav]
   
    # spec augmentation
    mels_and_ps = [spec_augment(m, p) for m, p in zip(mels, ps)]
    mels = [m for (m, p) in mels_and_ps]
    ps = [p.astype(np.int) for (m, p) in mels_and_ps]
    
    emb = [x[1] for x in batch]
    fname = [x[2] for x in batch]

    mels = torch.FloatTensor(mels)
    ps = torch.LongTensor(ps)
    emb = torch.FloatTensor(emb)
    
    onehot = torch.zeros(ps.shape[0] * ps.shape[1], 256, dtype=torch.float)
    onehot = onehot.scatter_(1, ps.view(-1).unsqueeze(1), 1)
    onehot = onehot.view(ps.shape[0], ps.shape[1], 256)
    ps = onehot
    
    mels = mels.transpose(2,1)

    return mels, ps, emb

def test_collate(batch):
    wavs = []
    embs = []
    for b in batch:
        wav = b[0]
        for p in range(0, len(wav), hp.seq_len):
            wav_seq = wav[p:p+hp.seq_len]
            if len(wav_seq) < hp.seq_len:
                wav_seq = np.pad(wav_seq, (0, hp.seq_len - len(wav_seq)), mode='constant')
            wavs.append(wav_seq)
            embs.append(b[1])
    
    mels = [pad_seq(melspectrogram(w))[0] for w in wavs]
    ps = [pad_pitch(pitch(w))[0] for w in wavs]
    
    mels = torch.FloatTensor(mels)
    ps = torch.LongTensor(ps)
    embs = torch.FloatTensor(embs)
    
    onehot = torch.zeros(ps.shape[0] * ps.shape[1], hp.dim_pitch, dtype=torch.float)
    onehot = onehot.scatter_(1, ps.view(-1).unsqueeze(1), 1)
    onehot = onehot.view(ps.shape[0], ps.shape[1], hp.dim_pitch)
    ps = onehot
    
    mels = mels.transpose(2,1)
    
    return mels, ps, embs
