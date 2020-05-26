import os
import math
import random
import json
import torch
import torch.utils.data
import numpy as np
from hparams import hparams as hp

from utils.dsp import load_wav
from utils.dsp import melspectrogram

class AudiobookDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, train=False):
        self.data = []
        for s, filenames in input_data.items():
            for f in filenames:
                self.data.append({'file': f, 'speaker': s})

    def __getitem__(self, index):
        p = self.data[index]
        f = p['file']
        
        wav = load_wav(f)
           
        return wav, f

    def __len__(self):
        return len(self.data)

def pad_seq(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[1])/base))
    len_pad = len_out - x.shape[1]
    assert len_pad >= 0
    return np.pad(x, ((0,0), (0,len_pad)), 'constant'), len_pad

def train_collate(batch):
    mel_win = hp.seq_len // hp.hop_length
    #max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    #mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    #sig_offsets = [(offset + pad) * hp.hop_length for offset in mel_offsets]
    
    max_offsets = [x[0].shape[-1] - hp.seq_len for x in batch]
    sig_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        
    #mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
    #        for i, x in enumerate(batch)]

    wav = [x[0][sig_offsets[i]:sig_offsets[i] + hp.seq_len] \
              for i, x in enumerate(batch)]
    
    mels = [melspectrogram(w[:-1]) for w in wav]

    fname = [x[1] for x in batch]

    mels = torch.FloatTensor(mels)
    wav = torch.FloatTensor(wav)
    
    #wav = 2 * wav[:, :hp.seq_len].float() / (2**hp.bits - 1.) - 1.
    
    return mels, wav, fname

def test_collate(batch):
    wav = [x[0] for i, x in enumerate(batch)]
    mels = [pad_seq(melspectrogram(w))[0] for w in wav]

    fname = [x[1] for x in batch]

    mels = torch.FloatTensor(mels)
    wav = torch.FloatTensor(wav)
    
    return mels, wav, fname
