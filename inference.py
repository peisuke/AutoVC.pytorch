import argparse
import math
import json
import numpy as np
import os
import torch

import librosa

from hparams import hparams as hp
from utils.dsp import load_wav
from utils.dsp import melspectrogram
from model_vc import Generator
from synthesis import build_model
from synthesis import wavegen

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output', type=str, required=True, help='path to output wav(./output.wav)')
    parser.add_argument('--src-wav', type=str, required=True, help='path to src wav(./data/test/[speaker]/[filename]/0000.wav')
    parser.add_argument('--src-emb', type=str, required=True, help='path to src wav(./data/test/[speaker]/emb.npy')
    parser.add_argument('--tgt-emb', type=str, required=True, help='path to src wav(./data/test/[speaker]/emb.npy')
    parser.add_argument('--vocoder', type=str, required=True, help='path to checkpoint_step001000000_ema.pth')
    parser.add_argument('--autovc', type=str, required=True, help='checkpoints/checkpoint_step000600.pth')
    args = parser.parse_args()
    
    output_path = args.output
    src_wav_path = args.src_wav
    src_emb_path = args.src_emb
    tgt_emb_path = args.tgt_emb
    vocoder_checkpoint_path = args.vocoder
    autovc_checkpoint_path = args.autovc

    dim_neck = 32
    dim_emb = 256
    dim_pre = 512
    freq = 32

    device = torch.device('cpu')
    wavnet = build_model().to(device)
    checkpoint = torch.load(vocoder_checkpoint_path, map_location=device)
    wavnet.load_state_dict(checkpoint["state_dict"])

    wav = load_wav(src_wav_path)
    emb = np.load(src_emb_path)
    emb_tgt = np.load(tgt_emb_path)

    mel = melspectrogram(wav)

    pad_len = math.ceil(mel.shape[1] / 32) * 32 - mel.shape[1]
    mel = np.pad(mel, ((0,0), (0, pad_len)), mode='constant')

    mel = torch.FloatTensor(mel)
    emb = torch.FloatTensor(emb)
    emb_tgt = torch.FloatTensor(emb_tgt)

    model = Generator(dim_neck, dim_emb, dim_pre, freq)

    checkpoint = torch.load(autovc_checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    x = mel.unsqueeze(0).transpose(2,1) 
    e = emb.unsqueeze(0)
    et = emb_tgt.unsqueeze(0)

    mel_outputs, mel_outputs_postnet, codes = model(x, e, et)
    mel_rec = mel_outputs_postnet.transpose(2,1).cpu().detach().numpy()[0]

    mel_rec = mel_rec[:,:-pad_len]

    c = np.transpose(mel_rec, (1, 0))
    waveform = wavegen(wavnet, device, c=c)
    librosa.output.write_wav(output_path, waveform, sr=16000)