import numpy as np
import librosa, math
from hparams import hparams as hp

def load_wav(filename):
    x = librosa.load(filename, sr=hp.sample_rate)[0]
    return x

def save_wav(y, filename) :
    librosa.output.write_wav(filename, y, hp.sample_rate)
    
#def encode_16bits(x) :
#    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

mel_basis = None

def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def build_mel_basis():
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)

def normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

def amp_to_db(x):
    #return 20 * np.log10(np.maximum(1e-5, x))
    return np.log(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - hp.ref_level_db
    return normalize(S)

def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return S

def stft(y):
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
