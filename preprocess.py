import argparse
import math
import json
import os
import glob
import sys
import tqdm
import numpy as np

def get_files(path, extension='.wav') :
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames

#def convert_file(path) :
#    wav = load_wav(path)
#    mel = melspectrogram(wav)
#    quant = wav * (2**15 - 0.5) - 0.5
#    return mel.astype(np.float32), quant.astype(np.int16)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wav-dir', type=str, required=True, help='The directory that contains wave files [./wavs]')
    parser.add_argument('--output-dir', default='./data', type=str, help='Output dir [./data]')
    parser.add_argument('--nb-test', default=10, type=int, help='The number of test sample data')
    args = parser.parse_args()

    wav_dir = args.wav_dir
    output_dir = args.output_dir
    nb_test = args.nb_test
    
    wav_files = sorted(get_files(wav_dir))
    
    #quant_dir = os.path.join(output_dir, 'quant')
    #mel_dir = os.path.join(output_dir, 'mel')

    os.makedirs(output_dir, exist_ok=True)
    #os.makedirs(quant_dir, exist_ok=True)
    #os.makedirs(mel_dir, exist_ok=True)
    
    # This will take a while depending on size of dataset
    dataset_ids = []
    for i, path in enumerate(tqdm.tqdm(wav_files)):
        #m, x = convert_file(path)
        
        #id = os.path.splitext(os.path.basename(path))[0]
        #mel_file = os.path.join(mel_dir, f'{id}.npy')
        #quant_file = os.path.join(quant_dir, f'{id}.npy')

        dataset_ids.append({
            'file': path
            #'mel': mel_file,
            #'quant': quant_file
        })
        #np.save(mel_file, m)
        #np.save(quant_file, x)
  
    nb_data = len(dataset_ids)
    nb_test = nb_data // 10
    train_ids = dataset_ids[:-nb_test]
    test_ids = dataset_ids[-nb_test:]

    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_ids, f)
    
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_ids, f)
