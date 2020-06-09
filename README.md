# AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss

This repository is reproduced code of "AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss"

The paper link is [here](https://arxiv.org/abs/1905.05879)

# Preparation

Pretrained model of WaveNet vocoder is from [author's original repogitory](https://github.com/auspicious3000/autovc)

# How to use

```
./scripts/downloadVCTK.sh
```

```
python preprocess.py --wav-dir ./VCTK-Corpus/wav48/
```

```
python train.py
```

```
python3 inference.py --output ./output.wav \
                     --src-wav [path to src wav] \
                     --src-emb [path to src embedding] \
                     --tgt-emb [path to target embedding] \
                     --vocoder [path to vocoder checkpoint] \
                     --autovc [path to autovc checkpoint]
```

# Sample Audio

WIP
