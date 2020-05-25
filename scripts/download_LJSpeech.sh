#!/bin/bash

wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar jxf LJSpeech-1.1.tar.bz2
mv LJSpeech-1.1/wavs/ ./wavs
#rm -rf LJSpeech-1.1/
#rm -rf LJSpeech-1.1.tar.bz2
