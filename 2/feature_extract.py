import argparse
import os
import tempfile
import pandas as pd
import librosa
import numpy as np
import torch
from  laughter_classification.data_extract import DataLoader, Extractor

def get_config():
    parser = argparse.ArgumentParser(description='get fbank and mfcc features from audio files')

    parser.add_argument('--corpus_path', type=str, default='vocalizationcorpus', help='path to corpus dir')
    parser.add_argument('--feature_path', type=str, default=os.getcwd(), help='path to feature dir')                 
    parser.add_argument('--frame_sec', type=float, default=0.03,
                            help='length of window in sec')
    parser.add_argument('--nfbank', type=int, default=128, help='fbank feature count')
    parser.add_argument('--nmfcc', type=int, default=13, help='mfcc feature count')
    parser.add_argument('--naudio', type=int, default=None, help='number of audios to parse, if not defined parses all')
    
  



    config = parser.parse_args()

    return config


def main():
    
    config = get_config()
    data = DataLoader(config.corpus)
    result = data.get_data(frame_sec=config.frame_sec, naudio=config.naudio,
                       feature_path=config.feature_path, load=False, nfbank=config.nfbank, nmfcc=config.nmfcc)
    
if __name__ == '__main__':
    main()	
