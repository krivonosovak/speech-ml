import argparse
import os

from models.dataloader import KaldiExtractor


def get_config():
    parser = argparse.ArgumentParser(description='get fbank and mfcc features from audio files')

    parser.add_argument('--clean_path', type=str, default='clean_data', help='path to clean data dir')
    parser.add_argument('--noise_path', type=str, default='noise_data', help='path to noise data dir')
    parser.add_argument('--out_dir', type=str, default=None, help='dir for extraction feature files')
    parser.add_argument('--kaldi_path', type=str, default='kaldi', help='kaldi dir')

    config = parser.parse_args()

    return config


def main():
    config = get_config()

    extr = KaldiExtractor(kaldipath=config.kaldi_path)
    extr.extract(dir_in=config.clean_path, dir_out=config.out_dir)
    extr.extract(dir_in=config.noise_path, dir_out=config.out_dir)


if __name__ == '__main__':
    main()
