import argparse
import logging
import os
from noisifier.noisifier import Noisifier



def get_config():
	parser = argparse.ArgumentParser(description='add noise to audio ')

	parser.add_argument('--indir', type=str, default='audio_test', help='directory or file wich you want to add noise')
	parser.add_argument('--out', type=str, help='path to directory with result noise data')
	parser.add_argument('--ndirs', type=str,  default='bg_noise/FRESOUND_BEEPS_gsm bg_noise/FRESOUND_BEEPS_gsm',
							help='list of directories with noise files space-separated')
	parser.add_argument('--a', type=float, default=0.005, help='noise rate')
	config = parser.parse_args()
	return config


def main():
	config = get_config()
	n = Noisifier(in_dir=config.indir, out_dir=config.out) 
	n.add_noise(noise_dirs=config.ndirs.strip().split(), a=config.a)
if __name__ == '__main__':
	main()
