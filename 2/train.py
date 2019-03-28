import argparse
import logging
import os

import torch
from laughter_classification.data_extract import DataLoader
from laughter_classification.rnn import TranerTwoLossRNN, TwoLossRNN




def get_config():
	

	parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')

	parser.add_argument('--log', type=str, default='logs')
	parser.add_argument('--corpus', type=str, default='vocalizationcorpus')
	parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--frame_sec', type=float, default=0.03, help='length of window in sec')
	parser.add_argument('--feature_path', type=str, default=os.getcwd(), help='path to feature dir')
	parser.add_argument('--nfbank', type=int, default=128, help='fbank feature count')
	parser.add_argument('--load', type=bool, default=True, help='If True feature are loaded from feature_path directory')
	parser.add_argument('--nmfcc', type=int, default=13, help='mfcc feature count')
	parser.add_argument('--naudio', type=int, default=None, help='number of audios to parse, if not defined parses all')
	parser.add_argument('--model_dir', type=str, default='ckpt', help='dir for saving model weights')


	config = parser.parse_args()
#    config.cuda = not config.no_cuda and torch.cuda.is_available()

	return config


def main():
	
	
	config = get_config()
	print('batch_size {} epochs {} frame_sec {}'.format(config.batch_size, config.epochs, config.frame_sec ))


	data = DataLoader(corpus_root=config.corpus)
	train, test = data.get_data(frame_sec = config.frame_sec, batch_size=config.batch_size,
								feature_path=config.feature_path, load=config.load,
								naudio=config.naudio, nfbank=config.nfbank, nmfcc=config.nmfcc)
	model = TwoLossRNN(insize_1=config.nmfcc*3, insize_2=config.nfbank)
	trainer = TranerTwoLossRNN(model,  model_dir=config.model_dir, log_dir=config.log)
	trainer.train(train, test, epochs=config.epochs)

	
if __name__ == '__main__':
	main()	
