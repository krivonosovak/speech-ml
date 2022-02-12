import argparse
import torch
import numpy as np
from laughter_classification.predictors import RnnPredictor
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from laughter_classification.data_extract import LabelDataSampler
from laughter_classification.data_extract import Extractor



def get_config():
	

	parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')
	parser.add_argument('--model_path', type=str)
	parser.add_argument('--path_to_wavfile', type=str)
	parser.add_argument('--corpus_path', type=str, default='vocalizationcorpus')
	parser.add_argument('--result_path', type=str, default='result.csv')

	config = parser.parse_args()
	return config


def main():

	config = get_config()

	labels_extr = LabelDataSampler(corpus_root=config.corpus_path)
	df_labels = labels_extr.get_labels_for_file(config.path_to_wavfile)
	labels = np.array(df_labels['IS_LAUGHTER'])

	extr = Extractor()
	fbank, mfcc = extr.extract_features_from_file(config.path_to_wavfile)
	fbank = torch.from_numpy(fbank.iloc[:, 2:].values).float().unsqueeze(0)
	mfcc = torch.from_numpy(mfcc.iloc[:, 2:].values).float().unsqueeze(0)

	predictor = RnnPredictor(config.model_path)
	prob = predictor.predict_proba(fbank, mfcc)
	pred_labels = predictor.predict(fbank, mfcc)

	print('accuracy: {}'.format(metrics.accuracy_score(labels,pred_labels)))
	print('auc: {}'.format(roc_auc_score(labels, prob)))

	df_labels['LABEL'] = pred_labels
	df_labels.iloc[:, 1:].to_csv(config.result_path, index=False)

if __name__ == '__main__':
	main()
