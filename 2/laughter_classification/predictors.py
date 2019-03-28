import numpy as np
import torch

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from laughter_classification.rnn import TwoLossRNN


class Predictor:
    """
    Wrapper class used for loading serialized model and
    using it in classification task.
    Defines unified interface for all inherited predictors.
    """

    def predict(self, X):
        """
        Predict target values of X given a model

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array predicted classes
        """
        raise NotImplementedError("Should have implemented this")

    def predict_proba(self, X):
        """
        Predict probabilities of target class

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array target class probabilities
        """
        raise NotImplementedError("Should have implemented this")


class RnnPredictor(Predictor):

    def __init__(self, path_to_wav, model_path='ckpt/model_2_14.pt'):
        self.model = TwoLossRNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self,fbank,  mfcc):
        return np.round(self.predict_proba(fbank, mfcc))
        

    def predict_proba(self, fbank, mfcc ):
        out1, out2 = self.model(fbank, mfcc)
        return np.reshape(out1.detach().numpy(), -1)
        