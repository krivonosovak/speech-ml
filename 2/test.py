from laughter_classification.data_extract import DataLoader
import os
from laughter_classification.rnn import TwoLossRNN, TranerTwoLossRNN
import torch

CORPUS_PATH = 'vocalizationcorpus_1'


data = DataLoader(CORPUS_PATH)
train, test = data.get_data(frame_sec=0.03, naudio=None,
                       feature_path=os.getcwd(), load=True, nfbank=128, nmfcc=13)
model = TwoLossRNN()
model.load_state_dict(torch.load('ckpt/model_2_14.pt'))
trainer = TranerTwoLossRNN(model)
trainer.train(train, test, epochs=15)