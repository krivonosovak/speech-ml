import torch.nn as nn
import torch.nn.functional as F
import torch 
import os
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score


	
class TwoLossRNN(nn.Module):
	def __init__(self,  insize_1=13*3, insize_2=128):
		
		super(TwoLossRNN, self).__init__()
		self.insize_1 = insize_1
		self.insize_2 = insize_2
		self.hidden_dim = 64
		self.batch_size = 3
		self.rnn1 = nn.LSTM(
			input_size=insize_1,
			hidden_size=self.hidden_dim,
			num_layers=3,
			bidirectional=False,
			dropout=0.2
		)

		self.rnn2 = nn.LSTM(
			input_size= insize_2,
			hidden_size=self.hidden_dim,
			num_layers=3,
			dropout=0.2,
			bidirectional=False
		)

		self.fc1 = nn.Linear(self.hidden_dim, 32)
		self.out1 = nn.Linear(32, 1)
		self.fc2 = nn.Linear(self.hidden_dim * 2, 32)
		self.out2 = nn.Linear(32, 1)
		self.hidden = self.init_hidden()


	def init_hidden(self):
		return (torch.zeros(1, self.batch_size, self.hidden_dim),
				torch.zeros(1, self.batch_size, self.hidden_dim))

	def forward(self, fbank, mfcc):
		# x shape (batch, time_step, input_size)

		r_out1, (h_n1, h_c1) = self.rnn1(mfcc.type(torch.FloatTensor), None)   # None represents zero initial hidden state
		r_out2, (h_n2, h_c2) = self.rnn2(fbank.type(torch.FloatTensor), None)


		r_out2 = torch.cat((r_out1, r_out2), dim=2)

		r_out1 = F.relu(self.fc1(r_out1))
		r_out1 = torch.sigmoid(self.out1(r_out1))

		r_out2 = F.relu(self.fc2(r_out2))
		r_out2 = torch.sigmoid(self.out2(r_out2))
		return r_out1[0], r_out2[0]
		


class TranerTwoLossRNN:

	def __init__(self, model, device='cpu', model_dir='ckpt', log_dir='log'):
		self.model = model
		self.model_dir = model_dir
		self.log_dir = log_dir
		self.device = device
		self.model.to(self.device)
		self.writer = SummaryWriter(log_dir=log_dir)

	def save(self, epoch):
		os.makedirs(self.model_dir, exist_ok=True)
		torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'model_3_{epoch}.pt'))


	def train(self, train_data, test_data, epochs=30, lr = 1e-5, loss_every=10):

		test_fbank, test_mfcc, test_labels = test_data
		test_mfcc = test_mfcc.to(self.device).float().view(1, -1, 13*3)
		test_fbank = test_fbank.to(self.device).float().view(1, -1, 128)
		test_labels = test_labels.to(self.device).int().view(-1).numpy()

		optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		global_step = 0
		loss = nn.BCELoss()
		for epoch in range(epochs):
			for i, (fbanc, mfcc, labels) in enumerate(train_data):
				fbanc = fbanc.to(self.device)
				mfcc = mfcc.to(self.device)
				labels = labels.to(self.device)

				optimizer.zero_grad()
				self.model.hidden = self.model.init_hidden()
				out1, out2  = self.model(fbanc, mfcc)
				mfcc_loss = loss(out1.view(len(labels)), labels.float())
				fbank_loss = loss(out2.view(len(labels)), labels.float())
				big_loss = 0.2 * fbank_loss + mfcc_loss
				if global_step % loss_every == 0:
					print(f'{epoch}:{global_step} loss mfcc {mfcc_loss}, combine {big_loss}')
					self.writer.add_scalar('loss_train/mfcc_loss_3', mfcc_loss, global_step, epoch)
					self.writer.add_scalar('loss_train/general_loss_3', big_loss, global_step, epoch)



				global_step += 1
				big_loss.backward()
				optimizer.step()

			self.save(epoch)
			self.model.eval()

			# test auc value
			test_out1, test_out2 = self.model(test_fbank, test_mfcc)
			auc1 = roc_auc_score(test_labels, test_out1.detach().numpy())
			auc2 = roc_auc_score(test_labels, test_out2.detach().numpy())
			auc3 = roc_auc_score(test_labels, (test_out1.detach().numpy() + test_out2.detach().numpy()) / 2)
			self.writer.add_scalar('auc/mfcc_3', auc1, global_step, epoch)
			self.writer.add_scalar('auc/general_3', auc2, global_step, epoch)
			self.writer.add_scalar('auc/mean_3', auc3, global_step, epoch)
			print(f'{epoch}:{global_step} auc1 {auc1}, auc2 {auc2}, auc3 {auc3}')
			self.model.train()


	

