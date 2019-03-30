import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score


class Classifier(nn.Module):

    def __init__(self, in_dim=39, num_classes=109, batch_size=64):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = 64
        self.batch_size = batch_size
        self.rnn1 = nn.LSTM(
            input_size=in_dim,
            hidden_size=self.hidden_dim,
            num_layers=3,
            bidirectional=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.BatchNorm1d(self.batch_size),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_classes)
        )
        self.fc1 = nn.Linear(self.hidden_dim * 2, 128)
        self.out =  nn.Linear(128, self.num_classes)



    def forward(self, feat):
        r_out, (h_n1, h_c1) = self.rnn1(feat.type(torch.FloatTensor), None)
        r_out = self.fc(r_out)
        return r_out


class  TrainerClass():

    def __init__(self, model, device='cpu', model_dir='ckpt_class', log_dir='logs_class'):
        self.device = device
        self.model_dir = model_dir
        self.model = model
        self.model.to(self.device)
        self.writer = SummaryWriter(log_dir=log_dir)


    def save(self, epoch, train_name):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'model_{train_name}_{epoch}.pt'))


    def train(self, train_loader, lr=1e-6, n_epoch=25, train_name='', loss_point_every=5):
        """
        tra
        :param train_loader: traind dataset (train, label)
        :param lr: float, learning rate
        :param n_epoch: count of epochs
        :param train_name: str, suffix for model name - model_<train_name>_<epoch>.pt
        :param loss_point_every: int,  save loss every

        """
        print('training model...')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        global_step = 0
        for epoch in range(n_epoch):

            for i, data in enumerate(train_loader):
                feat, labels = data
                self.model.zero_grad()
                feat = feat.to(self.device).float().unsqueeze(0)

                output = self.model(feat)
                loss = criterion(output.squeeze(), labels.long())

                if global_step % loss_point_every == 0:
                    print(f'epoch: [{epoch + 1}/{n_epoch}] iter: [{i}] loss: {loss:.4f}')
                    self.writer.add_scalar('res_loss', loss, global_step)
                global_step += 1
                loss.backward(retain_graph=True)
                optimizer.step()

            self.save(epoch, train_name)


    def test(self, test_loader):
        """

        :param test_loader:
        :return:
        """
        print('testing model...')
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        noise, target = test_loader
        noise = noise.to(self.device).float()
        target = target.to(self.device)

        output = self.model(noise)
        res =  torch.argmax(output, 2).view(-1)
        target = target.view(-1)
        loss = torch.sum(res == target.long()).item()/target.size()[0]

        print(f'accurancy: {loss:.4f}')
