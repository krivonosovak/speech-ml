import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from tensorboardX import SummaryWriter



class AE(nn.Module):
    """
    Denoising autoencoder
    """
    def __init__(self, in_dim=39, botleneck_dim=20):
        super(AE, self).__init__()
        self.in_dim = in_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 20),
            nn.ReLU(True),
            nn.Linear(20, botleneck_dim))
        self.decoder = nn.Sequential(
            nn.Linear(botleneck_dim, 20),
            nn.ReLU(True),
            nn.Linear(20, 32),
            nn.ReLU(True),
            nn.Linear(32, in_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AETranner:

    """
    Trainner for AE
    """

    def __init__(self, model, device='cpu', model_dir='ckpt', log_dir='logs'):
        self.device = device
        self.model_dir = model_dir
        self.model = model
        self.model.to(self.device)
        self.writer = SummaryWriter(log_dir=log_dir)

    def save(self, epoch, train_name):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'model{train_name}_{epoch}.pt'))

    def train(self, train_loader, lr=1e-4, n_epoch=25, train_name=None,  loss_point_every=5):
        """
        train AE
        :param train_loader: train data (noise, clean)
        :param lr: float, learning rate
        :param n_epoch: int, count of epochs
        :param train_name: str, suffix for model name - model_<train_name>_<epoch>.pt
        :param loss_point_every: int,  save loss every

        """
        print('training model...')
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr)

        global_step = 0
        for epoch in range(n_epoch):

            for i, data in enumerate(train_loader):
                noise, target = data
                self.model.zero_grad()
                noise = noise.to(self.device).float()
                target = target.to(self.device).float()

                output = self.model(noise)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()
                if global_step % loss_point_every == 0:
                    print(f'epoch: [{epoch + 1}/{n_epoch}] iter: [{i}] loss: {loss:.4f}')
                    self.writer.add_scalar('res_loss', loss, global_step)
                global_step += 1

            self.save(epoch, train_name)

    def test(self, test_loader):
        """
        test AE
        :param test_loader:  test data (noise, clean)
        """
        print('testing model...')
        criterion = nn.MSELoss()

        noise, target = test_loader
        noise = noise.view(-1, 39)
        target = target.view(-1, 39)
        noise = noise.to(self.device).float()
        target = target.to(self.device).float()

        output = self.model(noise)
        loss = criterion(output, target)

        print(f'test_loss: {loss:.4f}')

