import argparse
import os

from models.dataloader import Dataload
from models.autoencoder import AE, AETranner


def get_config():
    parser = argparse.ArgumentParser(description='get fbank and mfcc features from audio files')

    parser.add_argument('--clean_feat_path', type=str, default='clean_data_feat.csv', help='path to clean feat csv')
    parser.add_argument('--noise_feat_path', type=str, default='noise_data_feat.csv', help='path to noise feat csv')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='count of epochs')
    parser.add_argument('--lr', type=int, default=1e-6, help='learning rate')
    parser.add_argument('--log_dir', type=str, default='logs', help='log dir for loss')
    parser.add_argument('--model_dir', type=str, default='ckpt', help='dir for saving model weights')



    config = parser.parse_args()

    return config


def main():
    config = get_config()

    loader = Dataload()
    ntrain, ctarin, ntest, ctest = loader.autoen_train_test(path_to_clean=config.clean_feat_path,
                                                            path_to_noise=config.noise_feat_path,
                                                            batch_size=config.batch_size, )

    model = AE()
    trainer = AETranner(model, model_dir=config.model_dir, log_dir=config.log_dir)
    trainer.train(list(zip(ntrain, ctarin)), lr=config.lr, n_epoch=config.epochs)
    model.eval()
    trainer.test((ntest, ctest))
if __name__ == '__main__':
    main()
