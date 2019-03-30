import argparse
import os

from models.dataloader import Dataload
from models.classifier import Classifier, TrainerClass


def get_config():
    parser = argparse.ArgumentParser(description='get fbank and mfcc features from audio files')

    parser.add_argument('--feat_path', type=str, default='clean_data_feat.csv', help='path to features csv')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--epochs', type=int, default=4, help='count of epochs')
    parser.add_argument('--lr', type=int, default=1e-6, help='learning rate')
    parser.add_argument('--log_dir', type=str, default='class_logs', help='log dir for loss')
    parser.add_argument('--model_dir', type=str, default='class_ckpt', help='dir for saving model weights')

    config = parser.parse_args()

    return config


def main():
    config = get_config()

    loader = Dataload()
    train, label_train, test, label_test = loader.class_train_test(path_feat_data=config.feat_path,
                                                                   batch_size=config.batch_size,
                                                                   shuffle=True)

    model = Classifier(batch_size=config.batch_size)
    trainer = TrainerClass(model, model_dir=config.model_dir, log_dir=config.log_dir)
    trainer.train(list(zip(train, label_train)), lr=config.lr, n_epoch=config.epochs)
    model.eval()
    trainer.test((test, label_test))
if __name__ == '__main__':
    main()
