import argparse
import os
import torch
from models.dataloader import Dataload
from models.classifier import Classifier, TrainerClass
from models.autoencoder import AE


def get_config():
    parser = argparse.ArgumentParser(description='get fbank and mfcc features from audio files')

    parser.add_argument('--feat_path', type=str, default='test/noise_data_feat.csv', help='path to features csv')
    parser.add_argument('--class_model_path', type=str, default='ckpt_class/model__3.pt',
                        help='path to classifier train model weights')
    parser.add_argument('--ae_model_path', type=str, default='ckpt/model1_2_9.pt',
                        help='path to autoencoder train model weights')
    parser.add_argument('--batch_size', type=int, default=50)

    config = parser.parse_args()

    return config


def main():
    config = get_config()

    loader = Dataload()
    train, label_train, test, label_test = loader.class_train_test(path_feat_data=config.feat_path,
                                                                   batch_size=config.batch_size,
                                                                   train_ratio=1,
                                                                   shuffle=True)


    ae = AE(13*3)
    ae.load_state_dict(torch.load(config.ae_model_path))
    ae.eval()

    classifier = Classifier(13*3, num_classes=109, batch_size=50)
    classifier.load_state_dict(torch.load(config.class_model_path))
    classifier.eval()

    trainer = TrainerClass(classifier)
    unnoise_data = ae(train)
    trainer.test((unnoise_data, label_train))

if __name__ == '__main__':
    main()
