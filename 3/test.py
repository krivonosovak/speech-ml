import os
import pandas as pd
from models.dataloader import KaldiExtractor

DIR_NAME = 'clean_data'
DIR_NOISE_NAME = 'noise_data'
#
# DIR_NAME = 'test/clean_data'
# DIR_NOISE_NAME = 'test/noise_data'

# n = Noisifier(in_dir=DIR_NAME, out_dir=DIR_NOISE_NAME)
# n.add_noise(noise_dirs=['freesound_background_gsm'], a=0.01)


# data, _ = librosa.core.load('clean_data/p233_012.wav', 48000)
# sf.write('p233_012.wav', data, 48000, format='wav', subtype='PCM_16')
# FEATURE_DIRECTORY = 'feature_res'
CUR_DIR = os.getcwd()
# extr = KaldiExtractor()
# extr.extract(dir_in='test/clean_data')
# extr.extract(dir_in='test/noise_data')


NUM_CLASSES = 376

#
# model = AE(13*3)
# model.load_state_dict(torch.load('ckpt/model41_9.pt'))
# model.eval()
extr = KaldiExtractor()
print('read csv...')
df_feat_data = pd.read_csv('test/noise_data_feat.csv')
train_feat, train_labels, test_feat, test_label = extr.class_train_test(df_feat_data, shuffle=True, train_ratio=1)

# unnoise = model(train_feat)
# print(unnoise.shape)
# clasifier = Classifier(13*3, num_classes=376)
# class_trainer = TrainerClass(clasifier)
# class_trainer.train(list(zip(unnoise, train_labels)))

# for i in range(3):
#     train, test = extr.class_train_test('test/clean_data_feat.csv',
#                                                                'test/noise_data_feat.csv',
#                                                                train_ratio=1)
#     trainer.train(train, n_epoch=10, train_name=str(i) + '_2')

# print(df[0].shape, df[1].shape)
# name = 'noise_data_feat4'
# df = pd.read_csv('noise_data_feat4.csv')
# l = len(df)
# cur = 0
# for i, p in enumerate(np.arange(0.5,1.1,0.5)):
#     nex = int(p * l)
#     print(cur, nex)
#     df_part = df.loc[cur:nex,:].to_csv(name + str(i+1) + '.csv', index=False)
#     cur = nex
