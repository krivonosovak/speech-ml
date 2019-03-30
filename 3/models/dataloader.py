import glob
import os
from os.path import join
import numpy as np
import pandas as pd
import kaldi_io as kl
import torch


class KaldiExtractor:
    """
    Class for extractraction mfcc features from wav files
    """

    def __init__(self, kaldipath='kaldi'):
        # read all wav files from dir_in directory
        self.CUR_PATH = os.getcwd()
        self.KALDI_PATH = kaldipath


    def write_scp_file(self, dir_in):
        """
        Create scp file for kaldi extraction in format:
        <base-name> <path_to_wav1>
        ...

        Helper-method for extract
            :param  dir_in str,  path for save resukt
            :return name of scp file
        """
        scp_name = os.path.basename(dir_in) + '_wav.scp'
        with open(scp_name, 'w') as scp_file:
            all_files = glob.glob(join(self.CUR_PATH, dir_in, "*.wav"))
            for f in all_files:
                base_name = os.path.basename(f)
                base_name = base_name.split('.')[0]
                print("{} {}".format(base_name, f), file=scp_file)
        return scp_name

    def read_feat(self, feat_path):
        """
        Parse ark-kaldi features result file, helper-method for extract
             :param feat_path str,  path to .ark file with result features
             :return dataframe: features:39, labels:name of files,
                                no_frame: no for frame in file, speaker: label of speaker
        """

        res = []
        for (label, feat) in kl.read_mat_ark(feat_path):
            arr = np.array(feat)
            df = pd.DataFrame(arr)
            df.set_axis(['F' + str(i + 1) for i in range(arr.shape[1])], axis='columns', inplace=True)
            df['labels'] = np.repeat(label, arr.shape[0])
            df['no_frame'] = np.arange(arr.shape[0])
            df['speaker'] = np.repeat(label.split('_')[0], arr.shape[0])
            res.append(df)

        res_df = pd.concat(res)
        return res_df


    def extract(self, dir_in, dir_out=None, sf=48000):
        """
        Extract mfcc + delta-mfcc + delta2-mfcc features from wav files in dir_in directory
        result feature file: <dir_in>_feat.csv
            :param dir_in str,  path to dir with wav files
            :param dir_out str, path to result features dir
            :param sf int, sample frequency
            :return dataframe with features: features:39, labels:name of files,
                                       no_frame: no for frame in file, speaker: label of speaker
        """

        scp_name = self.write_scp_file(dir_in)
        # scp_name = "my.scp"
        if dir_out is None:
            dir_out = os.path.dirname(dir_in)


        path_to_res = join(dir_out, os.path.basename(dir_in) + '_feat.ark')
        path_to_res_delta = join(dir_out, os.path.basename(dir_in) + '_delta_feat.ark')
        new_name = join(dir_out, os.path.basename(dir_in) + '_feat.csv')

        if os.path.isfile(new_name):
            print('reading features from exist file {}'.format(new_name))
            return pd.read_csv(new_name)
        print('extracting features...')
        os.system('{} --sample-frequency={} scp:{} ark:{}'.format(join(self.KALDI_PATH, 'src/featbin/compute-mfcc-feats'),
                                                                        sf, scp_name, path_to_res))
        os.system('{} --delta-order=2 ark:{} ark:{}'.format(join(self.KALDI_PATH, 'src/featbin/add-deltas'),
                                                            path_to_res, path_to_res_delta))

        delta_mfcc_feat = self.read_feat(path_to_res_delta)
        print('saving features...')
        delta_mfcc_feat.to_csv(new_name, index=False)
        return delta_mfcc_feat

class Dataload:

    """
    Load feature data and split it to train-test datasets
    """

    def add_pad(self, data_df, batch_size=64, count_feat=39):

        spkrs, _ = pd.factorize(data_df['speaker'])
        spkrs = torch.from_numpy(np.array(spkrs))
        un_labels = torch.unique(spkrs)
        all_data = torch.from_numpy(data_df.iloc[:, :count_feat].values)

        res = torch.cat((spkrs.unsqueeze(1).float(), all_data.float()), 1)
        add_row = 0
        for i, label in enumerate(un_labels):
            l = torch.sum(spkrs == label)
            add_row = add_row + int(np.ceil(l.item() / batch_size) * batch_size - l.item())
        pad = torch.zeros(add_row, count_feat + 1)
        cur = 0
        nex = 0
        for label in un_labels:
            indx = spkrs == label
            l = torch.sum(indx)
            nex = cur + int(np.ceil(l.item() / batch_size) * batch_size - l.item())
            pad[cur:nex, 0] = label
            cur = nex
        res = torch.cat((res, pad.float()), 0)
        res, _ = torch.sort(res, dim=0)
        return res[:, 1:], res[:, 1]

    def get_shufle_index(self, all_len, train_ratio= 0.8):
        indx = np.arange(all_len)
        np.random.shuffle(indx)
        train_indx = indx[:int(train_ratio * all_len)]
        test_mask = np.ones(all_len, dtype=bool)
        test_mask[train_indx] = False
        test_indx = indx[test_mask]
        return train_indx, test_indx

    def get_index(self, all_len, train_ratio= 0.8):

        indx = np.arange(all_len)
        train_indx = indx[:int(train_ratio * all_len)]
        test_indx = indx[int(train_ratio * all_len):]
        return train_indx, test_indx

    def autoen_train_test(self, path_to_clean, path_to_noise, shuffle=False, train_ratio=0.8,
                          batch_size=64, feat_size=13*3):
        """
        Split data to train test-datasets for classifier (mfcc noise features, mfcc clean features)
        :param path_to_clean: str, path to clean_feature csv
        :param path_to_noise: str, path to noise_feature csv
        :param shuffle: bool, shuffle data in True
        :param train_ratio: ratio of train data
        :param batch_size: batch size
        :param feat_size: count of features
        :return: noise_train, clean_train, noise_test, noise_train
        """
        print('geting train and test datasets...')
        df_clean_data = pd.read_csv(path_to_clean)
        df_noise_data = pd.read_csv(path_to_noise)

        noise_add, labels = self.add_pad(df_noise_data, count_feat=feat_size, batch_size=batch_size)
        clean_add, labels = self.add_pad(df_clean_data, count_feat=feat_size, batch_size=batch_size)

        noise_add = noise_add.view(-1, batch_size, feat_size)
        clean_add = clean_add.view(-1, batch_size, feat_size)
        labels =  labels.view(-1,batch_size)

        bcount = clean_add.size()[0]
        if shuffle:
            train_indx, test_indx = self.get_shufle_index(bcount, train_ratio=train_ratio)
        else:
            train_indx, test_indx = self.get_index(bcount, train_ratio=train_ratio)

        return noise_add[train_indx, :], clean_add[train_indx, :], \
               noise_add[test_indx, :], clean_add[test_indx, :]


    def class_train_test(self, path_feat_data,  shuffle=False, train_ratio=0.8,
                         batch_size=64, feat_size=13*3):
        """
        Split data to train test-datasets for classifier (mfcc_feature, speakers labels)
        :param path_feat_data: str,  path to features data .csv
        :param shuffle: bool, if True shuffle datasets
        :param train_ratio: float, ratio of train data
        :param batch_size: int, batch size
        :param feat_size: int, number of features
        :return: train, train_label, test, test_label
        """
        df_feat_data = pd.read_csv(path_feat_data)

        feat_add, labels = self.add_pad(df_feat_data, count_feat=feat_size, batch_size=batch_size)

        labels = labels.view(-1, batch_size)
        feat_add = feat_add.view(-1,batch_size, feat_size)

        bcount = feat_add.size()[0]
        if shuffle:
            train_indx, test_indx = self.get_shufle_index(bcount, train_ratio=train_ratio)
        else:
            train_indx, test_indx = self.get_index(bcount, train_ratio=train_ratio)


        return feat_add[train_indx, :] , labels[train_indx, :], \
               feat_add[test_indx, :], labels[test_indx, :]







