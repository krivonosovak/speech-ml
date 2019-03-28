import os
from os.path import join
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import librosa
import torch
from  laughter_classification.utils import chunks, in_any, time_to_num, interv_to_range, get_sname, most
import re

class DataLoader:
    
    """
    Class for merging labels and features to batch for traning
    """
    
    def __init__(self, corpus_root):
        
        self.corpus_root = corpus_root
        self.sampler = LabelDataSampler(self.corpus_root)
        self.extractor = Extractor(os.path.join(corpus_root, 'data'))


    
    def get_data(self, frame_sec = 0.03, batch_size=64, test=0.2, feature_path=os.getcwd(),
                    load=False, naudio=None, nfbank=128, nmfcc=13):
        
        """
        Returns train and test dataset in the following format: (fbank_features, mfcc_features, labels)
        :param frame_sec: float, length of each frame in sec
        :param batch_size: int, size of batch
        :param test: float, test ratio
        :param load: boolean, if True features are loaded from files
        :param nfbank: int, fbank feature count
        :param nmfcc: int, mfcc feature count 
        :param naudio: int, number of audios to parse, if not defined parses all
        """	
        
    
        labels = self.sampler.create_sampled_df(frame_sec, naudio=naudio, feature_dir=feature_path, load=load)
        fbank, mfcc = self.extractor.extract_features(frame_sec, feature_dir=feature_path,
                                naudio=naudio, load=load, nfbank=nfbank, nmfcc=nmfcc)

        labels.sort_values(['SNAME', 'NO_FRAME'])
        mfcc.sort_values(['SNAME', 'NO_FRAME'])
        fbank.sort_values(['SNAME', 'NO_FRAME'])
        
        pad = int(np.ceil(len(labels) / batch_size) *  batch_size - len(labels))
    
        labels = np.pad(np.array(labels['IS_LAUGHTER']), (0, pad), mode='constant')
        fbank = np.pad(fbank.loc[:,~fbank.columns.isin(['SNAME', 'NO_FRAME'])].values, ((0,pad),(0,0)), mode='constant')
        mfcc = np.pad(mfcc.loc[:, ~mfcc.columns.isin(['SNAME', 'NO_FRAME']) ].values, ((0,pad),(0,0)), mode='constant')
    

        labels = torch.from_numpy(np.reshape(labels, (-1, batch_size)))
        fbank = torch.from_numpy(np.reshape(fbank, (-1, 1, batch_size, nfbank)))
        mfcc = torch.from_numpy(np.reshape(mfcc, (-1, 1, batch_size, nmfcc*3)))
        count_batch = len(labels)
        train = int((1 - test) * count_batch)
        
        return list(zip(fbank[:train], mfcc[:train], labels[:train])),\
               (fbank[train:], mfcc[train:], labels[train:])
        

class LabelDataSampler:
    
    """
    Class for loading and sampling audio data by frames for SSPNet Vocalization Corpus
    """

    @staticmethod
    def read_labels(labels_path):
        def_cols = ['Sample', 'original_spk', 'gender', 'original_time']
        label_cols = ["{}_{}".format(name, ind) for ind in range(6) for name in ('type_voc', 'start_voc', 'end_voc')]
        def_cols.extend(label_cols)
        labels = pd.read_csv(labels_path, names=def_cols, engine='python', skiprows=1)
        return labels

    def __init__(self, corpus_root):
        self.sample_rate = 16000
        self.duration = 11
        self.default_len = self.sample_rate * self.duration
        self.data_dir = join(corpus_root, "data")
        labels_path = join(corpus_root, "labels.txt")
        self.labels = self.read_labels(labels_path)

    @staticmethod
    def _interval_generator(incidents):
        for itype, start, end in chunks(incidents, 3):
            if itype == 'laughter':
                yield start, end

    def get_labels_for_file(self, wav_path, frame_sec=0.03):
        
        sname = get_sname(wav_path)
        sample = self.labels[self.labels.Sample == sname]

        incidents = sample.loc[:, 'type_voc_0':'end_voc_5']
        incidents = incidents.dropna(axis=1, how='all')
        incidents = incidents.values[0]

        rate, audio = wav.read(wav_path)
        
        laughts = self._interval_generator(incidents)
        laughts = [interv_to_range(x, len(audio), self.duration) for x in laughts]
        laught_along = [1 if in_any(t, laughts) else 0 for t, _ in enumerate(audio)]
        
        frame_size = int(self.sample_rate * frame_sec)
        is_laughter = np.array([most(la) for la in chunks(laught_along, frame_size)])

        df = pd.DataFrame({'IS_LAUGHTER': is_laughter,
                           'SNAME': sname, 'NO_FRAME': np.arange(len(is_laughter))})
        return df

    def get_valid_wav_paths(self):
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            fullpaths = [join(dirpath, fn) for fn in filenames]
            return [path for path in fullpaths if len(wav.read(path)[1]) == self.default_len]
    
    def create_sampled_df(self, frame_sec, naudio=None, feature_dir='result', feature_name='', load=False):
        """
        Returns sampled data for whole corpus
        :param frame_sec: int, length of each frame in sec
        :param naudio: int, number of audios to parse, if not defined parses all
        :param feature_dir: string, path to save or load parsed corpus
        :param feature_name: string, prefix in names of feature files
        :param force_save: boolean, if you want to override file with same name
        :param load: boolean, if True features are loaded from files
        :return:
        """
        label_path = os.path.join(feature_dir, feature_name + 'labels_val.csv')
        if load:
            print('load labels from', label_path )
            return pd.read_csv(label_path)
            
        fullpaths = self.get_valid_wav_paths()[:naudio]
        dataframes = [self.get_labels_for_file(wav_path, frame_sec) for wav_path in fullpaths]
        df = pd.concat(dataframes)

        if feature_dir is not None:
            print('saving labels to', label_path)
            df.to_csv(label_path, index=False)

        return df

class Extractor():
    
    """
    Python Audio Analysis features extractor
    """
    def __init__(self, data_dir=os.getcwd()):
        self.sr = 16000
        self.data_dir = data_dir
        self.duration = 11
        self.default_len = self.sr * self.duration


    def get_valid_wav_paths(self):
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            fullpaths = [os.path.join(dirpath, fn) for fn in filenames]
            return [path for path in fullpaths if len(wav.read(path)[1]) == self.default_len]

    def extract_features(self, frame_sec=0.03, naudio=None, feature_dir='result', feature_name='',
                         load=False, nfbank=128, nmfcc=13):
        """
        Returns fbank and mfcc features for whole corpus
        :param frame_sec int, length of each frame in sec
        :param naudio int, number of audios to parse, if not defined parses all
        :param feature_dir string, path to features dir (fbank_features.csv, mfcc_features.csv)
        :param feature_name str, prefix for features file (fbank_features.csv, mfcc_features.csv)
        :param load boolean, if True features are loaded from files
        :param nfbank int, fbank feature count
        :param nmfcc int, mfcc feature count
        """
        fbank_path = os.path.join(feature_dir, feature_name + 'fbank_features.csv')
        mfcc_path = os.path.join(feature_dir, feature_name + 'mfcc_features.csv')
        if load:
            print('load features from ',fbank_path, mfcc_path)
            fbank = pd.read_csv(fbank_path)
            mfcc = pd.read_csv(mfcc_path)
            return fbank, mfcc
                
        fbank = []
        mfcc = []
        fullpaths = self.get_valid_wav_paths()[:naudio]
        for wav_path in fullpaths:
            fb, mf = self.extract_features_from_file(wav_path, frame_sec, nfbank=nfbank, nmfcc=nmfcc)
            fbank.append(fb)
            mfcc.append(mf)
        
        
        fbank = pd.concat(fbank)
        mfcc = pd.concat(mfcc)
        
        if feature_path is not None:
            fbank.to_csv(fbank_path, index=False)
            mfcc.to_csv(mfcc_path, index=False)
            print('save features to {} and {}'.format(fbank_path, mfcc_path))
                
        return fbank, mfcc

    def extract_features_from_file(self, wav_path, frame_sec=0.03, nfbank=128, nmfcc=13):
        """
        Extract fbank and mfcc and delta_mfcc, delta2_mfcc features for audio by given time window
        :param wav_path: path to .wav file
        :param frame_sec: float, sampling frame size in sec
        :param nfbank: int, count of fbank features
        :param nmfcc: int, count of mfcc features
        """
        y, _ = librosa.load(wav_path, sr=self.sr)
        frame = int(frame_sec * self.sr)
        last = len(y) // frame * frame
        S = librosa.feature.melspectrogram(y=y[:last], sr=self.sr, n_mels=nfbank, n_fft=frame, hop_length=frame)

        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=nmfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        sname = get_sname(wav_path)
        df = pd.DataFrame({'SNAME': sname, 'NO_FRAME': np.arange(log_S.shape[1])})
        return pd.concat([df, pd.DataFrame(log_S.T)], axis=1),  \
               pd.concat([df, pd.DataFrame(np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0).T)], axis=1)
    
