import os
import sys
import fnmatch 
import numpy as np
import librosa
import soundfile as sf


def read_audio(file_path, duration=None, sr=16000):
    if file_path.endswith('.wav'):
        return librosa.core.load(file_path, sr, duration=duration)
    elif file_path.endswith('flac'):
        return sf.read(file_path)
    else:
        raise ValueError(file_path + ': invalide format') 
    

def load_audio(file_path, size, sr=16000):
    duration = np.ceil(size / sr) 
    data, _ = read_audio(file_path, duration, sr)
    if len(data) > size:
        data = data[:size]
    else:
        data = np.pad(data, pad_width=(0, max(0, size - len(data))), mode="constant")
    return data
    
def write_audio(file_path, data, sr=16000):
    if file_path.endswith('.wav'):
        librosa.output.write_wav(file_path, data, sr)
    elif file_path.endswith('flac'):
        sf.write(file_path, data, sr, format='flac')
    else:
      raise ValueError(file_path + ': invalide format')  
        

class Noisifier:
    
    def __init__(self, in_dir='.', out_dir=None):
        self.files = []
        self.noise = []
        self.in_dir = in_dir
        self.files = self.add_audio_files(self.in_dir)
        
        if out_dir is None:
            base_name = os.path.basename(self.in_dir)
            self.out_dir = os.path.join(os.path.dirname(self.in_dir), base_name + '_noise')
            os.mkdir(self.out_dir)
            
        
    def add_audio_files(self, path):
        res_list = []
        if os.path.isfile(path):
            self.files.append(path)
        elif os.path.isdir(path):
            for base, dirs, files in os.walk(path):
                results = fnmatch.filter(files, '*.wav') + fnmatch.filter(files, '*.flac')
                res_list.extend([os.path.join(base, f) for f in files])
        else:
            raise ValueError('Path needs to point to an existing file/folder')
            
        return res_list
  
    def add_noise(self, noise_dirs, a=0.005):
        for dir in noise_dirs:
            self.noise.append(self.add_audio_files(dir))
        
        for f in self.files:
            data, sr = read_audio(f,sr=None)
            for dir in self.noise:
                indx = np.random.randint(len(dir))
                data_noise = load_audio(dir[indx],len(data), sr)
                data = data + a * data_noise
            basename = os.path.basename(f)
            write_audio(os.path.join(self.out_dir, basename), data, sr)

