
## Denoising Autoencoder for audio files and LSTM classifier for speaker identification


Train on VCTK-Corpus corpus that you can get from [here](http://www.dcs.gla.ac.uk/vincia/?p=378)




### Usage


#### Feature extraction

For extracting features, kaldi is used. 
Instruction for downloading and installation kaldi you can get [here](http://kaldi-asr.org) 


Feature are saved to `<dir_name>_feat.csv`

```
python extract_features.py —-clean_dir clean_data —-noise_dir noise_data --kaldi_path kaldi
```

#### Train

Train Autoencoder network:
```
python train_autoencoder.py —-clean_feat_path clean_data_feat.csv noise_feat_path noise_data_feat.csv —-log_dir logs model_dir ckpt —epochs 4 —-batch_size 50
```

Train Classifier network:
```
python train_classifier.py —-feat_path clean_data_feat.csv  —-log_dir logs model_dir ckpt —epochs 4 —-batch_size 50
```

#### Classify 

```
python classification.py —feat_path clean_data_feat.csv —class_model_path <path_to classification model weights> —ae_model_path <path_to autoencoder model weights>
```

