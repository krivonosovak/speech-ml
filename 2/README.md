
# RNN classifier laugh in audio files


Train on Vocalization corpus that you can get from [here](http://www.dcs.gla.ac.uk/vincia/?p=378)


### Usage


#### Feature extraction

Fbank and MFCC features are extracted by using librosa 

Feature are saved to `fbank_feature.csv` and `mfcc_features.csv` files. Labels are saved to `labels_val.csv` file
```bash
python feature_extract.py —-corpus vocalizationcorpus —-feature_path <dir_save_features>
```

#### Train

Train RNN network:
```bash
python train.py —-corpus vocalizationcorpus —-log logs --epochs 10 —-batch_size 20
```
If feature already have been extracted you can add `-—load True —-feature_path <your_feature path>` for example 

#### Classify 

```bash
python predict.py —corpus vocalizationcorpus -path_to_wavfile some.wav
```