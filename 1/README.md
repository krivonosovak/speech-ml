## Script for adding background noise.

* Support .wav and .flac formats


## Usage

based on the noisifier.noisifier. Noisifier class. Here is a short example:
```python

from maracas.dataset import Dataset

n = Noisifier(in_dir='audio_test', out_dir='audio_test_noise') 
# a - noise rate
n.add_noise(noise_dirs=['bg_noise/FRESOUND_BEEPS_gsm', 'bg_noise/FRESOUND_BEEPS_gsm'], a=0.005)

```

terminal:
```bash
python add_noise.py —-indir audio_test
			    --ndirs bg_noise/FRESOUND_BEEPS_gsm bg_noise/FRESOUND_BEEPS_gsm
			    —-a 0.005 —-out audio_test_noise
```
	
