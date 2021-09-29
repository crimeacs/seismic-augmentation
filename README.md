<!-- ![Logo](logo.svg?raw=true "Logo") -->

<object type="image/svg+xml" data="logo.svg?raw=true"></object>

# seismic-augmentation
Pytorch library for seismic data augmentation

## Setup

`pip install --upgrade git+https://github.com/IMGW-univie/seismic-augmentation.git`

## Usage example

```python
import torch
from seismic_augmentation.composition import Compose
from seismic_augmentation.augmentations import *

aug = Compose([
         FlipChannels(init_channel_order='ZNE'),
         AddRandomNoise(snr_level_db = -10),
         RandomLowPassFilter(cutoff_freq_range=[1,10]),
         RandomHighPassFilter(cutoff_freq_range=[3,14]),
         Taper(max_percentage = 0.5, max_length=10),
         PolarityChange(),
         Normalize()
         ],  
         p=0.5)

transformed = aug(data=waveform, sample_rate=30)
```
## Contribute
Contributors welcome!

## Documentation
For now this library is very simple

```python
FlipChannels(init_channel_order='ZNE')
'''
Swaps N and E channels. Easiest way to change azimuth of a signal

init_channel_order - ordering of the channels of your seismic data
'''
```

```python
AddRandomNoise(snr_level_db=-10)
'''
Adds random noise with desired SNR

snr_level_db - desired signal to noise ratio after augmentation
'''
```

```python
RandomLowPassFilter(cutoff_freq_range=[1,10])
'''
Applies Low Pass Filter with a random cutoff frequency

cutoff_freq_range - range of possible cutoff frequencies
'''
```

```python
RandomHighPassFilter(cutoff_freq_range=[1,10])
'''
Applies High Pass Filter with a random cutoff frequency

cutoff_freq_range - range of possible cutoff frequencies
'''
```

```python
LowPassFilter(cutoff_freq=9.)
'''
Applies Low Pass Filter with a desired cutoff frequency

cutoff_freq - desired cutoff frequency
'''
```
```python
HighPassFilter(cutoff_freq=9.)
'''
Applies High Pass Filter with a desired cutoff frequency

cutoff_freq - desired cutoff frequency
'''
```

```python
Taper(max_percentage=0.5, max_length=10)
'''
Applies a taper with specified parameters

max_percentage - how strongly the signal is suppresed
max_length - maximum length of a taper in samples
'''
```

```python
PolarityChange()
'''
Flips polarity of the signal
'''
```

```python
Normalize()
'''
Global normalization of 3-channel signal
'''
```

`p` - probability that an augmentation would be applied

## Inspiration
Highly inspired by [Facebook Augly](https://github.com/facebookresearch/AugLy)
