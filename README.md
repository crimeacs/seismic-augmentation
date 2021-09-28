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

transformed = aug(data=test_data, sample_rate=30)
```
## Contribute
Contributors welcome!
