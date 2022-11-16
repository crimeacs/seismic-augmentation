import random
import julius
import torch

class BaseAugmentation(object):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"
        self.p = p

    def __call__(
            self,
            data: torch.tensor,
            sample_rate: int = 100,
    ):
        """
        @param data: the waveform array to be augmented
        @param sample_rate: the waveform sample rate of the inputted waveform
        @returns: the augmented waveform array
        """
        assert isinstance(data, torch.Tensor), "Waveform passed in must be a torch.tensor"

        if random.random() > self.p:
            return data

        return self.apply_transform(data, sample_rate)

    def apply_transform(
            self,
            data: torch.tensor,
            sample_rate: int = 100,
    ):
        """
        This function is to be implemented in the child classes.
        From this function, call the augmentation function with the
        parameters specified
        """
        raise NotImplementedError()


class FlipChannels(BaseAugmentation):
    def __init__(self,
                 init_channel_order: str = 'ZNE',

                 p: float = 1.0):
        super().__init__(p)

        assert init_channel_order == 'ZNE', 'Please convert your channel order to ZNE first'

    def apply_transform(self,
                        data: torch.tensor,
                        sample_rate: int = 100):
        assert data.size(0) == 3, "You need 3-component waveform to use this augmentation"

        permute = [0, 2, 1]
        permuted = data[permute, :]

        assert permuted.size() == data.size(), "Size permuted and original tensor must be the same"
        return permuted


class AddRandomNoise(BaseAugmentation):
    def __init__(self,
                 snr_level_db: float = 1.,
                 p: float = 1.0):
        '''TO DO: ADD MORE DISTRIBTUIONS'''
        super().__init__(p)

        self.seed = torch.random.seed()
        self.snr_level_db = snr_level_db

    def apply_transform(self,
                        data: torch.tensor,
                        sample_rate: int = 100):
        noise = torch.randn_like(data)

        waveform_rms = torch.sqrt(torch.mean(torch.square(data), axis=-1))
        noise_rms = torch.sqrt(torch.mean(torch.square(noise), axis=-1))
        desired_noise_rms = waveform_rms / (10 ** (self.snr_level_db / 20))

        noise *= (desired_noise_rms / noise_rms).unsqueeze(-1)

        return data + noise


class LowPassFilter(BaseAugmentation):
    def __init__(self,
                 cutoff_freq: float = 1.,
                 p: float = 1.0):
        super().__init__(p)

        self.seed = torch.random.seed()
        self.cutoff_freq = cutoff_freq

    def apply_transform(self,
                        data: torch.tensor,
                        sample_rate: int = 100):
        cutoff = self.cutoff_freq / sample_rate
        assert cutoff < 0.5, 'Cutoff frequency is higher than Nyquist frequency'

        return julius.lowpass_filter(input=data, cutoff=cutoff)


class HighPassFilter(LowPassFilter):
    def apply_transform(self,
                        data: torch.tensor,
                        sample_rate: int = 100):
        cutoff = self.cutoff_freq / sample_rate
        assert cutoff < 0.5, 'Cutoff frequency is higher than Nyquist frequency'

        return data - julius.lowpass_filter(input=data, cutoff=cutoff)


class RandomLowPassFilter(BaseAugmentation):
    def __init__(self,
                 cutoff_freq_range: list = [1., 10.],
                 p: float = 1.0):
        super().__init__(p)

        self.seed = torch.random.seed()
        self.cutoff_freq_range = cutoff_freq_range

    def apply_transform(self,
                        data: torch.tensor,
                        sample_rate: int = 100):
        cutoff_freq = torch.FloatTensor(1).uniform_(*self.cutoff_freq_range)
        cutoff = cutoff_freq / sample_rate
        assert cutoff < 0.5, 'Cutoff frequency is higher than Nyquist frequency'

        return julius.lowpass_filter(input=data, cutoff=cutoff)


class RandomHighPassFilter(RandomLowPassFilter):
    def apply_transform(self,
                        data: torch.tensor,
                        sample_rate: int = 100):
        cutoff_freq = torch.FloatTensor(1).uniform_(*self.cutoff_freq_range)
        cutoff = cutoff_freq / sample_rate
        assert cutoff < 0.5, 'Cutoff frequency is higher than Nyquist frequency'

        return data - julius.lowpass_filter(input=data, cutoff=cutoff)


class Taper(BaseAugmentation):
    def __init__(self,
                 max_percentage: float = 0.5,
                 max_length: int = 100,
                 p: float = 1.0):
        super().__init__(p)
        self.max_percentage = max_percentage
        self.max_length = max_length

    def apply_transform(self,
                        data: torch.tensor,
                        sample_rate: int = 100):

        npts = data.size(-1)

        # store all constraints for maximum taper length
        max_half_lenghts = []
        if self.max_percentage is not None:
            max_half_lenghts.append(int(self.max_percentage * npts))
        if self.max_length is not None:
            max_half_lenghts.append(int(self.max_length * sample_rate))

        # add full trace length to constraints
        max_half_lenghts.append(int(npts / 2))
        # select shortest acceptable window half-length
        wlen = min(max_half_lenghts)

        if 2 * wlen == npts:
            taper_sides = torch.hann_window(2 * wlen)
        else:
            taper_sides = torch.hann_window(2 * wlen + 1)
        taper = torch.tensor(torch.hstack((taper_sides[:wlen], torch.ones(npts - 2 * wlen),
                                        taper_sides[len(taper_sides) - wlen:])))

        data *= taper
        return data


class PolarityChange(BaseAugmentation):
    def __init__(self,
                 p: float = 1.0):
        super().__init__(p)

        self.seed = torch.random.seed()

    def apply_transform(self,
                        data: torch.tensor,
                        sample_rate: int = 100):
        return data * -1


class Normalize(BaseAugmentation):
    def __init__(self,
                 p: float = 1.0):
        super().__init__(p)

    def apply_transform(self,
                        data: torch.tensor,
                        sample_rate: int = 100):
        return data / (data.abs().max(dim=-1, keepdim=True).values.max() + 1e-8)
