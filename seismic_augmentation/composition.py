class BaseComposition(object):
    def __init__(self, transforms: list, p: float = 1.0):
        """
        @param transforms: a list of transforms
        @param p: the probability of the transform being applied; default value is 1.0
        """
        for transform in transforms:
            assert isinstance(
                transform, (BaseAugmentation, BaseComposition)
            ), "Expected instances of type `BaseTransform` or `BaseComposition` for variable `transforms`"  # noqa: B950
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"

        self.transforms = transforms
        self.p = p

class Compose(BaseComposition):
    def __call__(
        self,
        data: torch.tensor,
        sample_rate: int,
    ):
        """
        Applies the list of transforms in order to the audio
        @param data: the waveform array to be augmented
        @param sample_rate: the audio sample rate of the inputted audio
        @returns: the augmented waveform array and sample rate
        """
        for transform in self.transforms:
            data = transform(data, sample_rate)
        return data