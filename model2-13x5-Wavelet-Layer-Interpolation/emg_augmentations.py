#import numba
import numpy as np
import pywt
from scipy import interpolate


class EMGAugmentation:
    """Base class for EMG augmentations.
    
    Methods
    -------
    __call__(chunk: numpy.ndarray) -> numpy.ndarray
        Augments the chunk.
    """
    
    def __init__(self):
        pass

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        return self._augment(chunk)

    def __str__(self):
        return "".join([" " + i if i.isupper() else i for i in self.__class__.__name__]).strip()

    def __repr__(self):
        return self.__str__()

    def _augment(self, chunk: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


#@numba.njit(nogil=True, fastmath=True)
def _gaussian_noise(chunk: np.ndarray, target_snr__db: float = 5.0) -> np.ndarray:
    squared_chunk = np.square(chunk.astype(np.float64))
    mean_squared_chunk = np.zeros((chunk.shape[0], chunk.shape[1]))
    for i in range(chunk.shape[2]):
        mean_squared_chunk += squared_chunk[:, :, i]
    mean_squared_chunk /= chunk.shape[2]

    noise_avg_per_channel_and_electrode__watts = np.sqrt(
        np.power(10, ((10 * np.log10(mean_squared_chunk + 1e-15)) - target_snr__db) / 10)
    )

    for channel_index in range(chunk.shape[0]):
        for electrode_index in range(chunk.shape[1]):
            chunk[channel_index, electrode_index] += np.random.normal(
                loc=0.0,
                scale=noise_avg_per_channel_and_electrode__watts[channel_index, electrode_index],
                size=chunk.shape[2],
            ).astype(np.int16)

    return chunk


class GaussianNoise(EMGAugmentation):
    """Adds Gaussian gaussian_noise to the input EMG data. This augmentation is based on the paper [1].
    
    Parameters
    ----------
    target_snr__db : float, optional
        The target signal to gaussian_noise ratio in decibels, by default 5.0.
        
    Methods
    -------
    __call__(chunk: numpy.ndarray) -> numpy.ndarray
        Augments the chunk.
    
    References
    ----------
    [1] Tsinganos, P., Cornelis, B., Cornelis, J., Jansen, B., Skodras, A., 2020.
    Data Augmentation of Surface Electromyography for Hand Gesture Recognition. Sensors 20, 4892.
    https://doi.org/10/grc7ph
    """
    def __init__(self, target_snr__db: float = 5.0):
        super().__init__()
        self.target_snr__db = target_snr__db

    def _augment(self, chunk: np.ndarray) -> np.ndarray:
        return _gaussian_noise(chunk=chunk, target_snr__db=self.target_snr__db)


class MagnitudeWarping(EMGAugmentation):
    """Magnitude warping augmentation. This augmentation is based on the paper [1].
    
    Parameters
    ----------
    nr_of_point_for_spline : int, optional
        The number of points to use for the spline, by default 6.
    gaussian_mean : float, optional
        The mean of the Gaussian distribution, by default 1.0.
    gaussian_std : float, optional
        The standard deviation of the Gaussian distribution, by default 0.35.
    nr_of_grids : int, optional
        The number of grids to use, by default 5.
        
    Methods
    -------
    __call__(chunk: numpy.ndarray) -> numpy.ndarray
        Augments the chunk.
        
    References
    ----------
    [1] Tsinganos, P., Cornelis, B., Cornelis, J., Jansen, B., Skodras, A., 2020.
    Data Augmentation of Surface Electromyography for Hand Gesture Recognition. Sensors 20, 4892.
    https://doi.org/10/grc7ph
    """
    def __init__(
        self,
        nr_of_point_for_spline: int = 6,
        gaussian_mean: float = 1.0,
        gaussian_std: float = 0.35,
        nr_of_grids: int = 5,
    ):
        super().__init__()
        self.nr_of_point_for_spline = nr_of_point_for_spline
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        self.nr_of_grids = nr_of_grids

    def _augment(self, chunk: np.ndarray) -> np.ndarray:
        random_gens = [np.random.default_rng() for _ in range(self.nr_of_grids)]

        return np.multiply(
            chunk.astype(np.float64),
            np.repeat(
                np.repeat(
                    np.array(
                        [
                            interpolate.interp1d(
                                np.linspace(
                                    start=0,
                                    stop=chunk.shape[-1],
                                    num=self.nr_of_point_for_spline,
                                ),
                                random_gens[i].normal(
                                    loc=self.gaussian_mean,
                                    scale=self.gaussian_std,
                                    size=self.nr_of_point_for_spline,
                                ),
                                kind="cubic",
                            )(np.arange(chunk.shape[-1]))
                            for i in range(self.nr_of_grids)
                        ]
                    ),
                    repeats=chunk.shape[1] // self.nr_of_grids,
                    axis=0,
                )[None, ...],
                repeats=chunk.shape[0],
                axis=0,
            ),
        )


class WaveletDecomposition(EMGAugmentation):
    """Wavelet decomposition augmentation. This augmentation is based on the paper [1].
    
    Parameters
    ----------
    b : float, optional
        The scaling factor, by default 0.25.
    wavelet : str, optional
        The wavelet to use, by default "db7".
    level : int, optional
        The level of decomposition, by default 5.
    nr_of_grids : int, optional
        The number of grids to use, by default 5.
        
    Methods
    -------
    __call__(chunk: numpy.ndarray) -> numpy.ndarray
        Augments the chunk.
        
    References
    ----------
    [1] Tsinganos, P., Cornelis, B., Cornelis, J., Jansen, B., Skodras, A., 2020.
    Data Augmentation of Surface Electromyography for Hand Gesture Recognition. Sensors 20, 4892.
    https://doi.org/10/grc7ph
    """
    def __init__(
        self,
        b: float = 0.25,
        wavelet: str = "db7",
        level: int = 5,
        nr_of_grids: int = 5,
    ):
        super().__init__()
        self.b = b
        self.wavelet = wavelet
        self.level = level
        self.nr_of_grids = nr_of_grids

    def _augment(self, chunk: np.ndarray) -> np.ndarray:
        coefficients_per_grid = [
            pywt.wavedec(
                grid.astype(np.float64),
                wavelet=self.wavelet,
                level=self.level,
                mode="reflect",
                axis=2,
            )
            for grid in np.array_split(chunk, self.nr_of_grids, axis=1)
        ]
        return np.concatenate(
            [
                pywt.waverec(
                    [coefficients[0]] + [coefficients[i] * self.b for i in range(1, self.level + 1)],
                    wavelet=self.wavelet,
                    axis=2,
                )
                for coefficients in coefficients_per_grid
            ],
            axis=1,
        )
