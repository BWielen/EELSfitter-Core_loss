from perlin_noise import PerlinNoise
import numpy as np


class SpectralImageGenerator:
    def __init__(self, size_x, size_y, size_E, E_start, E_stop):
        self.spectral_image = np.zeros((size_E, size_x, size_y))
        self.energy_axis = np.linspace(E_start, E_stop, size_E)

        self.si_shape = (size_E, size_y, size_x)

    def generate_perlin_noise(self, seed, scale=10):
        noise = PerlinNoise(octaves=5, seed=seed)
        self.noise_map = np.zeros(self.si_shape[1:])

        for y in range(self.si_shape[1]):
            for x in range(self.si_shape[2]):
                nx = x / scale
                ny = y / scale
                noise_val = noise([nx, ny])
                self.noise_map[y][x] = (noise_val + 1) / 2 

        self.noise_map = (self.noise_map - np.min(self.noise_map)) / (np.max(self.noise_map) - np.min(self.noise_map)) 

    def generate_powerlaw_background(self, A_range=(1e4, 1e5), r_range=(1.5, 2.5)):
        """
        Generate power-law background using Perlin noise for A and r.
        """
        # Map noise values -> A and r
        A_map = A_range[0] + self.noise_map * (A_range[1] - A_range[0])
        r_map = r_range[0] + self.noise_map * (r_range[1] - r_range[0])
        
        # Broadcast energy axis to (E, Y, X)
        E = self.energy_axis[:, None, None]

        # Compute power law background E^-r, scaled to be A at E=E_start
        self.background = E**(-r_map[None, :, :])/(self.energy_axis[0]**(-r_map[None, :, :])) * A_map[None, :, :]
        self.spectral_image += self.background

    def add_poisson_noise(self):
        """
        Add Poisson (shot) noise to the spectral image.
        """
        self.spectral_image = np.random.poisson(self.spectral_image).astype(np.float64)

    def add_gaussian_noise_per_pixel(self, snr_per_pixel=50):
        """
        Add Gaussian noise with a specified per-pixel SNR.
        """
        signal_rms = np.sqrt(np.mean(self.spectral_image**2, axis=0))  # shape (Y, X)

        noise_sigma = signal_rms / snr_per_pixel  # shape (Y, X)

        gaussian_noise = np.random.normal(0, 1, size=self.si_shape) * noise_sigma[None, :, :]
        self.spectral_image += gaussian_noise

    def generate_realistic_spectral_image(self, A_range=(1e4, 1e5), r_range=(1.5, 4.0),
                                          poisson_scale=1e4, gaussian_snr=50, seed=0, scale=10):
        """
        Full pipeline: power-law background + Poisson noise + per-pixel Gaussian noise.
        """
        self.generate_perlin_noise(seed=seed, scale=scale)
        self.generate_powerlaw_background(A_range=A_range, r_range=r_range)
        self.add_gaussian_noise_per_pixel(snr_per_pixel=gaussian_snr)
        return self.spectral_image, self.energy_axis



def evaluate_mc_goodness_of_fit_global(predictions, predictions_std, signal_range):
    """
    Evaluate MC-based goodness-of-fit metrics for multiple spectra and energy points,
    returning a single scalar per metric summarizing the whole prediction. Ignores NaNs.

    Parameters
    ----------
    predictions : np.ndarray
        Shape (n_mc, n_spectrum, n_energy), MC replicas of predicted background.
    predictions_std : np.ndarray
        Shape (n_spectrum, n_energy), standard deviation of predictions.
    signal_range : np.ndarray
        Shape (n_spectrum, n_energy), ground truth signal.

    Returns
    -------
    reduced_chi2 : np.ndarray
        shape (1,) reduced chi^2 

    """
    n_mc, n_spectrum, n_energy = predictions.shape
    N_total = n_spectrum * n_energy

    # Safe standard deviation
    sigma_safe = np.where(predictions_std == 0, 1e-10, predictions_std)

    # Residuals per MC replica
    residuals_mc = (predictions - signal_range) / sigma_safe  # shape (n_mc, n_spectrum, n_energy)

    # Mask NaNs
    valid_mask = ~np.isnan(residuals_mc)

    # Empirical chi-squared per MC replica
    chi2_mc = np.nansum(residuals_mc**2, axis=(1,2))
    N_valid_total = np.sum(valid_mask, axis=(1,2))
    reduced_chi2 = np.nanmean(chi2_mc / N_valid_total)

    return reduced_chi2