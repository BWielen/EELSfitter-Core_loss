from perlin_noise import PerlinNoise
import numpy as np
import constants_values as cv
import matplotlib.pyplot as plt
import math

class LowLossEELS:
    def __init__(self, E_0, beta, dispersion):
        self.E_0 = E_0
        self.beta = beta
        self.dispersion = dispersion
        self.energy_values = np.arange(-50, 102)
        self.low_loss_spectrum = np.zeros_like(self.energy_values, dtype=float)

    def lorenzian(self, x, x0, FWHM):
        gamma = FWHM / 2
        return 1/np.pi * gamma / ((x - x0)**2 + gamma**2)

    def P_n(self, scattering_parameter, n):
        return 1 / math.factorial(n) * scattering_parameter**n * np.exp(-scattering_parameter)

    def elastic_cross_section(self, Z):
        return 1.87e-24 * Z**(4/3) * (cv.v(self.E_0) / cv.c())**-2

    def generate_zlp(self):
        zlp_width = np.random.uniform(1, 3)
        zlp = self.lorenzian(self.energy_values, 0, zlp_width)
        sigma_e = self.elastic_cross_section(26)
        scattering_parameter = np.random.uniform(0.1, 1)
        self.low_loss_spectrum += zlp * sigma_e * self.P_n(scattering_parameter, 0)
        self.ZLP = zlp * sigma_e * self.P_n(scattering_parameter, 0)

    def calculate_plasmon_peaks(self, num_peaks=4):
        plasmon_width = np.random.uniform(20, 20)
        plasmon_energy = np.random.uniform(20, 20)
        plasmon_spectrum = (self.energy_values * plasmon_width * plasmon_energy**2) / \
                           ((self.energy_values**2 - plasmon_energy**2)**2 + (plasmon_energy * plasmon_width)**2)
        plasmon_spectrum[self.energy_values < 0] = 0

        Delta_E = self.energy_values[1] - self.energy_values[0]
        shift_idx = plasmon_energy / Delta_E
        scattering_parameter = np.random.uniform(0.1, 1)

        for n in range(1, num_peaks + 1):
            shifted_spectrum = np.roll(plasmon_spectrum, int(n * shift_idx))
            plasmon_spectrum += shifted_spectrum * self.P_n(scattering_parameter, n)
            plasmon_spectrum[self.energy_values < 0] = 0

        plasmon_spectrum /= np.pi * cv.a_0() * cv.m_e() * cv.v(self.E_0)**2 * 1e28 / cv.eVtoJ()
        plasmon_spectrum *= np.log(1 + self.beta**2 / cv.theta_E(plasmon_energy, self.E_0)**2)

        self.low_loss_spectrum += plasmon_spectrum
        self.plasmon_spectrum = plasmon_spectrum

    def get_spectrum(self):
        self.low_loss_spectrum = np.zeros_like(self.energy_values, dtype=float)
        self.generate_zlp()
        self.calculate_plasmon_peaks()
        return self.low_loss_spectrum

    def convolve_low_loss_spectrum(self, core_loss_spectrum):
        """
        Convolve a 1D core-loss spectrum with the low-loss spectrum.
        Generates a fresh low-loss spectrum on each call (stochastic per pixel).
        """
        self.get_spectrum()
        plural_scattering_spectrum = np.convolve(
            core_loss_spectrum, self.low_loss_spectrum, mode='same'
        )
        return plural_scattering_spectrum

class SpectralImageGenerator:
    def __init__(self, size_x, size_y, size_E, E_start, E_stop):
        self.spectral_image = np.zeros((size_E, size_x, size_y))
        self.energy_axis = np.linspace(E_start, E_stop, size_E)
        self.si_shape = (size_E, size_y, size_x)
        self.E_start = E_start
        self.E_stop = E_stop

    def generate_perlin_noise(self, seed, scale=10):
        noise = PerlinNoise(octaves=5, seed=seed)
        self.noise_map = np.zeros(self.si_shape[1:])

        for y in range(self.si_shape[1]):
            for x in range(self.si_shape[2]):
                nx = x / scale
                ny = y / scale
                self.noise_map[y][x] = (noise([nx, ny]) + 1) / 2

        self.noise_map = (self.noise_map - np.min(self.noise_map)) / \
                         (np.max(self.noise_map) - np.min(self.noise_map))
        plt.imshow(self.noise_map, cmap='viridis')
        plt.show()

    def generate_powerlaw_background(self, A_range=(1e4, 1e5), r_range=(1.5, 2.5)):
        A_map = A_range[0] + self.noise_map * (A_range[1] - A_range[0])
        r_map = r_range[0] + self.noise_map * (r_range[1] - r_range[0])
        E = self.energy_axis[:, None, None]
        self.background = E**(-r_map[None, :, :]) / \
                          (self.energy_axis[0]**(-r_map[None, :, :])) * A_map[None, :, :]
        self.spectral_image += self.background

    def add_poisson_noise(self):
        self.spectral_image = np.random.poisson(self.spectral_image).astype(np.float64)

    def add_gaussian_noise_per_pixel(self, snr_per_pixel=50):
        signal_rms = np.sqrt(np.mean(self.spectral_image**2, axis=0))
        noise_sigma = signal_rms / snr_per_pixel
        gaussian_noise = np.random.normal(0, 1, size=self.si_shape) * noise_sigma[None, :, :]
        self.spectral_image += gaussian_noise

    def apply_plural_scattering(self, low_loss_eels: LowLossEELS):
        """
        Convolve each pixel spectrum with the low-loss kernel.

        The spectral_image is already on the padded energy axis when this is
        called from generate_realistic_spectral_image, so no padding logic
        is needed here — it operates on whatever array is currently stored.
        """
        convolved = np.zeros_like(self.spectral_image)
        kernel_len = len(low_loss_eels.energy_values)

        for y in range(self.si_shape[1]):
            for x in range(self.si_shape[2]):
                pixel = self.spectral_image[:, y, x]

                low_loss_eels.get_spectrum()
                kernel = low_loss_eels.low_loss_spectrum.copy()
                self.kernel = kernel

                if kernel.sum() != 0:
                    kernel /= kernel.sum()

                full = np.convolve(pixel, kernel, mode='full')

                zlp_offset = np.argmax(kernel)
                convolved[:, y, x] = full[zlp_offset: zlp_offset + len(pixel)]

        self.spectral_image = convolved

    def generate_realistic_spectral_image(
        self,
        low_loss_eels: LowLossEELS = None,
        A_range=(1e4, 1e5),
        r_range=(1.5, 4.0),
        gaussian_snr=50,
        seed=0,
        scale=10,
        apply_plural_scattering=False,
    ):
        if apply_plural_scattering:
            if low_loss_eels is None:
                raise ValueError(
                    "A LowLossEELS instance must be provided when "
                    "apply_plural_scattering=True."
                )

            kernel_len   = len(low_loss_eels.energy_values)
            dispersion   = (self.E_stop - self.E_start) / (self.si_shape[0] - 1)
            pad_eV       = kernel_len * dispersion
            E_start_ext  = self.E_start - pad_eV

            size_E_ext   = self.si_shape[0] + kernel_len
            energy_ext   = np.linspace(E_start_ext, self.E_stop, size_E_ext)

            # Temporarily swap in the extended axis and a taller spectral cube
            orig_energy  = self.energy_axis
            orig_si      = self.spectral_image
            orig_shape   = self.si_shape

            self.energy_axis     = energy_ext
            self.spectral_image  = np.zeros((size_E_ext, *orig_shape[1:]))
            self.si_shape        = self.spectral_image.shape

            # Generate background on extended axis
            self.generate_perlin_noise(seed=seed, scale=scale)
            self.generate_powerlaw_background(A_range=A_range, r_range=r_range)
            self.original_spectral_image = self.spectral_image.copy()

            # Convolve (operates on the extended cube)
            self.apply_plural_scattering(low_loss_eels)

            # Crop back to the original energy range
            self.spectral_image = self.spectral_image[kernel_len:, :, :]
            self.background     = self.spectral_image.copy()      # <-- add this line
            self.energy_axis    = orig_energy
            self.si_shape       = orig_shape
        else:
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