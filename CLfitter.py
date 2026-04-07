import rsciio.digitalmicrograph as dm

from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from scipy import signal as sp_signal

import math
import time
import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULT CONFIG
#  All tuneable hyperparameters in one place.
#  Pass a partial dict to any method — missing keys fall back to these values.
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # ── NNTrainer: energy preprocessing ─────────────────────────────────
    "log_energy":               True,
    "standardize_targets":      False,

    # ── NNTrainer.train() ────────────────────────────────────────────────
    "epochs":                   200,
    "lr":                       1e-3,
    "batch_size":               100,
    "patience":                 100,
    "min_delta":                1e-4,
    "progress":                 False,
    "lambda_deriv":             10.0,

    # ── NNTrainer.evaluate_model() ───────────────────────────────────────
    "effective_exponent_window_size": 5,

    # ── Pooler.pool_data() ───────────────────────────────────────────────
    "pool_radius":              2,
    "gaussian_kernel":          True,
    "pool_sigma":               None,   # None → library default (radius / 2)

    # ── ClusterAnalyzer.cluster_data() ───────────────────────────────────
    "n_clusters":               4,

    # ── BackgroundTrainer.train_MC_replica_consecutive() ─────────────────
    "n_mc_replicas":            10,
    "replica_version":          "covariance",   # "triangular"|"covariance"|"local"
    "logging":                  False,
    "local_k":                  50,             # K for local replica generation
}


def _cfg(config, key):
    """Return config[key] if present, else DEFAULT_CONFIG[key]."""
    return config.get(key, DEFAULT_CONFIG[key]) if config else DEFAULT_CONFIG[key]


# ═══════════════════════════════════════════════════════════════════════════

class NNTrainer:
    def __init__(self, x_data, y_data, edge_onset, model, config=None):
        """
        Parameters
        ----------
        x_data      : (N, 2) torch tensor  (energy, feature)
        y_data      : (N,)   torch tensor  (intensity, log-domain or raw)
        edge_onset  : float  onset energy in raw units
        model       : torch.nn.Module  background model
        config      : dict, optional
            Recognised keys (all optional, fall back to DEFAULT_CONFIG):
                log_energy, standardize_targets
        """
        config = config or {}
        self.log_energy = _cfg(config, "log_energy")
        self.edge_onset_raw = edge_onset

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.X = x_data.clone()
        self.y_raw = y_data.clone()

        eps = 1e-8
        if self.log_energy:
            self.X[:, 0] = torch.log(self.X[:, 0] + eps)
            self.edge_onset_raw = math.log(edge_onset + eps)

        self.min_x1, self.max_x1 = self.X[:, 0].min().item(), self.X[:, 0].max().item()
        self.min_x2, self.max_x2 = self.X[:, 1].min().item(), self.X[:, 1].max().item()

        self.X[:, 0] = (self.X[:, 0] - self.min_x1) / (self.max_x1 - self.min_x1)
        self.edge_onset_norm = (self.edge_onset_raw - self.min_x1) / (self.max_x1 - self.min_x1)
        self.X[:, 1] = (self.X[:, 1] - self.min_x2) / (self.max_x2 - self.min_x2)

        self.y = self.y_raw.clone()
        self.min_y, self.max_y = self.y.min().item(), self.y.max().item()
        self.y = (self.y - self.min_y) / (self.max_y - self.min_y)

        self.standardize_targets = _cfg(config, "standardize_targets")
        self.evaluation_loss = None
        self.outputs = None

    # ── Utility ─────────────────────────────────────────────────────────

    def _pre_edge_mask(self, X=None):
        if X is None:
            X = self.X
        return X[:, 0] < self.edge_onset_norm

    def _unnormalize_y(self, y):
        return y * (self.max_y - self.min_y) + self.min_y

    def _normalize_inputs(self, x_eval):
        x_eval[:, 0] = (x_eval[:, 0] - self.min_x1) / (self.max_x1 - self.min_x1)
        x_eval[:, 1] = (x_eval[:, 1] - self.min_x2) / (self.max_x2 - self.min_x2)
        return x_eval

    def loss_function(self, x, y_true, lambda_deriv=10.0):
        x = x.clone().detach().requires_grad_(True)
        y_pred = self.model(x).squeeze()
        mse = F.mse_loss(y_pred, y_true)

        grad = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0][:, 0]

        deriv_pen = torch.relu(grad).mean()
        return mse + lambda_deriv * deriv_pen

    # ── Training ─────────────────────────────────────────────────────────

    def train(self, config=None):
        """
        Train the NN model.

        Parameters
        ----------
        config : dict, optional
            Recognised keys:
                epochs, lr, batch_size, patience, min_delta,
                progress, lambda_deriv
        """
        config = config or {}
        epochs       = _cfg(config, "epochs")
        lr           = _cfg(config, "lr")
        batch_size   = _cfg(config, "batch_size")
        patience     = _cfg(config, "patience")
        min_delta    = _cfg(config, "min_delta")
        progress     = _cfg(config, "progress")
        lambda_deriv = _cfg(config, "lambda_deriv")

        optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        dataset    = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss, epochs_no_improve = float("inf"), 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            start_time = time.time()

            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                loss = self.loss_function(inputs, targets, lambda_deriv=lambda_deriv)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)

            if progress and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} "
                      f"- Time: {time.time() - start_time:.2f}s")

            if epoch_loss + min_delta < best_loss:
                best_loss, epochs_no_improve = epoch_loss, 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if progress:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate_model(self, x_eval, y_eval, config=None):
        """
        Parameters
        ----------
        x_eval : (N_spec, N_energy, 2)  raw energies in [:,:,0], feature in [:,:,1]
        y_eval : (N_spec, N_energy)     raw (linear) intensities
        config : dict, optional
            Recognised keys:
                effective_exponent_window_size
        """
        config = config or {}
        effective_exponent_window_size = _cfg(config, "effective_exponent_window_size")

        x_eval = x_eval.clone()

        if self.log_energy:
            energy_for_model = torch.log(x_eval[:, :, 0] + 1e-8)
        else:
            energy_for_model = x_eval[:, :, 0].clone()

        energy_log = torch.log(x_eval[:, :, 0] + 1e-8)
        x_eval_model = torch.stack([energy_for_model, x_eval[:, :, 1]], dim=-1)

        n_spec, n_energy, _ = x_eval_model.shape
        outputs = torch.zeros((n_spec, n_energy), dtype=torch.float32)
        self.evaluation_loss = np.zeros(n_spec)

        edge_onset_log = self.edge_onset_raw if self.log_energy else math.log(self.edge_onset_raw + 1e-8)
        edge_onset_idx = torch.argmin(torch.abs(energy_log[0, :] - edge_onset_log)).item()

        start = max(0, edge_onset_idx - effective_exponent_window_size)
        end   = edge_onset_idx
        if end <= start:
            end   = min(n_energy, edge_onset_idx + 1)
            start = max(0, end - effective_exponent_window_size)
        eff_slice = slice(start, end)

        for i in range(n_spec):
            x_i = torch.stack([energy_for_model[i, :], x_eval[i, :, 1]], dim=-1)
            x_i = self._normalize_inputs(x_i)
            x_i = x_i.to(self.device)
            x_i.requires_grad_(True)

            self.model.eval()
            outputs_log_norm = self.model(x_i).squeeze()
            outputs_log      = self._unnormalize_y(outputs_log_norm)

            grad_norm = torch.autograd.grad(outputs_log_norm.sum(), x_i, retain_graph=True)[0][:, 0]
            dlogI_dx  = (self.max_y - self.min_y) * grad_norm / (self.max_x1 - self.min_x1)

            if self.log_energy:
                dlogI_dlogE = dlogI_dx
            else:
                E_linear    = x_eval[i, :, 0].to(self.device)
                dlogI_dlogE = E_linear * dlogI_dx

            m = (-dlogI_dlogE[eff_slice].mean()).item()

            x0_logE  = energy_log[i, edge_onset_idx].item()
            log_C    = outputs_log[edge_onset_idx].detach().item() + m * x0_logE
            powerlaw_logI = (
                torch.tensor(log_C, device=self.device)
                - m * energy_log[i, edge_onset_idx:].to(self.device)
            )

            outputs[i, :edge_onset_idx] = outputs_log[:edge_onset_idx].detach().cpu().clone()
            outputs[i, edge_onset_idx:] = powerlaw_logI.detach().cpu().clone()

        return outputs

    # ── Diagnostics ──────────────────────────────────────────────────────

    def check_fit_on_training_data(self):
        self.model.eval()
        with torch.no_grad():
            x_sample, y_true = self.X, self.y
            mask = self._pre_edge_mask(x_sample)
            if mask.sum() == 0:
                print("No pre-edge points in sample")
                return
            x_sample, y_true = x_sample[mask].to(self.device), y_true[mask].to(self.device)
            y_pred     = self.model(x_sample).squeeze()
            y_true_un  = self._unnormalize_y(y_true)
            y_pred_un  = self._unnormalize_y(y_pred)

        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        plt.scatter(y_true_un.cpu().numpy(), y_pred_un.cpu().numpy(),
                    alpha=0.6, label="Data points", c="#4B4DED")
        lims = [y_true_un.min().item(), y_true_un.max().item()]
        plt.plot(lims, lims, "--", label="Ideal", color="#FF6F61")
        plt.xlabel("Training Values (a.u.)", fontsize=20)
        plt.ylabel("Prediction Values (a.u.)", fontsize=20)
        plt.xlim(lims); plt.ylim(lims)
        plt.xticks([round(lims[0], 1), round(lims[1], 1)], fontsize=20)
        plt.yticks([round(lims[0], 1), round(lims[1], 1)], fontsize=20)
        plt.savefig("training-prediction-plot.svg")
        plt.show()

        plt.scatter(x_sample[:, 0].cpu().numpy(), y_true_un.cpu().numpy(), label="True")
        plt.scatter(x_sample[:, 0].cpu().numpy(), y_pred_un.cpu().numpy(), label="Predicted")
        plt.xlabel("Energy (normalized)")
        plt.ylabel("Log-intensity (pre-edge)")
        plt.legend()
        plt.title("NN Fit Check (pre-edge) - Energy vs Log-intensity")
        plt.show()

    def check_fit_interpolation(self, E_value=None, n_grid=200):
        self.model.eval()
        with torch.no_grad():
            X, y = self.X, self.y
            if E_value is None:
                E_value = X[:, 0].median().item()

            tol  = 1e-6
            mask = torch.abs(X[:, 0] - E_value) < tol
            X_real, y_real = X[mask], y[mask]

            t_min, t_max = X[:, 1].min().item(), X[:, 1].max().item()
            t_grid = torch.linspace(t_min, t_max, n_grid).to(X.device)
            E_grid = torch.full_like(t_grid, E_value)
            X_grid = torch.stack([E_grid, t_grid], dim=1).to(self.device)
            y_pred = self.model(X_grid).squeeze()
            y_pred_un = self._unnormalize_y(y_pred)
            if X_real.shape[0] > 0:
                y_real_un = self._unnormalize_y(y_real)

        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(t_grid.cpu().numpy(), y_pred_un.cpu().numpy(), label="NN prediction", lw=2)
        if X_real.shape[0] > 0:
            plt.scatter(X_real[:, 1].cpu().numpy(), y_real_un.cpu().numpy(),
                        color="r", alpha=0.6, label="True data")
        plt.xlabel("TII (X[:,1])")
        plt.ylabel("Log-intensity")
        plt.title(f"Interpolation at fixed Energy E={E_value:.3f}")
        plt.legend()
        plt.show()

    def train_with_epoch_predictions(self, config=None):
        """
        Train while recording per-epoch predictions on the training set.

        Parameters
        ----------
        config : dict, optional
            Same keys as train().

        Returns
        -------
        torch.Tensor  shape (n_epochs, N_train)
        """
        config = config or {}
        epochs       = _cfg(config, "epochs")
        lr           = _cfg(config, "lr")
        batch_size   = _cfg(config, "batch_size")
        lambda_deriv = _cfg(config, "lambda_deriv")
        progress     = _cfg(config, "progress")

        optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        dataset    = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        all_preds = []
        for epoch in range(epochs):
            self.model.train()
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                loss = self.loss_function(inputs, targets, lambda_deriv=lambda_deriv)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                preds_norm = self.model(self.X.to(self.device)).squeeze()
                preds      = self._unnormalize_y(preds_norm).cpu()
            all_preds.append(preds.clone())

            if progress and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"[Epoch {epoch+1}/{epochs}] loss={loss.item():.4f}")

        return torch.stack(all_preds, dim=0)

    def train_with_epoch_predictions_full_spectrum(self, x_eval, config=None):
        """
        Train while recording full-spectrum (including post-edge extrapolation)
        predictions at every epoch.

        Parameters
        ----------
        x_eval : torch.Tensor  shape [1, n_E, 2]
        config : dict, optional
            Same keys as train(); also effective_exponent_window_size.

        Returns
        -------
        torch.Tensor  shape (n_epochs, n_E)
        """
        config = config or {}
        epochs       = _cfg(config, "epochs")
        lr           = _cfg(config, "lr")
        batch_size   = _cfg(config, "batch_size")
        lambda_deriv = _cfg(config, "lambda_deriv")
        progress     = _cfg(config, "progress")

        optimizer  = optim.Adam(self.model.parameters(), lr=lr)
        dataset    = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        all_outputs = []
        for epoch in range(epochs):
            self.model.train()
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                loss = self.loss_function(inputs, targets, lambda_deriv=lambda_deriv)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                x_eval_clone = x_eval.clone().detach().requires_grad_(True)
                outputs       = self.evaluate_model(x_eval_clone, None, config=config)
                outputs_linear = torch.exp(outputs).squeeze()
            all_outputs.append(outputs_linear.cpu().clone())

            if progress and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"[Epoch {epoch+1}/{epochs}] loss={loss.item():.4f}")

        return torch.stack(all_outputs, dim=0)


# ═══════════════════════════════════════════════════════════════════════════

class DataHandler:
    def __init__(self):
        self.signal             = None
        self.spatial_axis_x     = None
        self.spatial_axis_y     = None
        self.energy_axis        = None
        self.si_size            = None

    def other_data(self, signal, spatial_axis_x, spatial_axis_y, energy_axis,
                   low_loss=None, low_loss_energy_axis=None):
        self.signal         = signal
        self.spatial_axis_x = spatial_axis_x
        self.spatial_axis_y = spatial_axis_y
        self.energy_axis    = energy_axis
        self.si_size        = (energy_axis.size, spatial_axis_x.size, spatial_axis_y.size)
        if low_loss is not None:
            self.low_loss              = low_loss
            self.low_loss_energy_axis  = low_loss_energy_axis

    def read_dm3_linescan(self, path):
        file = dm.file_reader(path)
        self.signal = file[0]["data"].T

        spatial_y_metadata = file[0]["axes"][0]
        self.spatial_axis_y = np.linspace(
            spatial_y_metadata["offset"],
            spatial_y_metadata["offset"] + spatial_y_metadata["scale"] * (spatial_y_metadata["size"] - 1),
            spatial_y_metadata["size"],
        )
        self.spatial_axis_x = np.ones(1)

        energy_axis_metadata = file[0]["axes"][1]
        self.energy_axis = np.linspace(
            energy_axis_metadata["offset"],
            energy_axis_metadata["offset"] + energy_axis_metadata["scale"] * (energy_axis_metadata["size"] - 1),
            energy_axis_metadata["size"],
        )

    def read_dm4_SI(self, path, core_loss_index=3, lowloss=False):
        file = dm.file_reader(path)
        data = file[core_loss_index]

        energy_axis_metadata = data["axes"][0]
        self.energy_axis = np.linspace(
            energy_axis_metadata["offset"],
            energy_axis_metadata["offset"] + energy_axis_metadata["scale"] * (energy_axis_metadata["size"] - 1),
            energy_axis_metadata["size"],
        )
        spatial_y_metadata = data["axes"][1]
        self.spatial_axis_y = np.linspace(
            spatial_y_metadata["offset"],
            spatial_y_metadata["offset"] + spatial_y_metadata["scale"] * (spatial_y_metadata["size"] - 1),
            spatial_y_metadata["size"],
        )
        spatial_x_metadata = data["axes"][2]
        self.spatial_axis_x = np.linspace(
            spatial_x_metadata["offset"],
            spatial_x_metadata["offset"] + spatial_x_metadata["scale"] * (spatial_x_metadata["size"] - 1),
            spatial_x_metadata["size"],
        )
        self.si_size = (
            energy_axis_metadata["size"],
            spatial_y_metadata["size"],
            spatial_x_metadata["size"],
        )
        self.signal = data["data"][:, :].reshape(
            -1, spatial_x_metadata["size"] * spatial_y_metadata["size"]
        )

        if lowloss:
            self.low_loss = file[2]["data"].reshape(
                -1, spatial_x_metadata["size"] * spatial_y_metadata["size"]
            )
            low_loss_metadata = file[2]["axes"][0]
            self.low_loss_energy_axis = np.linspace(
                low_loss_metadata["offset"],
                low_loss_metadata["offset"] + low_loss_metadata["scale"] * (low_loss_metadata["size"] - 1),
                low_loss_metadata["size"],
            )

    def align_data_cross_correlate(self, reference_spectrum_index=0):
        shifts = []
        reference_spectrum = self.signal[:, reference_spectrum_index]
        for i, x in enumerate(self.signal.T):
            cross_correlate = sp_signal.correlate(x, reference_spectrum, mode="full")
            shift = len(x) - np.argmax(cross_correlate) - 1
            self.signal[:, i] = np.roll(x, shift)
            shifts.append(int(shift))
        self.window_data(
            lower=self.energy_axis[max(shifts)],
            higher=self.energy_axis[min(shifts)] if min(shifts) < 0 else 1e6,
        )

    def align_data_ZLP(self):
        shifts = np.argmin(abs(self.low_loss_energy_axis)) - np.argmax(self.low_loss, axis=0)
        for i, x in enumerate(self.signal.T):
            shift = shifts[i]
            self.signal[:, i]    = np.roll(x, shift)
            self.low_loss[:, i]  = np.roll(self.low_loss[:, i], shift)
        self.window_data(
            lower=self.energy_axis[np.max(shifts)],
            higher=self.energy_axis[np.min(shifts)] if np.min(shifts) < 0 else 1e6,
        )

    def window_data(self, lower, higher):
        window = (self.energy_axis < higher) & (self.energy_axis > lower)
        self.signal      = self.signal[window]
        self.energy_axis = self.energy_axis[window]
        self.si_size     = (self.signal.shape[0], self.spatial_axis_y.size, self.spatial_axis_x.size)

    def plot_spectra(self, spectra_indices=(1, 2, 3), energy_range=None, logscale=False, legend=False):
        mask = (self.energy_axis > energy_range[0]) & (self.energy_axis < energy_range[1])
        plt.figure(figsize=(10, 6))
        for i in spectra_indices:
            plt.plot(self.energy_axis[mask], self.signal[mask, i], label=f"Spectrum {i}")
        plt.xlabel("Energy Loss (eV)")
        plt.ylabel("Intensity (a.u.)")
        plt.title("Core Loss Spectra")
        if logscale:
            plt.yscale("log")
        if legend:
            plt.legend()
        plt.show()

    def plot_intensity_histogram(self, bins_nr=None):
        total_integrated_intensity = np.sum(self.signal, axis=0).flatten()
        if bins_nr is None:
            bins_nr = int(len(total_integrated_intensity) * 0.1)
        plt.hist(total_integrated_intensity, bins=bins_nr)


# ═══════════════════════════════════════════════════════════════════════════

class Pooler:
    def __init__(self, signal, si_shape):
        self.signal = signal.copy().reshape(si_shape)

    def pool_data(self, config=None):
        """
        Parameters
        ----------
        config : dict, optional
            Recognised keys: pool_radius, gaussian_kernel, pool_sigma
        """
        config        = config or {}
        sqr_radius    = _cfg(config, "pool_radius")
        gaussian_kernel = _cfg(config, "gaussian_kernel")
        sigma         = _cfg(config, "pool_sigma")

        if gaussian_kernel:
            if sigma is None:
                sigma = sqr_radius / 2
            size = 2 * sqr_radius - 1
            x, y = np.meshgrid(
                np.linspace(-sqr_radius + 1, sqr_radius - 1, size),
                np.linspace(-sqr_radius + 1, sqr_radius - 1, size),
            )
            kernel  = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel /= np.sum(kernel)
        else:
            kernel = np.ones((2 * sqr_radius - 1, 2 * sqr_radius - 1)) / ((2 * sqr_radius - 1) ** 2)

        for i in range(self.signal.shape[0]):
            self.signal[i, :, :] = sp_signal.convolve2d(
                self.signal[i, :, :], kernel, mode="same", boundary="symm"
            )

        return self.signal.reshape(-1, self.signal.shape[1] * self.signal.shape[2])


# ═══════════════════════════════════════════════════════════════════════════

class ClusterAnalyzer:
    def __init__(self, signal):
        self.signal           = signal.copy()
        self.cluster_centers  = None
        self.clusters_mean    = None
        self.clusters_covariance = None

    def cluster_data(self, pre_edge_mask=None, config=None):
        """
        Parameters
        ----------
        pre_edge_mask : np.ndarray of bool, optional
        config : dict, optional
            Recognised keys: n_clusters
        """
        config     = config or {}
        n_clusters = _cfg(config, "n_clusters")

        n_E = self.signal.shape[0]
        self.total_integrated_intensity = np.sum(self.signal, axis=0)

        kmeans = KMeans(n_clusters=n_clusters)
        self.clusters = kmeans.fit_predict(
            np.log(self.total_integrated_intensity).reshape(-1, 1)
        )
        self.cluster_centers = kmeans.cluster_centers_

        if pre_edge_mask is None:
            use_signal = self.signal
        else:
            use_signal = self.signal[pre_edge_mask, :]
            n_E = use_signal.shape[0]

        self.clusters_mean       = np.zeros((n_E, n_clusters))
        self.clusters_covariance = np.zeros((n_E, n_E, n_clusters))

        for i in range(n_clusters):
            cluster_data_log = np.log(
                np.clip(use_signal[:, self.clusters == i], a_min=1e-10, a_max=None)
            )
            self.clusters_mean[:, i] = np.mean(cluster_data_log, axis=1)
            if cluster_data_log.shape[1] > 1:
                lw = LedoitWolf().fit(cluster_data_log.T)
                self.clusters_covariance[:, :, i] = lw.covariance_
            else:
                self.clusters_covariance[:, :, i] = np.eye(cluster_data_log.shape[0])

    def cholesky_decomp(self):
        self.triangular_matices = np.zeros_like(self.clusters_covariance)
        for i in range(self.clusters_covariance.shape[2]):
            try:
                self.triangular_matices[:, :, i] = np.linalg.cholesky(self.clusters_covariance[:, :, i])
            except np.linalg.LinAlgError:
                print(f"Cholesky decomposition failed for cluster {i}. Using identity matrix instead.")
                self.triangular_matices[:, :, i] = np.eye(self.clusters_covariance.shape[0])


# ═══════════════════════════════════════════════════════════════════════════

class X_Builder:
    def __init__(self, energy_axis):
        self.energy_axis = energy_axis

    def prepare_X_mc_data(self, cluster_centers, edge_onset, energy_range=None):
        mask = self.energy_axis < edge_onset
        if energy_range is not None:
            mask &= (self.energy_axis >= energy_range[0]) & (self.energy_axis <= energy_range[1])

        energy_axis_masked = self.energy_axis[mask]
        num_energy_loss, num_clusters = len(energy_axis_masked), len(cluster_centers)

        energy_axis_expanded = (
            torch.tensor(energy_axis_masked, dtype=torch.float32)
            .unsqueeze(0).expand(num_clusters, -1)
        )
        clustered_spectra_centers = torch.tensor(cluster_centers, dtype=torch.float32).squeeze()
        clustered_spectra_centers_expanded = clustered_spectra_centers.unsqueeze(1).expand(-1, num_energy_loss)

        X_mc = torch.stack([energy_axis_expanded, clustered_spectra_centers_expanded], dim=2)
        self.X_mc        = X_mc.reshape(-1, 2)
        self.pre_edge_mask = mask

    def prepare_X_eval_data(self, total_integrated_intensity):
        num_energy_loss = len(self.energy_axis)
        num_spectra     = len(total_integrated_intensity)

        energy_axis_expanded = (
            torch.tensor(self.energy_axis, dtype=torch.float32)
            .unsqueeze(0).expand(num_spectra, -1)
        )
        tii_expanded = torch.log(
            torch.tensor(total_integrated_intensity, dtype=torch.float32)
            .unsqueeze(1).expand(-1, num_energy_loss)
        )
        self.X_eval = torch.stack([energy_axis_expanded, tii_expanded], dim=2)


# ═══════════════════════════════════════════════════════════════════════════

class BackgroundTrainer:
    def __init__(self, signal, pre_edge_mask, X_mc, X_eval,
                 clustered_spectra_mean, triangular_matices,
                 covariance_matrices, cluster_labels):
        self.signal                  = signal.copy()
        self.pre_edge_mask           = pre_edge_mask
        self.X_mc                    = X_mc
        self.X_eval                  = X_eval
        self.clustered_spectra_mean  = clustered_spectra_mean
        self.triangular_matices      = triangular_matices
        self.covariance_matrices     = covariance_matrices
        self.cluster_labels          = cluster_labels

    # ── Replica generators ───────────────────────────────────────────────

    def _generate_mc_replica_local(self, cluster_id, K=50, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.cluster_labels is None:
            raise ValueError("cluster_labels must be provided to use replica_version='local'")

        signal = self.signal if self.pre_edge_mask is None else self.signal[self.pre_edge_mask, :]
        mask   = self.cluster_labels == cluster_id
        X_log  = np.log(np.clip(signal[:, mask], 1e-10, None)).T

        if X_log.shape[0] == 0:
            return self.clustered_spectra_mean[:, cluster_id].copy()

        anchor_idx = rng.integers(0, X_log.shape[0])
        x_anchor   = X_log[anchor_idx]
        K_eff      = min(K, X_log.shape[0])
        nn         = NearestNeighbors(n_neighbors=K_eff).fit(X_log)
        idxs       = nn.kneighbors([x_anchor], return_distance=False)
        local      = X_log[idxs[0]]
        lw         = LedoitWolf().fit(local)
        z          = rng.multivariate_normal(np.zeros(x_anchor.size), lw.covariance_)
        return x_anchor + z

    def _generate_mc_replica_local_all_clusters(self, K=50, rng=None):
        n_E, n_clusters = self.clustered_spectra_mean.shape
        out = np.zeros((n_E, n_clusters))
        for c in range(n_clusters):
            out[:, c] = self._generate_mc_replica_local(c, K=K, rng=rng)
        return out

    def _generate_mc_replica_triangular(self):
        mc_replica_log = np.zeros_like(self.clustered_spectra_mean)
        n_E = self.clustered_spectra_mean.shape[0]
        for cluster_id in range(mc_replica_log.shape[1]):
            z  = np.random.randn(n_E)
            L  = self.triangular_matices[:, :, cluster_id]
            mc_replica_log[:, cluster_id] = self.clustered_spectra_mean[:, cluster_id] + np.dot(L, z)
        return mc_replica_log

    def _generate_mc_replica_covariance(self):
        mc_replica_log = np.zeros_like(self.clustered_spectra_mean)
        n_E = self.clustered_spectra_mean.shape[0]
        for cluster_id in range(mc_replica_log.shape[1]):
            z = np.random.multivariate_normal(
                mean=np.zeros(n_E), cov=self.covariance_matrices[:, :, cluster_id]
            )
            mc_replica_log[:, cluster_id] = self.clustered_spectra_mean[:, cluster_id] + z
        return mc_replica_log

    def _make_replica(self, config):
        """Dispatch to the correct replica generator using config."""
        version = _cfg(config, "replica_version")
        K       = _cfg(config, "local_k")
        if version == "triangular":
            return self._generate_mc_replica_triangular()
        elif version == "covariance":
            return self._generate_mc_replica_covariance()
        elif version == "local":
            return self._generate_mc_replica_local_all_clusters(K=K)
        else:
            raise ValueError("replica_version must be 'triangular', 'covariance', or 'local'")

    # ── Main training loop ───────────────────────────────────────────────

    def train_MC_replica_consecutive(self, edge_onset, model, config=None):
        """
        Train neural network background models for multiple MC replicas.

        Parameters
        ----------
        edge_onset : float
            Energy value for the edge onset.
        model : torch.nn.Module
            A fresh (untrained) model instance.
        config : dict, optional
            Recognised keys:
                n_mc_replicas, replica_version, local_k, logging,
                + all keys forwarded to NNTrainer.__init__ and NNTrainer.train()
        """
        config       = config or {}
        n_mc_replicas = _cfg(config, "n_mc_replicas")
        print(n_mc_replicas, "MC replicas will be trained.")
        logging       = _cfg(config, "logging")

        self.background = np.zeros((
            n_mc_replicas,
            self.signal.shape[1],
            self.signal.shape[0],
        ))

        for i in range(n_mc_replicas):
            print(f"Starting Replica {i+1}/{n_mc_replicas}")
            mc_replica = self._make_replica(config)
            y = torch.tensor(mc_replica.T, dtype=torch.float32).reshape(-1)

            if logging:
                print("X_mc shape:", self.X_mc.shape)
                print("y shape:",    y.shape)
                plt.scatter(self.X_mc[:, 0], y, s=3, alpha=0.5)
                plt.title("X_mc[:,0] vs y before training")
                plt.show()

            nn_obj = NNTrainer(self.X_mc, y, edge_onset, model, config=config)
            nn_obj.train(config=config)

            if logging:
                nn_obj.check_fit_on_training_data()
                nn_obj.check_fit_interpolation()

            self.background[i] = np.exp(
                nn_obj.evaluate_model(self.X_eval, self.signal.T, config=config)
                .detach().numpy()
            )

    # ── Diagnostics ──────────────────────────────────────────────────────

    def check_mc_replicas_vs_clusters(self, n_samples=20, cluster_id=0):
        cluster_mean = self.clustered_spectra_mean[:, cluster_id]
        replicas = np.stack(
            [self._generate_mc_replica_covariance()[:, cluster_id] for _ in range(n_samples)],
            axis=0,
        )
        plt.plot(cluster_mean, label="Cluster mean", color="black", linewidth=2)
        plt.fill_between(
            np.arange(len(cluster_mean)),
            np.percentile(replicas, 5,  axis=0),
            np.percentile(replicas, 95, axis=0),
            alpha=0.5, label="MC 5–95% interval",
        )
        plt.title("MC Replicas vs. Cluster Mean")
        plt.xlabel("Energy index")
        plt.ylabel("Log Intensity")
        plt.legend()
        plt.show()

    def run_single_replica_with_predictions(self, x_eval, edge_onset, model, config=None):
        """
        Run one MC replica and record full-spectrum predictions at every epoch.

        Parameters
        ----------
        x_eval     : torch.Tensor  shape [1, n_E, 2]
        edge_onset : float
        model      : torch.nn.Module
        config     : dict, optional
            Same keys as train_MC_replica_consecutive().

        Returns
        -------
        epoch_preds : np.ndarray  shape (n_epochs, n_E)
        y           : torch.Tensor  (training targets)
        """
        config     = config or {}
        mc_replica = self._make_replica(config)
        y          = torch.tensor(mc_replica.T, dtype=torch.float32).reshape(-1)

        nn_obj      = NNTrainer(self.X_mc, y, edge_onset, model, config=config)
        epoch_preds = nn_obj.train_with_epoch_predictions_full_spectrum(x_eval, config=config)
        return epoch_preds.cpu().numpy(), y


# ═══════════════════════════════════════════════════════════════════════════

class PredictionSaver:
    def __init__(self, signal, energy_axis, spatial_axis_x, spatial_axis_y, predictions):
        self.signal         = signal
        self.energy_axis    = energy_axis
        self.spatial_axis_x = spatial_axis_x
        self.spatial_axis_y = spatial_axis_y
        self.predictions    = predictions

    def save_predictions(self, path):
        np.savez(
            path,
            signal=self.signal,
            energy_axis=self.energy_axis,
            spatial_axis_x=self.spatial_axis_x,
            spatial_axis_y=self.spatial_axis_y,
            predictions=self.predictions,
        )