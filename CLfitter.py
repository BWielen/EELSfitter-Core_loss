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


class NNTrainer:
    def __init__(self, x_data, y_data, edge_onset, model, log_energy=True, standardize_targets=False):
        """
        Args:
            x_data: (N, 2) torch tensor (energy, feature)
            y_data: (N,) torch tensor (intensity, already in log-domain or raw)
            edge_onset: float, onset energy in raw units
            model: A torch model used for the background training
            log_energy: if True, log-transform energy axis
            standardize_targets: if True, standardize y before training
        """
        self.log_energy = log_energy
        self.edge_onset_raw = edge_onset
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # Clone input
        self.X = x_data.clone()
        self.y_raw = y_data.clone()

        # Energy preprocessing
        eps = 1e-8
        if self.log_energy:
            self.X[:, 0] = torch.log(self.X[:, 0] + eps)
            self.edge_onset_raw = math.log(edge_onset + eps)

        # Store min/max for normalization
        self.min_x1, self.max_x1 = self.X[:, 0].min().item(), self.X[:, 0].max().item()
        self.min_x2, self.max_x2 = self.X[:, 1].min().item(), self.X[:, 1].max().item()

        # Normalize inputs
        self.X[:, 0] = (self.X[:, 0] - self.min_x1) / (self.max_x1 - self.min_x1)
        self.edge_onset_norm = (self.edge_onset_raw - self.min_x1) / (self.max_x1 - self.min_x1)
        self.X[:, 1] = (self.X[:, 1] - self.min_x2) / (self.max_x2 - self.min_x2)

        # ---- Target preprocessing ----
        self.y = self.y_raw.clone()

        self.min_y, self.max_y = self.y.min().item(), self.y.max().item()

        self.y = (self.y - self.min_y) / (self.max_y - self.min_y)  # Normalize targets to [0, 1]

        self.standardize_targets = standardize_targets
        self.evaluation_loss = None
        self.outputs = None

    # Utility Functions
    def _pre_edge_mask(self, X=None):
        """Return boolean mask for pre-edge region."""
        if X is None:
            X = self.X
        return X[:, 0] < self.edge_onset_norm

    def _unnormalize_y(self, y):
        """Undo optional target standardization."""
        return y * (self.max_y - self.min_y) + self.min_y

    def _normalize_inputs(self, x_eval):
        """Normalize evaluation inputs (energy + feature)."""
        x_eval[:, 0] = (x_eval[:, 0] - self.min_x1) / (self.max_x1 - self.min_x1)
        x_eval[:, 1] = (x_eval[:, 1] - self.min_x2) / (self.max_x2 - self.min_x2)
        return x_eval

    def loss_function(self, x, y_true, lambda_deriv=10.0):
        # x must require grad for autograd derivative
        x = x.clone().detach().requires_grad_(True)

        y_pred = self.model(x).squeeze()
        mse = F.mse_loss(y_pred, y_true)

        # partial derivative wrt x1 (column 0)
        grad = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, 0]

        deriv_pen = torch.relu(grad).mean()  # penalize ∂y/∂x1 >= 0
        return mse + lambda_deriv * deriv_pen

    # ---------------- Training ----------------
    def train(self, epochs=200, lr=1e-3, batch_size=100,
              patience=100, min_delta=1e-4, progress=False,
              lambda_deriv=10.0):
        """Train the NN model.
        Args:
            epochs: max number of epochs
            lr: learning rate
            batch_size: batch size
            patience: epochs with no improvement to wait before early stopping
            min_delta: minimum change in loss to qualify as improvement
            progress: if True, print progress every 10 epochs
            lambda_deriv: weight for derivative penalty in loss function
        """

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        dataset = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss, epochs_no_improve = float("inf"), 0

        # Training loop
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
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Time: {time.time() - start_time:.2f}s")

            # Early stopping
            if epoch_loss + min_delta < best_loss:
                best_loss, epochs_no_improve = epoch_loss, 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if progress:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluation
    def evaluate_model(self, x_eval, y_eval, effective_exponent_window_size=5):
        """
        Args:
            x_eval: (N_spec, N_energy, 2) with raw energies in [:,:,0] and feature in [:,:,1]
            y_eval: (N_spec, N_energy) raw (linear) intensities
        Returns:
            outputs: (N_spec, N_energy) predicted log-intensities in the RAW domain (not normalized)
        """
        x_eval = x_eval.clone()

        # energy_for_model is what the NN expects in its first channel (log(E) if self.log_energy)
        if self.log_energy:
            energy_for_model = torch.log(x_eval[:, :, 0] + 1e-8)
        else:
            energy_for_model = x_eval[:, :, 0].clone()

        # We'll also keep logE for the power-law continuation regardless of training mode
        energy_log = torch.log(x_eval[:, :, 0] + 1e-8)

        # Compose an input tensor with the correct energy channel + the feature channel
        x_eval_model = torch.stack([energy_for_model, x_eval[:, :, 1]], dim=-1)

        n_spec, n_energy, _ = x_eval_model.shape
        outputs = torch.zeros((n_spec, n_energy), dtype=torch.float32)

        self.evaluation_loss = np.zeros(n_spec)

        # Edge onset in log domain for indexing
        edge_onset_log = self.edge_onset_raw if self.log_energy else math.log(self.edge_onset_raw + 1e-8)
        edge_onset_idx = torch.argmin(torch.abs(energy_log[0, :] - edge_onset_log)).item()

        # Window [start:end) just before the edge
        start = max(0, edge_onset_idx - effective_exponent_window_size)
        end = edge_onset_idx
        if end <= start:  # ensure at least one point
            end = min(n_energy, edge_onset_idx + 1)
            start = max(0, end - effective_exponent_window_size)
        eff_slice = slice(start, end)

        for i in range(n_spec):
            # Per-spectrum inputs (energy_for_model, feature), then normalize like training
            x_i = torch.stack([energy_for_model[i, :], x_eval[i, :, 1]], dim=-1)
            x_i = self._normalize_inputs(x_i)
            x_i = x_i.to(self.device)
            x_i.requires_grad_(True)

            self.model.eval()
            # NN output in normalized target space
            outputs_log_norm = self.model(x_i).squeeze()
            # Un-normalize to raw log-intensity domain
            outputs_log = self._unnormalize_y(outputs_log_norm)

            # Effective exponent r_eff = -d(logI)/d(logE)
            # grad_norm: d(outputs_norm)/d(x_norm)
            grad_norm = torch.autograd.grad(outputs_log_norm.sum(), x_i, retain_graph=True)[0][:, 0]

            # Chain rule to get d(logI)/d(logE):
            # d(logI)/d(x) = (max_y-min_y) * d(outputs_norm)/d(x_norm) * d(x_norm)/d(x)
            # where x = log(E) if self.log_energy, else x = E.
            dlogI_dx = (self.max_y - self.min_y) * grad_norm / (self.max_x1 - self.min_x1)

            if self.log_energy:
                # x is log(E) already → d(logI)/d(logE) = dlogI_dx
                dlogI_dlogE = dlogI_dx
            else:
                # x is E -> d(logI)/d(logE) = E * d(logI)/dE
                E_linear = x_eval[i, :, 0].to(self.device)
                dlogI_dlogE = E_linear * dlogI_dx

            m = (-dlogI_dlogE[eff_slice].mean()).item()

            # Power-law extrapolation: logI = logC - m * logE
            x0_logE = energy_log[i, edge_onset_idx].item()
            log_C = outputs_log[edge_onset_idx].detach().item() + m * x0_logE

            powerlaw_logI = torch.tensor(log_C, device=self.device) - m * energy_log[i, edge_onset_idx:].to(self.device)

            # Stitch pre-edge NN prediction with post-edge power-law
            outputs[i, :edge_onset_idx] = outputs_log[:edge_onset_idx].detach().cpu().clone()
            outputs[i, edge_onset_idx:] = powerlaw_logI.detach().cpu().clone()

        return outputs


    # Diagnostics
    def check_fit_on_training_data(self):
        """Scatter: true vs predicted log-intensity in pre-edge."""
        self.model.eval()
        with torch.no_grad():
            x_sample, y_true = self.X, self.y

            mask = self._pre_edge_mask(x_sample)
            if mask.sum() == 0:
                print("No pre-edge points in sample")
                return

            x_sample, y_true = x_sample[mask].to(self.device), y_true[mask].to(self.device)
            y_pred = self.model(x_sample).squeeze()

            # undo standardization for plotting
            y_true_un = self._unnormalize_y(y_true)
            y_pred_un = self._unnormalize_y(y_pred)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0.2,0.2,0.7,0.7])    
        plt.scatter(y_true_un.cpu().numpy(), y_pred_un.cpu().numpy(), alpha=0.6, label="Data points", c = '#4B4DED')
        lims = [y_true_un.min().item(), y_true_un.max().item()]
        plt.plot(lims, lims, '--', label="Ideal", color = '#FF6F61')
        plt.xlabel("Training Values (a.u.)", fontsize=20)
        plt.ylabel("Prediction Values (a.u.)", fontsize=20)
        plt.xlim(lims)
        plt.ylim(lims)
        plt.xticks([round(lims[0],1), round(lims[1],1)], fontsize=20)
        plt.yticks([round(lims[0],1), round(lims[1],1)], fontsize=20)
        # plt.legend(fontsize=30)
        plt.savefig('training-prediction-plot.svg')
        # plt.title("NN Fit Check (pre-edge)")
        plt.show()

        plt.scatter(x_sample[:, 0].cpu().numpy(), y_true_un.cpu().numpy(), label="True")
        plt.scatter(x_sample[:, 0].cpu().numpy(), y_pred_un.cpu().numpy(), label="Predicted")
        plt.xlabel("Energy (normalized)")
        plt.ylabel("Log-intensity (pre-edge)")
        plt.legend()
        plt.title("NN Fit Check (pre-edge) - Energy vs Log-intensity")
        plt.show()

    def check_fit_interpolation(self, E_value=None, n_points=200, n_grid=200):
        """
        Plot NN interpolation at fixed Energy (X[:,0]) while varying TII (X[:,1]).
        Helps diagnose overfitting to TII.
        """
        self.model.eval()
        with torch.no_grad():
            X, y = self.X, self.y

            # pick an E value to slice at
            if E_value is None:
                # pick median of training distribution
                E_value = X[:,0].median().item()

            # extract real samples at this E (within tolerance)
            tol = 1e-6
            mask = (torch.abs(X[:,0] - E_value) < tol)
            X_real, y_real = X[mask], y[mask]

            # sweep across TII space
            t_min, t_max = X[:,1].min().item(), X[:,1].max().item()
            t_grid = torch.linspace(t_min, t_max, n_grid).to(X.device)
            E_grid = torch.full_like(t_grid, E_value)

            X_grid = torch.stack([E_grid, t_grid], dim=1).to(self.device)
            y_pred = self.model(X_grid).squeeze()

            # unnormalize
            y_pred_un = self._unnormalize_y(y_pred)
            if X_real.shape[0] > 0:
                y_real_un = self._unnormalize_y(y_real)

        plt.figure(figsize=(6,4), dpi=300)
        plt.plot(t_grid.cpu().numpy(), y_pred_un.cpu().numpy(), label="NN prediction", lw=2)

        if X_real.shape[0] > 0:
            plt.scatter(X_real[:,1].cpu().numpy(), y_real_un.cpu().numpy(),
                        color='r', alpha=0.6, label="True data")

        plt.xlabel("TII (X[:,1])")
        plt.ylabel("Log-intensity")
        plt.title(f"Interpolation at fixed Energy E={E_value:.3f}")
        plt.legend()
        plt.show()

    def train_with_epoch_predictions(self, epochs=200, lr=1e-3, batch_size=100,
                                     lambda_deriv=10.0, progress=False):
        """
        Train the NN model while saving predictions at every epoch.
        
        Returns
        -------
        all_preds : torch.Tensor
            Predictions on the training set for each epoch,
            shape (n_epochs, N_train).
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        dataset = TensorDataset(self.X, self.y)
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

            #store predictions at this epoch ---
            self.model.eval()
            with torch.no_grad():
                preds_norm = self.model(self.X.to(self.device)).squeeze()
                
                preds = self._unnormalize_y(preds_norm).cpu()  # back to raw log-intensity

            all_preds.append(preds.clone())

            if progress and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"[Epoch {epoch+1}/{epochs}] loss={loss.item():.4f}")

        return torch.stack(all_preds, dim=0)  # shape [epochs, N_train]
    
    def train_with_epoch_predictions_full_spectrum(self, x_eval, epochs=200, lr=1e-3, batch_size=100,
                                               lambda_deriv=10.0, progress=False):
        """
        Train the NN model while saving predictions at every epoch, including 
        the full evaluation spectrum with post-edge extrapolation.

        Parameters
        ----------
        x_eval : torch.Tensor
            Evaluation input for a single spectrum, shape [1, n_E, 2]

        Returns
        -------
        all_outputs : torch.Tensor
            Predictions for each epoch on the full spectrum including extrapolation,
            shape [n_epochs, n_E]
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        dataset = TensorDataset(self.X, self.y)
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

            #store predictions for full spectrum
            self.model.eval()
            with torch.no_grad():
                x_eval_for_model = x_eval.clone().detach().requires_grad_(True)
                outputs = self.evaluate_model(x_eval_for_model, None)
                outputs_linear = torch.exp(outputs).squeeze()  # back to linear scale

            all_outputs.append(outputs_linear.cpu().clone())

            if progress and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"[Epoch {epoch+1}/{epochs}] loss={loss.item():.4f}")

        return torch.stack(all_outputs, dim=0)  # shape [epochs, n_E]

class DataHandler:
    """
    Handles reading, storing, and preprocessing EELS data.
    """
    def __init__(self):
        self.signal = None
        self.spatial_axis_x = None
        self.spatial_axis_y = None
        self.energy_axis = None
        self.si_size = None
    
    def other_data(self, signal, spatial_axis_x, spatial_axis_y, energy_axis, 
                   low_loss=None, low_loss_energy_axis=None):
        """
        Initializes the DataHandler with provided data arrays.

        Parameters
        ----------
        signal : np.ndarray
            Core loss signal, shape (n_E, n_y*n_x).
        spatial_axis_x : np.ndarray
            Spatial axis in x direction.
        spatial_axis_y : np.ndarray
            Spatial axis in y direction.
        energy_axis : np.ndarray
            Energy axis.
        low_loss : np.ndarray, optional
            Low-loss signal data.
        low_loss_energy_axis : np.ndarray, optional
            Energy axis for low-loss data.
        """
        self.signal = signal
        self.spatial_axis_x = spatial_axis_x
        self.spatial_axis_y = spatial_axis_y
        self.energy_axis = energy_axis
        self.si_size = (energy_axis.size, spatial_axis_x.size, spatial_axis_y.size)
        if low_loss is not None:
            self.low_loss = low_loss
            self.low_loss_energy_axis = low_loss_energy_axis

    def read_dm3_linescan(self, path):
        """
        Reads a DM3 linescan file and extracts signal and axes.

        Parameters
        ----------
        path : str
            Path to DM3 file.
        """
        file = dm.file_reader(path)
        
        self.signal = file[0]['data'].T

        spatial_y_metadata = file[0]['axes'][0]
        self.spatial_axis_y = np.linspace(  
            spatial_y_metadata['offset'], 
            spatial_y_metadata['offset'] + spatial_y_metadata['scale'] * (spatial_y_metadata['size'] - 1), 
            spatial_y_metadata['size']
        )                                   #shape [n_y]

        self.spatial_axis_x = np.ones(1)  # shape [1] since it's a linescan

        energy_axis_metadata = file[0]['axes'][1]
        
        self.energy_axis = np.linspace( 
            energy_axis_metadata['offset'], 
            energy_axis_metadata['offset'] + energy_axis_metadata['scale'] * (energy_axis_metadata['size'] - 1), 
            energy_axis_metadata['size']
        )                                   #shape [n_E]

    def read_dm4_SI(self, path, core_loss_index = 3, lowloss = False):
        """
        Reads a DM4 SI file and extracts signal, axes, and optionally low-loss data.

        Parameters
        ----------
        path : str
            Path to DM4 file.
        core_loss_index : int, optional
            Index for core loss data (default: 3).
        lowloss : bool, optional
            If True, reads low-loss data (default: False).
        """
        file = dm.file_reader(path)
        data = file[core_loss_index]

        energy_axis_metadata = data['axes'][0]
        self.energy_axis = np.linspace( 
            energy_axis_metadata['offset'], 
            energy_axis_metadata['offset'] + energy_axis_metadata['scale'] * (energy_axis_metadata['size'] - 1), 
            energy_axis_metadata['size']
        )                                   #shape [n_E]

        spatial_y_metadata = data['axes'][1]
        self.spatial_axis_y = np.linspace(  
            spatial_y_metadata['offset'], 
            spatial_y_metadata['offset'] + spatial_y_metadata['scale'] * (spatial_y_metadata['size'] - 1), 
            spatial_y_metadata['size']
        )                                   #shape [n_y]

        spatial_x_metadata = data['axes'][2]
        self.spatial_axis_x = np.linspace(  
            spatial_x_metadata['offset'], 
            spatial_x_metadata['offset'] + spatial_x_metadata['scale'] * (spatial_x_metadata['size'] - 1), 
            spatial_x_metadata['size']
        )                                   #shape [n_x]

        self.si_size = (
            energy_axis_metadata['size'],
            spatial_y_metadata['size'], 
            spatial_x_metadata['size']
        )

        self.signal = data['data'][:,:].reshape(-1, spatial_x_metadata['size']*spatial_y_metadata['size']) # shape [n_E, n_y*n_x]

        if lowloss == True:
            self.low_loss = file[2]['data'].reshape(-1, spatial_x_metadata['size']*spatial_y_metadata['size'])    #shape [n_E_low_loss, n_y, n_x]

            low_loss_metadata = file[2]['axes'][0]

            self.low_loss_energy_axis = np.linspace(
                low_loss_metadata['offset'], 
                low_loss_metadata['offset'] + low_loss_metadata['scale'] * (low_loss_metadata['size'] - 1), 
                low_loss_metadata['size']
            )                                   #shape [n_E_low_loss]

    def align_data_cross_correlate(self, reference_spectrum_index = 0):
        """
        Aligns spectra by cross-correlation with a reference spectrum.

        Parameters
        ----------
        reference_spectrum_index : int, optional
            Index of reference spectrum (default: 0).
        """
        shifts = []
        reference_spectrum = self.signal[:,reference_spectrum_index]
        for i, x in enumerate(self.signal.T):
            cross_correlate = sp_signal.correlate(x, reference_spectrum, mode='full')
            shift = len(x) - np.argmax(cross_correlate)-1  # -1 to account for the full mode
            self.signal[:,i] = np.roll(x, shift)
            shifts.append(int(shift))
        #window data such that all spectra starts/ends at the same energy loss
        self.window_data(lower = self.energy_axis[max(shifts)], 
                         higher = self.energy_axis[min(shifts)] if min(shifts)<0 else  1e6)

    def align_data_ZLP(self):
        """
        Aligns spectra by shifting the zero-loss peak to a common position.
        """
        shifts = np.argmin(abs(self.low_loss_energy_axis))-np.argmax(self.low_loss, axis=0) #shape [n_y*n_x]
        # signal = self.signal
        for i, x in enumerate(self.signal.T):
            shift = shifts[i]
            self.signal[:,i] = np.roll(x, shift)
            self.low_loss[:,i] = np.roll(self.low_loss[:,i], shift)  # Align low-loss data as well
        #window data such that all spectra starts/ends at the same energy loss
        self.window_data(lower = self.energy_axis[np.max(shifts)], 
                         higher = self.energy_axis[np.min(shifts)] if np.min(shifts)<0 else  1e6)
        
    def window_data(self, lower, higher):
        """
        Windows the data to a specified energy loss range.

        Parameters
        ----------
        lower : float
            Lower bound of energy loss.
        higher : float
            Upper bound of energy loss.
        """
        window = (self.energy_axis<higher)&(self.energy_axis>lower)
        self.signal = self.signal[window]
        self.energy_axis = self.energy_axis[window]
        self.si_size = (self.signal.shape[0], self.spatial_axis_y.size, self.spatial_axis_x.size)

    def plot_spectra(self, spectra_indices=(1,2,3), energy_range = None, logscale = False, legend = False):
        """
        Plots selected spectra within a specified energy range.

        Parameters
        ----------
        spectra_indices : tuple, optional
            Indices of spectra to plot.
        energy_range : tuple, optional
            (min, max) energy range for plotting.
        """

        mask = (self.energy_axis > energy_range[0])&(self.energy_axis < energy_range[1])
        plt.figure(figsize=(10, 6))
        for i in spectra_indices:
            plt.plot(self.energy_axis[mask], self.signal[mask,i], label=f'Spectrum {i}')
        plt.xlabel('Energy Loss (eV)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Core Loss Spectra')
        if logscale:
            plt.yscale('log')
        if legend:
            plt.legend()
        plt.show()
        
    def plot_intensity_histogram(self, bins_nr = None):
        """
        Plots a histogram of total integrated intensity across all spectra.
        """


        total_integrated_intensity = np.sum(self.signal, axis=0).flatten()
        if bins_nr == None:
            bins_nr = int(len(total_integrated_intensity)*0.1)
        plt.hist(total_integrated_intensity, bins = bins_nr)

class Pooler:
    """
    Pools (averages) EELS data spatially using square or Gaussian kernels.
    """
    def __init__(self, signal, si_shape):
        '''
        Initializes the Pooler with the core loss data.
        
        Parameters:
        intensity_data : np.ndarray
            The core loss data to be pooled.
        '''
        self.signal = signal.copy().reshape(si_shape)  # Reshape to the specified shape
        # self.si_shape = si_shape
        
    def pool_data(self, sqr_radius, gaussian_kernel = False, sigma = None):
        """
        Pools the data over a square or Gaussian kernel.

        Parameters
        ----------
        sqr_radius : int
            Radius of pooling region.
        gaussian_kernel : bool, optional
            If True, uses Gaussian kernel (default: False).
        sigma : float, optional
            Standard deviation for Gaussian kernel.

        Returns
        -------
        np.ndarray
            Pooled signal, shape (n_E, n_y*n_x).
        """

        if gaussian_kernel:
            if sigma is None:
                sigma = sqr_radius / 2

            # Create a grid of (x, y) coordinates
            size = 2 * sqr_radius - 1
            x, y = np.meshgrid(np.linspace(-sqr_radius + 1, sqr_radius - 1, size),
                            np.linspace(-sqr_radius + 1, sqr_radius - 1, size))
            
            # Compute the Gaussian function
            kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            
            # Normalize the kernel to ensure the sum is 1
            kernel /= np.sum(kernel)
        else:
            kernel = np.ones((2*sqr_radius-1, 2*sqr_radius-1))/ ((2*sqr_radius-1)**2)

        for i in range(self.signal.shape[0]):
            # Convolve each spectrum with the kernel
            self.signal[i, :, :] = sp_signal.convolve2d(self.signal[i, :, :], kernel, mode='same', boundary='symm')

        return self.signal.reshape(-1, self.signal.shape[1]*self.signal.shape[2])  # Reshape back to 2D array


class ClusterAnalyzer:
    """
    Performs clustering and covariance analysis on EELS spectra.
    """
    def __init__(self, signal):
        self.signal = signal.copy() # shape [n_E, n_spectrum]
        
        self.cluster_centers = None
        self.clusters_mean = None
        self.clusters_covariance = None

    def cluster_data(self, n_clusters, pre_edge_mask=None):
        """
        Clusters spectra using KMeans and computes cluster statistics.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        pre_edge_mask : np.ndarray of bool, optional
            If provided, restricts statistics to pre-edge region.
        """
        n_E = self.signal.shape[0]
        self.total_integrated_intensity = np.sum(self.signal, axis=0)

        kmeans = KMeans(n_clusters=n_clusters)
        self.clusters = kmeans.fit_predict(
            np.log(self.total_integrated_intensity).reshape(-1, 1)
        )
        self.cluster_centers = kmeans.cluster_centers_

        # use full range unless mask provided
        if pre_edge_mask is None:
            use_signal = self.signal
        else:
            use_signal = self.signal[pre_edge_mask, :]
            n_E = use_signal.shape[0]

        self.clusters_mean = np.zeros((n_E, n_clusters))
        self.clusters_covariance = np.zeros((n_E, n_E, n_clusters))

        for i in range(n_clusters):
            cluster_data_log = np.log(
                np.clip(use_signal[:, self.clusters == i], a_min=1e-10, a_max=None)
            )
            self.clusters_mean[:, i] = np.mean(cluster_data_log, axis=1)
            # Use Ledoit-Wolf shrinkage estimator for covariance
            if cluster_data_log.shape[1] > 1:
                lw = LedoitWolf().fit(cluster_data_log.T)
                self.clusters_covariance[:, :, i] = lw.covariance_
            else:
                # Fallback to zeros or identity if only one sample in cluster
                self.clusters_covariance[:, :, i] = np.eye(cluster_data_log.shape[0])


    def cholesky_decomp(self):
        """
        Computes Cholesky decomposition of cluster covariance matrices.
        """
        self.triangular_matices = np.zeros_like(self.clusters_covariance)  # shape [n_E, n_E, n_clusters]

        for i in range(self.clusters_covariance.shape[2]):
            try:
                self.triangular_matices[:, :, i] = np.linalg.cholesky(self.clusters_covariance[:, :, i])
            except np.linalg.LinAlgError:
                print(f'Cholesky decomposition failed for cluster {i}. Using identity matrix instead.')
                self.triangular_matices[:, :, i] = np.eye(self.clusters_covariance.shape[0])    


class X_Builder:
    """
    Prepares input features for neural network training and evaluation.
    """
    def __init__(self, energy_axis):
        self.energy_axis = energy_axis


    def prepare_X_mc_data(self, cluster_centers, edge_onset, energy_range=None):
        """
        Prepares Monte Carlo input features for neural network.
        Restricts to pre-edge region only.

        Parameters
        ----------
        cluster_centers : array-like
            Cluster centers for input features.
        edge_onset : float
            Energy onset (all points below are kept).
        energy_range : tuple, optional
            Additional energy window (min, max).
        """
        #Pre-edge mask
        mask = self.energy_axis < edge_onset
        if energy_range is not None:
            mask &= (self.energy_axis >= energy_range[0]) & (self.energy_axis <= energy_range[1])

        energy_axis_masked = self.energy_axis[mask]
        num_energy_loss, num_clusters = len(energy_axis_masked), len(cluster_centers)

        # Expand energy axis for each cluster
        energy_axis_expanded = torch.tensor(energy_axis_masked, dtype = torch.float32).unsqueeze(0).expand(num_clusters, -1)  # [n_clusters, n_E_pre]

        # Expand cluster centers for each energy
        clustered_spectra_centers = torch.tensor(cluster_centers, dtype = torch.float32).squeeze()
        clustered_spectra_centers_expanded = clustered_spectra_centers.unsqueeze(1).expand(-1, num_energy_loss)  # [n_clusters, n_E_pre]

        # Final feature stack: (energy, cluster_center)
        X_mc = torch.stack([
            energy_axis_expanded,
            clustered_spectra_centers_expanded
        ], dim=2)  # [n_clusters, n_E_pre, 2]

        self.X_mc = X_mc.reshape(-1, 2)  # [n_clusters*n_E_pre, 2]
        self.pre_edge_mask = mask  # Save mask if needed elsewhere

    def prepare_X_eval_data(self, total_integrated_intensity):
        """
        Prepares evaluation input features for neural network.

        Parameters
        ----------
        total_integrated_intensity : array-like
            Total integrated intensity for each spectrum.
        """
        num_energy_loss, num_spectra = len(self.energy_axis), len(total_integrated_intensity) 

        energy_axis_expanded = torch.tensor(self.energy_axis, dtype = torch.float32).unsqueeze(0).expand(num_spectra, -1)
        total_integrated_intensity_expanded = torch.log(torch.tensor(total_integrated_intensity, dtype = torch.float32).unsqueeze(1).expand(-1, num_energy_loss))

        self.X_eval = torch.stack([
            energy_axis_expanded, 
            total_integrated_intensity_expanded
        ], dim=2)

class BackgroundTrainer:
    """
    Trains neural network background models using Monte Carlo replicas.
    """
    def __init__(self, signal, pre_edge_mask, X_mc, X_eval, clustered_spectra_mean, 
                 triangular_matices, covariance_matrices, cluster_labels):
        self.signal = signal.copy()

        self.pre_edge_mask = pre_edge_mask  # shape [n_E]

        self.X_mc = X_mc  # shape [n_cluster*n_E, 2]
        self.X_eval = X_eval  # shape [n_spectrum*n_E, 2]

        self.clustered_spectra_mean = clustered_spectra_mean  # shape [n_E, n_clusters]
        self.triangular_matices = triangular_matices  # shape [n_E, n_E, n_clusters]
        self.covariance_matrices = covariance_matrices  # shape [n_E, n_E, n_clusters]

        self.cluster_labels = cluster_labels 

    def _generate_mc_replica_local(self, cluster_id, K=50, rng=None):
        """
        Generate one log-replica for a given cluster:
        - Pick random anchor spectrum in cluster,
        - Estimate local covariance from K nearest neighbors,
        - Sample N(anchor, local_cov).
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.cluster_labels is None:
            raise ValueError("cluster_labels must be provided to use replica_version='local'")

        signal = self.signal if self.pre_edge_mask is None else self.signal[self.pre_edge_mask, :]
        mask = (self.cluster_labels == cluster_id)
        X_log = np.log(np.clip(signal[:, mask], 1e-10, None)).T  # shape (n_members, n_E)

        if X_log.shape[0] == 0:
            # fallback to cluster mean
            return self.clustered_spectra_mean[:, cluster_id].copy()

        # pick random anchor
        anchor_idx = rng.integers(0, X_log.shape[0])
        x_anchor = X_log[anchor_idx]

        # local neighbors
        K_eff = min(K, X_log.shape[0])
        nn = NearestNeighbors(n_neighbors=K_eff).fit(X_log)
        idxs = nn.kneighbors([x_anchor], return_distance=False)
        local = X_log[idxs[0]]

        # shrinkage covariance
        lw = LedoitWolf().fit(local)
        Sigma = lw.covariance_

        # sample
        z = rng.multivariate_normal(np.zeros(x_anchor.size), Sigma)
        return x_anchor + z

    def _generate_mc_replica_local_all_clusters(self, K=50, rng=None):
        n_E, n_clusters = self.clustered_spectra_mean.shape
        out = np.zeros((n_E, n_clusters))
        for c in range(n_clusters):
            out[:, c] = self._generate_mc_replica_local(c, K=K, rng=rng)
        return out
    

    def _generate_mc_replica_triangular(self):
        """
        Generates a Monte Carlo replica in log space.

        Returns
        -------
        np.ndarray
            Log-space MC replica, shape (n_E, n_clusters).
        """
        mc_replica_log = np.zeros_like(self.clustered_spectra_mean)  # shape [n_E, n_clusters]
        n_E = self.clustered_spectra_mean.shape[0]

        for cluster_id in range(mc_replica_log.shape[1]):
            z = np.random.randn(n_E)
            L = self.triangular_matices[:, :, cluster_id]  # shape [n_E, n_E]
            mc_replica_log[:, cluster_id] = self.clustered_spectra_mean[:, cluster_id] + np.dot(L, z)  # sample in log-space

        return mc_replica_log  # log-space MC replicas
    
    def _generate_mc_replica_covariance(self):
        """
        Generates a Monte Carlo replica in log space from the covariance matrices

        Returns
        -------
        np.ndarray
            Log-space MC replica, shape (n_E, n_clusters).
        """
        mc_replica_log = np.zeros_like(self.clustered_spectra_mean)  # shape [n_E, n_clusters]
        n_E = self.clustered_spectra_mean.shape[0]

        for cluster_id in range(mc_replica_log.shape[1]):
            z = np.random.multivariate_normal(mean=np.zeros(n_E), cov=self.covariance_matrices[:, :, cluster_id])
            mc_replica_log[:, cluster_id] = self.clustered_spectra_mean[:, cluster_id] + z  # sample in log-space

        return mc_replica_log  # log-space MC replicas

    
    def train_MC_replica_consecutive(self, n_mc_replicas, epochs, edge_onset, model, progress=False, replica_version='triangular', logging = False):
        """
        Trains neural network background models for multiple MC replicas.

        Parameters
        ----------
        n_mc_replicas : int
            Number of Monte Carlo replicas.
        epochs : int
            Number of training epochs per replica.
        edge_onset : float
            Energy value for edge onset.
        """

        self.background = np.zeros((n_mc_replicas, self.signal.shape[1], self.signal.shape[0])) # shape [n_mc_replicas, n_spectrum, n_E]
        self.effective_exponents = np.zeros((n_mc_replicas, self.signal.shape[1], self.signal.shape[0]))  # shape [n_mc_replicas, n_spectrum, n_E]
    

        for i in range(n_mc_replicas):
            print(f'Starting Replica {i+1}/{n_mc_replicas}')
            if replica_version == 'triangular':
                mc_replica = self._generate_mc_replica_triangular()
            elif replica_version == 'covariance':
                mc_replica = self._generate_mc_replica_covariance()
            elif replica_version == 'local':
                mc_replica = self._generate_mc_replica_local_all_clusters(K=50)
            else:
                raise ValueError("replica_version must be either 'triangular' or 'covariance'")

            y = torch.tensor(mc_replica.T, dtype = torch.float32).reshape(-1)  # [n_clusters*n_E_pre]
            if logging:
                print("X_mc shape:", self.X_mc.shape)   # should be [n_clusters * n_E_pre, 2]
                print("y shape:", y.shape)              # should match exactly
                print(np.where(y>13), self.X_mc[np.where(y>13)])
                plt.scatter(self.X_mc[:,0], y, s=3, alpha=0.5)
                plt.title("X_mc[:,0] vs y before training")
                plt.show()
            NN_object = NNTrainer(self.X_mc, y, edge_onset, model)
            NN_object.train(epochs=epochs, progress=progress)
            
            if logging:
                NN_object.check_fit_on_training_data(n_points=300)
                NN_object.check_fit_interpolation()
            
            self.background[i] = np.exp(NN_object.evaluate_model(self.X_eval, self.signal.T).detach().numpy())  # convert back to original scale

    def check_mc_replicas_vs_clusters(self, n_samples=20, cluster_id=0):
            """
            Compare original cluster spectra with MC replicas for a chosen cluster.

            Parameters
            ----------
            n_samples : int
                Number of MC replicas to draw.
            cluster_id : int
                Cluster index to visualize.
            """
            # Original cluster mean (log-space)
            cluster_mean = self.clustered_spectra_mean[:, cluster_id]

            # Draw replicas
            replicas = [self._generate_mc_replica_covariance()[:, cluster_id] for _ in range(n_samples)]
            replicas = np.stack(replicas, axis=0)  # shape (n_samples, n_E)

            # plt.figure(figsize=(10, 6))
            plt.plot(cluster_mean, label="Cluster mean", color="black", linewidth=2)
            plt.fill_between(
                np.arange(len(cluster_mean)),
                np.percentile(replicas, 5, axis=0),
                np.percentile(replicas, 95, axis=0),
                alpha=0.5, label="MC 5–95% interval"
            )

            plt.title(f"MC Replicas vs. ")
            plt.xlabel("Energy index")
            plt.ylabel("Log Intensity")
            plt.legend()
            plt.show()

    def run_single_replica_with_predictions(self, x_eval, epochs, edge_onset, model,
                                            replica_version='triangular', K=50,
                                            progress=False):
        """
        Runs one MC replica, trains with epoch predictions,
        and returns predictions for each epoch.

        Returns
        -------
        predictions : np.ndarray
            Shape (epochs, n_spectrum, n_E), predictions in original scale.
        """
        if replica_version == 'triangular':
            mc_replica = self._generate_mc_replica_triangular()
        elif replica_version == 'covariance':
            mc_replica = self._generate_mc_replica_covariance()
        elif replica_version == 'local':
            mc_replica = self._generate_mc_replica_local_all_clusters(K=K)
        else:
            raise ValueError("replica_version must be 'triangular' or 'covariance' or 'local'")

        # Prepare y for training
        y = torch.tensor(mc_replica.T, dtype=torch.float32).reshape(-1)

        NN_object = NNTrainer(self.X_mc, y, edge_onset, model)
        epoch_preds = NN_object.train_with_epoch_predictions_full_spectrum(x_eval=x_eval,
            epochs=epochs, progress=progress
        )  # [epochs, N_train]
        return epoch_preds.cpu().numpy(), y  # shape [epochs, n_spectrum, n_E]

class PredictionSaver:
    """
    Saves prediction results and associated metadata to disk.
    """
    def __init__(self, signal, energy_axis, spatial_axis_x, spatial_axis_y, predictions):
        self.signal = signal
        self.energy_axis = energy_axis
        self.spatial_axis_x = spatial_axis_x
        self.spatial_axis_y = spatial_axis_y
        self.predictions = predictions

    def save_predictions(self, path):
        """
        Saves predictions and metadata to a .npz file.

        Parameters
        ----------
        path : str
            File path to save predictions.
        """
        np.savez(path, 
                 signal=self.signal, 
                 energy_axis=self.energy_axis, 
                 spatial_axis_x=self.spatial_axis_x, 
                 spatial_axis_y=self.spatial_axis_y, 
                 predictions=self.predictions)
