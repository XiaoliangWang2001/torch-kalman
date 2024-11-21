import torch
import torch.nn as nn
import warnings
from standard import KalmanFilter


class UDUDecomposition(nn.Module):
    """Represents a UDU' decomposition of a matrix as a PyTorch Module"""

    def __init__(self, U: torch.Tensor, D: torch.Tensor):
        super().__init__()
        # Register tensors as parameters or buffers
        # Use register_buffer if these shouldn't be updated during training
        # Use register_parameter if they should be trainable
        self.register_buffer("U", U)
        self.register_buffer("D", D)

    def forward(self) -> torch.Tensor:
        """Reconstruct the original matrix from UDU' decomposition"""
        return self.reconstruct()

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct the original matrix from UDU' decomposition"""
        return torch.matmul(
            torch.matmul(self.U, torch.diag_embed(self.D)), self.U.transpose(-2, -1)
        )

    def transpose(self, *args):
        """Return transpose of the reconstructed matrix"""
        # For a UDU' decomposition, the transpose is the same as the original
        # since (UDU')' = U(D')U' = UDU'
        return self.reconstruct().transpose(*args)
    
class UDUTensor:
    """A tensor-like container for batches of UDU decompositions that preserves gradients"""

    def __init__(self, batch_size, n_timesteps, n_dim_state, device):
        # Initialize with identity matrices for U and zeros for D
        self.U = torch.eye(n_dim_state, device=device).expand(
            batch_size, n_timesteps, n_dim_state, n_dim_state
        ).clone()
        self.D = torch.zeros(batch_size, n_timesteps, n_dim_state, device=device)

    def __getitem__(self, idx):
        # Handle both single index and tuple indices
        if isinstance(idx, tuple):
            batch_idx, time_idx = idx
            U = self.U[batch_idx, time_idx]
            D = self.D[batch_idx, time_idx]
        else:
            U = self.U[idx]
            D = self.D[idx]
        return UDUDecomposition(U, D)

    def __setitem__(self, idx, value):
        if isinstance(value, UDUDecomposition):
            U, D = value.U, value.D
        else:
            U, D = value
            
        # Handle input of different shape
        match len(U.shape):
            case 4:
                pass
            case 3:
                U = U.unsqueeze(1) # add time dimension
                D = D.unsqueeze(1)
                # Add warning
                warnings.warn("UDUTensor: adding time dimension to U and D, please check if this is correct")
            case 2:
                U = U[None, None, ...]
                D = D[None, None, ...]
            case _:
                raise ValueError(f"Expected U to have 2 or 3 or 4 dimensions, but got {len(U.shape)}")

        # Determine the indexing scheme
        if isinstance(idx, tuple):
            batch_idx, time_idx = idx
        else:
            batch_idx, time_idx = idx, slice(None)

         # Assert that U has the expected shape (B, T, D, D) - length 4
        assert len(U.shape) == 4, f"Expected U to have 4 dimensions, but got {len(U.shape)}"
        assert U.shape[-2:] == self.U.shape[-2:], f"Shape mismatch for U's last two dimensions: {U.shape[-2:]} vs {self.U.shape[-2:]}"

        # Assert that D has the expected shape (B, T, D)
        assert len(D.shape) == 3, f"Expected D to have 3 dimensions, but got {len(D.shape)}"
        assert D.shape[-1] == self.D.shape[-1], f"Shape mismatch for D's last dimension: {D.shape[-1]} vs {self.D.shape[-1]}"

        # Create the mask
        mask = torch.zeros_like(self.U, dtype=torch.bool)
        mask[batch_idx, time_idx] = True

        # Use torch.where for the non-inplace update
        self.U = torch.where(mask, U, self.U)
        self.D = torch.where(mask[..., 0], D, self.D)

    @property
    def shape(self):
        return self.U.shape
    
    def reconstruct(self, idx=None):
        """Reconstruct the full covariance matrices"""
        if idx is None:
            return torch.matmul(
                torch.matmul(self.U, torch.diag_embed(self.D)), self.U.transpose(-2, -1)
            )
        # Handle both single index and tuple indices
        if isinstance(idx, tuple):
            batch_idx, time_idx = idx
            U = self.U[batch_idx, time_idx]
            D = self.D[batch_idx, time_idx]
        else:
            U = self.U[idx]
            D = self.D[idx]
        return torch.matmul(
            torch.matmul(U, torch.diag_embed(D)),
            U.transpose(-2, -1),
        )


def udu(M: torch.Tensor) -> UDUDecomposition:
    """Construct the UDU' decomposition of a positive semidefinite matrix M

    Parameters
    ----------
    M : [..., n, n] tensor
        Batch of matrices to factorize

    Returns
    -------
    UDU : UDUDecomposition
        UDU' representation of M
    """
    # Check symmetry
    if not torch.allclose(M, M.transpose(-2, -1), rtol=1e-10, atol=1e-10):
        raise ValueError("M must be symmetric, positive semidefinite")

    n = M.shape[-1]
    batch_dims = M.shape[:-2]

    # Initialize U and d with better numerical precision
    U = torch.eye(n, device=M.device, dtype=M.dtype).expand(*batch_dims, n, n).clone()
    d = torch.zeros(*batch_dims, n, device=M.device, dtype=M.dtype)

    # Get upper triangular part and ensure we maintain gradients
    M_working = torch.triu(M.clone())

    # Small constant for numerical stability
    eps = torch.finfo(M.dtype).eps

    # Perform Bierman's COV2UD algorithm with improved numerical stability
    for j in range(n - 1, 0, -1):
        d[..., j] = torch.clamp(M_working[..., j, j], min=eps)
        alpha = torch.where(
            d[..., j] > eps, 1.0 / d[..., j], torch.zeros_like(d[..., j])
        )

        for k in range(j):
            beta = M_working[..., k, j]
            U[..., k, j] = alpha * beta

            # Fix the broadcasting issue and improve numerical stability
            beta_expanded = beta.unsqueeze(-1).expand(*batch_dims, k + 1)
            U_slice = U[..., : k + 1, j]

            # Use fused multiply-add for better numerical precision
            M_working[..., : k + 1, k] = torch.addcmul(
                M_working[..., : k + 1, k], beta_expanded, U_slice, value=-1.0
            )

    # Handle the final diagonal element
    d[..., 0] = torch.clamp(M_working[..., 0, 0], min=eps)

    # Ensure symmetry in the result
    U = torch.triu(U)

    return UDUDecomposition(U, d)





class BiermanKalmanFilter(KalmanFilter):
    """Kalman Filter implementation using Bierman's UDU' factorization for improved numerical stability"""

    def __init__(self, compile_mode: bool = False, diagonal_only: bool = False):
        self.diagonal_only = diagonal_only
        super().__init__(compile_mode)
        self.udu_decompositions = nn.ModuleDict()

    def _decorrelate_observations(
        self,
        observation_matrices,
        observation_offsets,
        observation_covariance,
        observations,
    ):
        """Make each coordinate of all observations independent

        Modify observations and all associated parameters such that all observation
        indices are expected to be independent.

        Parameters
        ----------
        observation_matrices : [..., n_timesteps, n_dim_obs, n_dim_obs] or [..., n_dim_obs, n_dim_obs] tensor
            observation matrix
        observation_offsets : [..., n_timesteps, n_dim_obs] or [..., n_dim_obs] tensor
            observations for times [0...n_timesteps-1]
        observation_covariance : [..., n_timesteps, n_dim_obs, n_dim_obs] or [..., n_dim_obs, n_dim_obs] tensor
            observation covariance matrix
        observations : [..., n_timesteps, n_dim_obs] tensor
            observations from times [0...n_timesteps-1]

        Returns
        -------
        observation_matrices2 : [..., n_timesteps, n_dim_obs, n_dim_obs] or [..., n_dim_obs, n_dim_obs] tensor
            observation matrix with each index decorrelated
        observation_offsets2 : [..., n_timesteps, n_dim_obs] or [..., n_dim_obs] tensor
            observations for times [0...n_timesteps-1] with each index decorrelated
        observation_covariance2 : [..., n_timesteps, n_dim_obs, n_dim_obs] or [..., n_dim_obs, n_dim_obs] tensor
            observation covariance matrix with each index decorrelated (diagonal)
        observations2 : [..., n_timesteps, n_dim_obs] tensor
            observations from times [0...n_timesteps-1] with each index decorrelated
        """
        n_dim_obs = observations.shape[-1]
        device = observations.device
        dtype = observations.dtype

        # Calculate Cholesky decomposition (R^{1/2})
        L = torch.linalg.cholesky(observation_covariance, upper=False)

        # Calculate (R^{1/2})^{-1}
        L_inv = torch.linalg.inv(L)  # shape [..., (n_timesteps), n_dim_obs, n_dim_obs]

        # Decorrelate observation_matrices
        # observation_matrices2 = torch.matmul(L_inv, observation_matrices)
        observation_matrices2 = torch.einsum(
            "...ij,...jk->...ik", L_inv, observation_matrices
        )

        # Decorrelate observation_offsets
        observation_offsets2 = torch.einsum(
            "...ij,...j->...i", L_inv, observation_offsets
        )

        # Decorrelate observations
        observations2 = torch.einsum("...ij,...j->...i", L_inv, observations)

        # Create identity covariance matrix with proper broadcasting
        observation_covariance2 = torch.eye(n_dim_obs, device=device, dtype=dtype)
        if len(observation_covariance.shape) > 2:
            observation_covariance2 = observation_covariance2.expand(
                *observation_covariance.shape[:-2], n_dim_obs, n_dim_obs
            )

        return (
            observation_matrices2,
            observation_offsets2,
            observation_covariance2,
            observations2,
        )

    def _filter_correct_single(
        self,
        h: torch.Tensor,  # [batch, n_dim_state]
        R: torch.Tensor,  # [batch]
        UDU: UDUDecomposition,
    ) -> tuple[UDUDecomposition, torch.Tensor]:  # Returns (UDU, k)
        """Process a single observation using Bierman's update algorithm.
        
        Args:
            h: Single row of observation matrix [batch, n_dim_state]
            R: Single diagonal element of observation covariance [batch]
            UDU: Current state covariance in UDU form
        
        Returns:
            tuple: (Updated UDU decomposition, Kalman gain for this observation)
        """
        batch_size = h.shape[0]
        n_dim_state = h.shape[-1]
        device = h.device

        U = UDU.U.clone()  # [batch, state, state]
        D = UDU.D.clone()  # [batch, state]
        
        # Initial computations
        f = torch.einsum('...i,...ij->...j', h, U)  # [batch, state]
        g = f * D  # [batch, state]
        alpha = torch.einsum('...i,...i->...', f, g) + R  # [batch]

        # Initialize arrays
        gamma = torch.zeros(batch_size, n_dim_state, device=device)
        U_bar = torch.zeros_like(U)
        D_bar = torch.zeros_like(D)
        k = torch.zeros(batch_size, n_dim_state, device=device)

        # Initial values
        gamma[..., 0] = R + g[..., 0] * f[..., 0]
        D_bar[..., 0] = D[..., 0] * R / gamma[..., 0]
        k[..., 0] = g[..., 0]
        U_bar[..., 0, 0] = 1

        # Sequential update
        for j in range(1, n_dim_state):
            gamma[..., j] = gamma[..., j-1] + g[..., j] * f[..., j]
            D_bar[..., j] = D[..., j] * gamma[..., j-1] / gamma[..., j]
            U_bar[..., :, j] = U[..., :, j] - (f[..., j] / gamma[..., j-1]).unsqueeze(-1) * k
            k = k + g[..., j].unsqueeze(-1) * U[..., :, j]

        return UDUDecomposition(U_bar, D_bar), k / alpha.unsqueeze(-1)

    def _filter_correct(
        self,
        observation_matrix: torch.Tensor,  # [batch, obs, state]
        observation_covariance: torch.Tensor,  # [batch, obs, obs]
        observation_offset: torch.Tensor,  # [batch, obs]
        predicted_state_mean: torch.Tensor,  # [batch, state]
        predicted_state_covariance: UDUDecomposition,
        observation: torch.Tensor,  # [batch, obs]
    ) -> tuple[torch.Tensor, UDUDecomposition]:
        """Apply Bierman's update algorithm sequentially for each observation."""
        #batch_size = observation.shape[0]
        n_dim_obs = observation.shape[-1]
        #device = observation.device

        # Initialize outputs
        corrected_state_mean = predicted_state_mean.clone()
        current_UDU = predicted_state_covariance

        # Process one observation at a time
        for i in range(n_dim_obs):
            # Extract observation components
            h = observation_matrix[..., i, :]  # [batch, state]
            R = observation_covariance[..., i, i]  # [batch]
            y = observation[..., i]  # [batch]
            d = observation_offset[..., i]  # [batch]

            # Predict current observation
            pred_obs = torch.einsum('...i,...i->...', h, corrected_state_mean) + d
            innovation = y - pred_obs  # [batch]

            # Update state covariance and get Kalman gain
            current_UDU, k = self._filter_correct_single(h, R, current_UDU)

            # Update state mean
            corrected_state_mean = corrected_state_mean + k * innovation.unsqueeze(-1)

        return corrected_state_mean, current_UDU

    def _filter_predict(
        self,
        transition_matrix,
        transition_covariance,
        transition_offset,
        current_state_mean,
        current_state_covariance: UDUDecomposition,
    ):
        """Predict next state distribution using UDU decomposition."""
        # Predict next state mean
        predicted_state_mean = (
            torch.einsum('...ij,...j->...i', transition_matrix, current_state_mean) 
            + transition_offset
        )

        # Predict next state covariance
        current_cov = current_state_covariance.reconstruct()

        # P = A*P*A' + Q in a single einsum operation
        predicted_cov = (
            torch.einsum('...ij,...jk,...lk->...il', transition_matrix, current_cov, transition_matrix) 
            + transition_covariance
        )

        # Convert final result to UDU form
        predicted_udu = udu(predicted_cov)

        return predicted_state_mean, predicted_udu

    def _filter(
        self,
        observations,
        transition_matrices,
        observation_matrices,
        transition_covariance,
        observation_covariance,
        transition_offsets,
        observation_offsets,
        initial_state_mean,
        initial_state_covariance,
    ):
        """Apply Kalman Filter using UDU decomposition throughout."""
        if not self.diagonal_only:
            (
                observation_matrices,
                observation_offsets,
                observation_covariance,
                observations,
            ) = self._decorrelate_observations(
                observation_matrices,
                observation_offsets,
                observation_covariance,
                observations,
            )
        batch_size, n_timesteps, n_dim_obs = observations.shape
        n_dim_state = transition_matrices.shape[-1]

        filtered_means = torch.zeros(
            batch_size, n_timesteps, n_dim_state, device=observations.device
        )
        filtered_covs = UDUTensor(
            batch_size, n_timesteps, n_dim_state, observations.device
        )
        predicted_means = torch.zeros_like(filtered_means)
        predicted_covs = UDUTensor(
            batch_size, n_timesteps, n_dim_state, observations.device
        )

        # Initialize first timestep with UDU decomposition
        time_indices = torch.arange(n_timesteps, device=observations.device)
        predicted_means = torch.where(
            (time_indices == 0).view(1, -1, 1),
            initial_state_mean.unsqueeze(1),
            predicted_means,
        )

        initial_udu = udu(initial_state_covariance)
        predicted_covs[:, 0] = initial_udu

        for t in range(n_timesteps):
            # Convert predicted covariance to UDU form
            predicted_udu = predicted_covs[:, t]

            # Correct
            filtered_means_t, filtered_covs_t = self._filter_correct(
                observation_matrices[:, t],
                observation_covariance[:, t],
                observation_offsets[:, t],
                predicted_means[:, t],
                predicted_udu,  # Pass UDU decomposition
                observations[:, t],
            )

            # Store results
            filtered_means = torch.where(
                (time_indices == t).view(1, -1, 1),
                filtered_means_t.unsqueeze(1),
                filtered_means,
            )
            filtered_covs[:, time_indices == t] = filtered_covs_t

            # Predict next state
            if t < n_timesteps - 1:
                # Convert filtered covariance to UDU form
                filtered_udu = filtered_covs[:, t]

                pred_mean, pred_udu = self._filter_predict(
                    transition_matrices[:, t],
                    transition_covariance[:, t],
                    transition_offsets[:, t],
                    filtered_means[:, t],
                    filtered_udu,  # Pass UDU decomposition
                )

                predicted_means = torch.where(
                    (time_indices == t + 1).view(1, -1, 1),
                    pred_mean.unsqueeze(1),
                    predicted_means,
                )
                predicted_covs[:, t + 1] = pred_udu

        return filtered_means, filtered_covs, predicted_means, predicted_covs
    
    
    def filter(self, *args, **kwargs):
        """Apply Kalman Filter using UDU decomposition throughout, reconstructing covariances after filtering."""
        filtered_means, filtered_udu, predicted_means, predicted_udu = self._filter(*args, **kwargs)
        filtered_covs = filtered_udu.reconstruct()
        predicted_covs = predicted_udu.reconstruct()
        return filtered_means, filtered_covs, predicted_means, predicted_covs

