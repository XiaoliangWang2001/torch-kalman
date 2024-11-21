import torch
from bierman import BiermanKalmanFilter, UDUDecomposition, udu, UDUTensor
import warnings

class RectIdentityBiermanFilter(BiermanKalmanFilter):
    def __init__(self, n_dim_state: int, n_dim_obs: int, compile_mode: bool = False):
        warnings.warn("RectIdentityBiermanFilter is experimental. Produces same results but is slightly slower than BiermanKalmanFilter.")
        super().__init__(compile_mode=False, diagonal_only=True)  # Don't compile parent methods
        if n_dim_obs >= n_dim_state:
            raise ValueError("n_dim_obs must be less than n_dim_state")
        
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self.compile_mode = compile_mode
        
        # Pre-allocate unit vectors for each observation dimension
        self.unit_vectors = []
        for i in range(n_dim_obs):
            unit_vec = torch.zeros(n_dim_state)
            unit_vec[i] = 1.0
            self.unit_vectors.append(unit_vec)
        if compile_mode:
            self.compiled_filter = torch.compile(self.filter)
            self.compiled_smooth = torch.compile(self.smooth)

    # def _filter_correct(
    #         self,
    #         observation_matrix: torch.Tensor,  # [batch, obs, state] (ignored - we know it's rectangular identity)
    #         observation_covariance: torch.Tensor,  # [batch, obs, obs] (ignored - we know it's identity)
    #         observation_offset: torch.Tensor,  # [batch, obs] (ignored - we know it's zero)
    #         predicted_state_mean: torch.Tensor,  # [batch, state]
    #         predicted_state_covariance: UDUDecomposition,
    #         observation: torch.Tensor,  # [batch, obs]
    #     ) -> tuple[torch.Tensor, UDUDecomposition]:
    #         """Specialized correction step for rectangular identity observation matrix."""
    #         batch_size = observation.shape[0]
    #         device = observation.device

    #         # Initialize outputs
    #         corrected_state_mean = predicted_state_mean.clone()
    #         current_UDU = predicted_state_covariance

    #         # Process one observation at a time for the first n_dim_obs components
    #         for i in range(self.n_dim_obs):
    #             # For rectangular identity matrix, h is a unit vector
    #             h = torch.zeros(batch_size, self.n_dim_state, device=device)
    #             h[:, i] = 1.0
                
    #             # R is 1.0 (unit observation covariance)
    #             R = torch.ones(batch_size, device=device)
                
    #             # Innovation is direct difference for observed components
    #             innovation = observation[..., i] - corrected_state_mean[..., i]

    #             # Update state covariance and get Kalman gain
    #             current_UDU, k = self._filter_correct_single(h, R, current_UDU)

    #             # Update state mean
    #             corrected_state_mean = corrected_state_mean + k * innovation.unsqueeze(-1)

    #         return corrected_state_mean, current_UDU
    
    def _filter_correct(
        self,
        observation_matrix: torch.Tensor,  # ignored
        observation_covariance: torch.Tensor,  # ignored
        observation_offset: torch.Tensor,  # ignored
        predicted_state_mean: torch.Tensor,  # [batch, state]
        predicted_state_covariance: UDUDecomposition,
        observation: torch.Tensor,  # [batch, obs]
    ) -> tuple[torch.Tensor, UDUDecomposition]:
        """Specialized correction step for rectangular identity observation matrix."""
        batch_size = observation.shape[0]
        device = observation.device

        # Initialize outputs
        corrected_state_mean = predicted_state_mean.clone()
        current_UDU = predicted_state_covariance

        # Process one observation at a time for the first n_dim_obs components
        for i in range(self.n_dim_obs):
            # Use pre-allocated unit vector
            h = self.unit_vectors[i].to(device).expand(batch_size, -1)
            
            # R is 1.0 (unit observation covariance)
            R = torch.ones(batch_size, device=device)
            
            # Innovation is direct difference for observed components
            innovation = observation[..., i] - corrected_state_mean[..., i]

            # Update state covariance and get Kalman gain
            current_UDU, k = self._filter_correct_single(h, R, current_UDU)

            # Update state mean
            corrected_state_mean = corrected_state_mean + k * innovation.unsqueeze(-1)

        return corrected_state_mean, current_UDU
        
        


    def _smooth(
        self,
        filtered_means,
        filtered_covs,
        predicted_means,
        predicted_covs,
        transition_matrices,
    ):
        """Specialized RTS smoother implementation."""
        batch_size, n_timesteps, _ = filtered_means.shape

        # Initialize with new tensors
        smoothed_means = torch.zeros_like(filtered_means)
        smoothed_covs = torch.zeros_like(filtered_covs)

        # Initialize last timestep
        smoothed_means[:, -1] = filtered_means[:, -1]
        smoothed_covs[:, -1] = filtered_covs[:, -1]

        for t in range(n_timesteps - 2, -1, -1):
            # Compute gain matrix
            gain = torch.matmul(
                torch.matmul(
                    filtered_covs[:, t], transition_matrices[:, t].transpose(-2, -1)
                ),
                torch.linalg.pinv(predicted_covs[:, t + 1]),
            )

            # Compute smoothed mean
            mean_diff = (smoothed_means[:, t + 1] - predicted_means[:, t + 1]).unsqueeze(-1)
            smoothed_means[:, t] = filtered_means[:, t] + torch.matmul(gain, mean_diff).squeeze(-1)

            # Compute smoothed covariance
            cov_diff = smoothed_covs[:, t + 1] - predicted_covs[:, t + 1]
            smoothed_covs[:, t] = filtered_covs[:, t] + torch.matmul(
                torch.matmul(gain, cov_diff),
                gain.transpose(-2, -1)
            )

        return smoothed_means, smoothed_covs
