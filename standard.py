import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    """PyTorch implementation of Kalman Filter and Smoother.

    This implementation assumes all parameters (transition/observation matrices etc.)
    are provided at runtime and focuses only on computing filtered and smoothed
    state distributions.

    The model follows the standard linear-Gaussian state space model:
        x_{t+1} = A_{t} x_{t} + b_{t} + N(0, Q_{t})
        y_{t} = C_{t} x_{t} + d_{t} + N(0, R_{t})
    """

    def __init__(self, compile_mode: bool = False):
        super().__init__()
        self.compile_mode = compile_mode
        if compile_mode:
            self.compiled_filter = torch.compile(self.filter)
            self.compiled_smooth = torch.compile(self.smooth)

    def _filter_predict(
        self,
        transition_matrix,
        transition_covariance,
        transition_offset,
        current_state_mean,
        current_state_covariance,
        control,
        control_matrix,
    ):
        """Predict next state distribution given current state distribution."""
        # Ensure current_state_mean has correct shape for matmul
        if len(current_state_mean.shape) == 2:
            current_state_mean = current_state_mean.unsqueeze(-1)

        # Predict next state mean
        predicted_state_mean = (
            torch.matmul(transition_matrix, current_state_mean).squeeze(-1)
            + torch.matmul(control_matrix, control).squeeze(-1)
            + transition_offset
        )

        # Predict next state covariance
        predicted_state_covariance = (
            torch.matmul(
                torch.matmul(transition_matrix, current_state_covariance),
                transition_matrix.transpose(-2, -1),
            )
            + transition_covariance
        )

        return predicted_state_mean, predicted_state_covariance

    def _filter_correct(
        self,
        observation_matrix,
        observation_covariance,
        observation_offset,
        predicted_state_mean,
        predicted_state_covariance,
        observation,
    ):
        """Correct predicted state with observation."""
        # Ensure predicted_state_mean has correct shape for matmul
        if len(predicted_state_mean.shape) == 2:
            predicted_state_mean = predicted_state_mean.unsqueeze(-1)

        # Compute predicted observation mean
        predicted_observation_mean = (
            torch.matmul(observation_matrix, predicted_state_mean).squeeze(-1)
            + observation_offset
        )

        # Compute predicted observation covariance
        predicted_observation_covariance = (
            torch.matmul(
                torch.matmul(observation_matrix, predicted_state_covariance),
                observation_matrix.transpose(-2, -1),
            )
            + observation_covariance
        )

        # Compute Kalman gain
        kalman_gain = torch.matmul(
            torch.matmul(
                predicted_state_covariance, observation_matrix.transpose(-2, -1)
            ),
            torch.linalg.pinv(predicted_observation_covariance),
        )

        # Ensure innovation has correct shape for matmul
        innovation = (observation - predicted_observation_mean).unsqueeze(-1)

        # Compute corrected state
        corrected_state_mean = predicted_state_mean.squeeze(-1) + torch.matmul(
            kalman_gain, innovation
        ).squeeze(-1)

        corrected_state_covariance = predicted_state_covariance - torch.matmul(
            torch.matmul(kalman_gain, observation_matrix), predicted_state_covariance
        )

        return kalman_gain, corrected_state_mean, corrected_state_covariance

    def _filter(
        self,
        observations,
        controls,
        transition_matrices,
        control_matrices,
        observation_matrices,
        transition_covariance,
        observation_covariance,
        transition_offsets,
        observation_offsets,
        initial_state_mean,
        initial_state_covariance,
    ):
        """Apply Kalman Filter to estimate state distributions.

        Parameters
        ----------
        observations : [batch_size, n_timesteps, n_dim_obs] tensor
            Observations for times [0...n_timesteps-1]
        controls : [batch_size, n_timesteps-1, n_dim_control] tensor
            Controls for times [0...n_timesteps-2]
        transition_matrices : [batch_size, n_timesteps-1, n_dim_state, n_dim_state] tensor
            State transition matrices
        control_matrices : [batch_size, n_timesteps-1, n_dim_control, n_dim_state] tensor
            Control matrices
        observation_matrices : [batch_size, n_timesteps, n_dim_obs, n_dim_state] tensor
            Observation matrices
        transition_covariance : [batch_size, n_timesteps-1, n_dim_state, n_dim_state] tensor
            State transition covariance matrices
        observation_covariance : [batch_size, n_timesteps, n_dim_obs, n_dim_obs] tensor
            Observation covariance matrices
        transition_offsets : [batch_size, n_timesteps-1, n_dim_state] tensor
            State transition offsets
        observation_offsets : [batch_size, n_timesteps, n_dim_obs] tensor
            Observation offsets
        initial_state_mean : [batch_size, n_dim_state] tensor
            Initial state mean
        initial_state_covariance : [batch_size, n_dim_state, n_dim_state] tensor
            Initial state covariance

        Returns
        -------
        filtered_means : [batch_size, n_timesteps, n_dim_state] tensor
        filtered_covs : [batch_size, n_timesteps, n_dim_state, n_dim_state] tensor
        predicted_means : [batch_size, n_timesteps, n_dim_state] tensor
        predicted_covs : [batch_size, n_timesteps, n_dim_state, n_dim_state] tensor
        """
        batch_size, n_timesteps, n_dim_obs = observations.shape
        n_dim_state = transition_matrices.shape[-1]

        filtered_means = torch.zeros(
            batch_size, n_timesteps, n_dim_state, device=observations.device
        )
        filtered_covs = torch.zeros(
            batch_size,
            n_timesteps,
            n_dim_state,
            n_dim_state,
            device=observations.device,
        )
        predicted_means = torch.zeros_like(filtered_means)
        predicted_covs = torch.zeros_like(filtered_covs)

        # Initialize first timestep
        time_indices = torch.arange(n_timesteps, device=observations.device)
        predicted_means = torch.where(
            (time_indices == 0).view(1, -1, 1),
            initial_state_mean.unsqueeze(1),
            predicted_means,
        )
        predicted_covs = torch.where(
            (time_indices == 0).view(1, -1, 1, 1),
            initial_state_covariance.unsqueeze(1),
            predicted_covs,
        )

        for t in range(n_timesteps):
            # Correct
            kalman_gain, filtered_means_t, filtered_covs_t = self._filter_correct(
                observation_matrices[:, t],
                observation_covariance[:, t],
                observation_offsets[:, t],
                predicted_means[:, t],
                predicted_covs[:, t],
                observations[:, t],
            )

            # Update using where operations with correct dimensions
            filtered_means = torch.where(
                (time_indices == t).view(1, -1, 1),
                filtered_means_t.unsqueeze(1),
                filtered_means,
            )
            filtered_covs = torch.where(
                (time_indices == t).view(1, -1, 1, 1),
                filtered_covs_t.unsqueeze(1),
                filtered_covs,
            )

            # Predict next state
            if t < n_timesteps - 1:
                pred_mean, pred_cov = self._filter_predict(
                    transition_matrices[:, t],
                    transition_covariance[:, t],
                    transition_offsets[:, t],
                    filtered_means[:, t],
                    filtered_covs[:, t],
                    controls[:, t],
                    control_matrices[:, t],
                )
                predicted_means = torch.where(
                    (time_indices == t + 1).view(1, -1, 1),
                    pred_mean.unsqueeze(1),
                    predicted_means,
                )
                predicted_covs = torch.where(
                    (time_indices == t + 1).view(1, -1, 1, 1),
                    pred_cov.unsqueeze(1),
                    predicted_covs,
                )

        return filtered_means, filtered_covs, predicted_means, predicted_covs

    def _smooth(
        self,
        filtered_means,
        filtered_covs,
        predicted_means,
        predicted_covs,
        transition_matrices,
    ):
        """Apply Kalman Smoother to get state distributions given all observations."""
        batch_size, n_timesteps, n_dim_state = filtered_means.shape
        time_indices = torch.arange(n_timesteps, device=filtered_means.device)

        # Initialize with new tensors
        smoothed_means = torch.zeros_like(filtered_means)
        smoothed_covs = torch.zeros_like(filtered_covs)

        # Initialize last timestep
        smoothed_means = torch.where(
            (time_indices == n_timesteps - 1).view(1, -1, 1),
            filtered_means[:, -1].unsqueeze(1),
            smoothed_means,
        )
        smoothed_covs = torch.where(
            (time_indices == n_timesteps - 1).view(1, -1, 1, 1),
            filtered_covs[:, -1].unsqueeze(1),
            smoothed_covs,
        )

        for t in range(n_timesteps - 2, -1, -1):
            # Compute gain matrix
            gain = torch.matmul(
                torch.matmul(
                    filtered_covs[:, t], transition_matrices[:, t].transpose(-2, -1)
                ),
                torch.linalg.pinv(predicted_covs[:, t + 1]),
            )

            # Compute smoothed mean
            mean_diff = (
                smoothed_means[:, t + 1] - predicted_means[:, t + 1]
            ).unsqueeze(-1)
            smoothed_t = filtered_means[:, t] + torch.matmul(gain, mean_diff).squeeze(
                -1
            )
            smoothed_means = torch.where(
                (time_indices == t).view(1, -1, 1),
                smoothed_t.unsqueeze(1),
                smoothed_means,
            )

            # Compute smoothed covariance
            cov_diff = smoothed_covs[:, t + 1] - predicted_covs[:, t + 1]
            smoothed_cov_t = filtered_covs[:, t] + torch.matmul(
                torch.matmul(gain, cov_diff), gain.transpose(-2, -1)
            )
            smoothed_covs = torch.where(
                (time_indices == t).view(1, -1, 1, 1),
                smoothed_cov_t.unsqueeze(1),
                smoothed_covs,
            )

        return smoothed_means, smoothed_covs

    def filter(
        self,
        observations,
        controls,
        transition_matrices,
        control_matrices,
        observation_matrices,
        transition_covariance,
        observation_covariance,
        transition_offsets,
        observation_offsets,
        initial_state_mean,
        initial_state_covariance,
    ):
        return self._filter(
            observations,
            controls,
            transition_matrices,
            control_matrices,
            observation_matrices,
            transition_covariance,
            observation_covariance,
            transition_offsets,
            observation_offsets,
            initial_state_mean,
            initial_state_covariance,
        )

    def smooth(
        self,
        filtered_means,
        filtered_covs,
        predicted_means,
        predicted_covs,
        transition_matrices,
    ):
        return self._smooth(
            filtered_means,
            filtered_covs,
            predicted_means,
            predicted_covs,
            transition_matrices,
        )

    def forward(
        self,
        observations,
        controls,
        transition_matrices,
        control_matrices,
        observation_matrices,
        transition_covariance,
        observation_covariance,
        transition_offsets,
        observation_offsets,
        initial_state_mean,
        initial_state_covariance,
        mode="filter",
    ):
        if mode == "filter":
            if self.compile_mode:
                filtered_means, filtered_covs, predicted_means, predicted_covs = (
                    self.compiled_filter(
                        observations,
                        controls,
                        transition_matrices,
                        control_matrices,
                        observation_matrices,
                        transition_covariance,
                        observation_covariance,
                        transition_offsets,
                        observation_offsets,
                        initial_state_mean,
                        initial_state_covariance,
                    )
                )
            else:
                filtered_means, filtered_covs, predicted_means, predicted_covs = (
                    self.filter(
                        observations,
                        controls,
                        transition_matrices,
                        control_matrices,
                        observation_matrices,
                        transition_covariance,
                        observation_covariance,
                        transition_offsets,
                        observation_offsets,
                        initial_state_mean,
                        initial_state_covariance,
                    )
                )
            return filtered_means, filtered_covs
        elif mode == "smooth":
            if self.compile_mode:
                filtered_means, filtered_covs, predicted_means, predicted_covs = (
                    self.compiled_filter(
                        observations,
                        controls,
                        transition_matrices,
                        control_matrices,
                        observation_matrices,
                        transition_covariance,
                        observation_covariance,
                        transition_offsets,
                        observation_offsets,
                        initial_state_mean,
                        initial_state_covariance,
                    )
                )
                return self.compiled_smooth(
                    filtered_means,
                    filtered_covs,
                    predicted_means,
                    predicted_covs,
                    transition_matrices,
                )
            else:
                filtered_means, filtered_covs, predicted_means, predicted_covs = (
                    self.filter(
                        observations,
                        controls,
                        transition_matrices,
                        control_matrices,
                        observation_matrices,
                        transition_covariance,
                        observation_covariance,
                        transition_offsets,
                        observation_offsets,
                        initial_state_mean,
                        initial_state_covariance,
                    )
                )
                return self.smooth(
                    filtered_means,
                    filtered_covs,
                    predicted_means,
                    predicted_covs,
                    transition_matrices,
                )
        else:
            raise ValueError(f"Invalid mode: {mode}")
