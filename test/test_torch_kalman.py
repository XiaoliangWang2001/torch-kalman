import torch
import numpy as np
import pytest
import sys
import os

# add parent folder to search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from standard import KalmanFilter


class TestKalmanFilter:
    @pytest.fixture
    def setup_simple_system(self):
        """Create a simple linear system for testing."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Dimensions
        batch_size = 2
        n_timesteps = 5
        n_dim_state = 2
        n_dim_obs = 1

        # Create simple transition matrix (rotation matrix)
        angle = np.pi / 6  # 30 degrees
        transition_matrix = torch.tensor(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=torch.float32,
        )

        # Expand for batch and time dimensions
        transition_matrices = (
            transition_matrix.unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, n_timesteps - 1, 1, 1)
        )

        # Create simple observation matrix (observe first dimension only)
        observation_matrix = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        observation_matrices = (
            observation_matrix.unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, n_timesteps, 1, 1)
        )

        # Create covariance matrices
        transition_covariance = (
            torch.eye(n_dim_state)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, n_timesteps - 1, 1, 1)
            * 0.1
        )
        observation_covariance = (
            torch.eye(n_dim_obs)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, n_timesteps, 1, 1)
            * 0.1
        )

        # Create offset vectors
        transition_offsets = torch.zeros(batch_size, n_timesteps - 1, n_dim_state)
        observation_offsets = torch.zeros(batch_size, n_timesteps, n_dim_obs)

        # Initial state
        initial_state_mean = torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(
            batch_size, 1
        )
        initial_state_covariance = (
            torch.eye(n_dim_state).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        # Generate synthetic observations
        true_states = torch.zeros(batch_size, n_timesteps, n_dim_state)
        observations = torch.zeros(batch_size, n_timesteps, n_dim_obs)

        # Initialize first state
        true_states[:, 0] = initial_state_mean

        # Generate subsequent states and observations
        for t in range(n_timesteps):
            # Generate observation
            observations[:, t] = torch.matmul(
                observation_matrices[:, t], true_states[:, t].unsqueeze(-1)
            ).squeeze(-1) + torch.normal(0, 0.1, size=(batch_size, n_dim_obs))

            # Generate next state
            if t < n_timesteps - 1:
                true_states[:, t + 1] = torch.matmul(
                    transition_matrices[:, t], true_states[:, t].unsqueeze(-1)
                ).squeeze(-1) + torch.normal(0, 0.1, size=(batch_size, n_dim_state))

        return {
            "observations": observations,
            "transition_matrices": transition_matrices,
            "observation_matrices": observation_matrices,
            "transition_covariance": transition_covariance,
            "observation_covariance": observation_covariance,
            "transition_offsets": transition_offsets,
            "observation_offsets": observation_offsets,
            "initial_state_mean": initial_state_mean,
            "initial_state_covariance": initial_state_covariance,
            "true_states": true_states,
        }

    def test_filter(self, setup_simple_system):
        """Test the Kalman filter implementation."""
        kf = KalmanFilter()
        data = setup_simple_system

        # Run filter
        filtered_means, filtered_covs, predicted_means, predicted_covs = kf._filter(
            observations=data["observations"],
            transition_matrices=data["transition_matrices"],
            observation_matrices=data["observation_matrices"],
            transition_covariance=data["transition_covariance"],
            observation_covariance=data["observation_covariance"],
            transition_offsets=data["transition_offsets"],
            observation_offsets=data["observation_offsets"],
            initial_state_mean=data["initial_state_mean"],
            initial_state_covariance=data["initial_state_covariance"],
        )

        # Basic shape checks
        assert filtered_means.shape == data["true_states"].shape
        assert filtered_covs.shape == (
            *data["true_states"].shape,
            data["true_states"].shape[-1],
        )

        # Check that covariances are symmetric and positive definite
        for t in range(filtered_covs.shape[1]):
            symm_diff = torch.norm(
                filtered_covs[:, t] - filtered_covs[:, t].transpose(-2, -1)
            )
            assert symm_diff < 1e-5, f"Covariance at step {t} is not symmetric"

            # Check positive definiteness using Cholesky decomposition
            try:
                torch.linalg.cholesky(filtered_covs[:, 0])
            except RuntimeError:
                pytest.fail(f"Covariance at step {t} is not positive definite")

    def test_smooth(self, setup_simple_system):
        """Test the Kalman smoother implementation."""
        kf = KalmanFilter()
        data = setup_simple_system

        # First run filter
        filtered_means, filtered_covs, predicted_means, predicted_covs = kf._filter(
            observations=data["observations"],
            transition_matrices=data["transition_matrices"],
            observation_matrices=data["observation_matrices"],
            transition_covariance=data["transition_covariance"],
            observation_covariance=data["observation_covariance"],
            transition_offsets=data["transition_offsets"],
            observation_offsets=data["observation_offsets"],
            initial_state_mean=data["initial_state_mean"],
            initial_state_covariance=data["initial_state_covariance"],
        )

        # Run smoother
        smoothed_means, smoothed_covs = kf._smooth(
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            predicted_means=predicted_means,
            predicted_covs=predicted_covs,
            transition_matrices=data["transition_matrices"],
        )

        # Basic shape checks
        assert smoothed_means.shape == data["true_states"].shape
        assert smoothed_covs.shape == (
            *data["true_states"].shape,
            data["true_states"].shape[-1],
        )

        # Check that smoothed estimates have lower uncertainty than filtered estimates
        # (at least for some timesteps)
        cov_norms_filtered = torch.norm(
            filtered_covs.reshape(filtered_covs.shape[0], -1), dim=1
        )
        cov_norms_smoothed = torch.norm(
            smoothed_covs.reshape(smoothed_covs.shape[0], -1), dim=1
        )
        assert torch.any(
            cov_norms_smoothed < cov_norms_filtered
        ), "Smoothing should reduce uncertainty for at least some timesteps"

        # Check that covariances are symmetric and positive definite
        for t in range(smoothed_covs.shape[1]):
            symm_diff = torch.norm(
                smoothed_covs[:, t] - smoothed_covs[:, t].transpose(-2, -1)
            )
            assert symm_diff < 1e-5, f"Smoothed covariance at step {t} is not symmetric"

            try:
                torch.linalg.cholesky(smoothed_covs[:, 0])
            except RuntimeError:
                pytest.fail(f"Smoothed covariance at step {t} is not positive definite")

    def test_gradient_flow_filter(self, setup_simple_system):
        """Test that gradients can flow through the filter."""
        kf = KalmanFilter()
        data = setup_simple_system

        # Make parameters require gradients
        transition_matrices = data["transition_matrices"].clone().requires_grad_(True)
        observation_matrices = data["observation_matrices"].clone().requires_grad_(True)
        with torch.autograd.set_detect_anomaly(True):
            # Run filter
            filtered_means, filtered_covs = kf.forward(
                observations=data["observations"],
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                transition_covariance=data["transition_covariance"],
                observation_covariance=data["observation_covariance"],
                transition_offsets=data["transition_offsets"],
                observation_offsets=data["observation_offsets"],
                initial_state_mean=data["initial_state_mean"],
                initial_state_covariance=data["initial_state_covariance"],
                mode="filter"
            )

            # Compute loss and backpropagate
            loss = torch.mean(filtered_means**2)
            loss.backward()

        # Check that gradients exist
        assert (
            transition_matrices.grad is not None
        ), "No gradients for transition matrices in filter"
        assert (
            observation_matrices.grad is not None
        ), "No gradients for observation matrices in filter"

    def test_gradient_flow_smoother(self, setup_simple_system):
        """Test that gradients can flow through the smoother."""
        kf = KalmanFilter()
        data = setup_simple_system

        # Make parameters require gradients
        transition_matrices = data["transition_matrices"].clone().requires_grad_(True)
        observation_matrices = data["observation_matrices"].clone().requires_grad_(True)
        with torch.autograd.set_detect_anomaly(True):
            # Run smoother
            smoothed_means, smoothed_covs = kf.forward(
                observations=data["observations"],
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                transition_covariance=data["transition_covariance"],
                observation_covariance=data["observation_covariance"],
                transition_offsets=data["transition_offsets"],
                observation_offsets=data["observation_offsets"],
                initial_state_mean=data["initial_state_mean"],
                initial_state_covariance=data["initial_state_covariance"],
                mode="smooth"
            )

            # Compute loss and backpropagate
            loss = torch.mean(smoothed_means**2)
            loss.backward()

        # Check that gradients exist
        assert (
            transition_matrices.grad is not None
        ), "No gradients for transition matrices in smoother"
        assert (
            observation_matrices.grad is not None
        ), "No gradients for observation matrices in smoother"

    def test_compiled_filter(self, setup_simple_system):
        """Test the compiled Kalman filter implementation."""
        kf = KalmanFilter(compile_mode=True)
        data = setup_simple_system

        # Run filter
        filtered_means, filtered_covs = kf.forward(
            observations=data["observations"],
            transition_matrices=data["transition_matrices"],
            observation_matrices=data["observation_matrices"],
            transition_covariance=data["transition_covariance"],
            observation_covariance=data["observation_covariance"],
            transition_offsets=data["transition_offsets"],
            observation_offsets=data["observation_offsets"],
            initial_state_mean=data["initial_state_mean"],
            initial_state_covariance=data["initial_state_covariance"],
            mode="filter"
        )

        # Compare with non-compiled version
        kf_normal = KalmanFilter(compile_mode=False)
        filtered_means_normal, filtered_covs_normal = kf_normal.forward(
            observations=data["observations"],
            transition_matrices=data["transition_matrices"],
            observation_matrices=data["observation_matrices"],
            transition_covariance=data["transition_covariance"],
            observation_covariance=data["observation_covariance"],
            transition_offsets=data["transition_offsets"],
            observation_offsets=data["observation_offsets"],
            initial_state_mean=data["initial_state_mean"],
            initial_state_covariance=data["initial_state_covariance"],
            mode="filter"
        )

        # Check that results are close
        assert torch.allclose(filtered_means, filtered_means_normal, rtol=1e-4)
        assert torch.allclose(filtered_covs, filtered_covs_normal, rtol=1e-4)

    def test_compiled_smoother(self, setup_simple_system):
        """Test the compiled Kalman smoother implementation."""
        kf = KalmanFilter(compile_mode=True)
        data = setup_simple_system

        # Run smoother
        smoothed_means, smoothed_covs = kf.forward(
            observations=data["observations"],
            transition_matrices=data["transition_matrices"],
            observation_matrices=data["observation_matrices"],
            transition_covariance=data["transition_covariance"],
            observation_covariance=data["observation_covariance"],
            transition_offsets=data["transition_offsets"],
            observation_offsets=data["observation_offsets"],
            initial_state_mean=data["initial_state_mean"],
            initial_state_covariance=data["initial_state_covariance"],
            mode="smooth"
        )

        # Compare with non-compiled version
        kf_normal = KalmanFilter(compile_mode=False)
        smoothed_means_normal, smoothed_covs_normal = kf_normal.forward(
            observations=data["observations"],
            transition_matrices=data["transition_matrices"],
            observation_matrices=data["observation_matrices"],
            transition_covariance=data["transition_covariance"],
            observation_covariance=data["observation_covariance"],
            transition_offsets=data["transition_offsets"],
            observation_offsets=data["observation_offsets"],
            initial_state_mean=data["initial_state_mean"],
            initial_state_covariance=data["initial_state_covariance"],
            mode="smooth"
        )

        # Check that results are close
        assert torch.allclose(smoothed_means, smoothed_means_normal, rtol=1e-4)
        assert torch.allclose(smoothed_covs, smoothed_covs_normal, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
