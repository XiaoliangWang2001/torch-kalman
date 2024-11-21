import torch
import pytest
import sys
import os

# add parent folder to search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bierman import BiermanKalmanFilter, udu, UDUDecomposition, UDUTensor


class TestBiermanKalmanFilter:
    @pytest.fixture
    def sample_data(self):
        batch_size = 2
        n_timesteps = 3
        n_dim_state = 2
        n_dim_obs = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate sample data
        observations = torch.randn(batch_size, n_timesteps, n_dim_obs, device=device)
        transition_matrices = torch.eye(n_dim_state, device=device).expand(
            batch_size, n_timesteps - 1, n_dim_state, n_dim_state
        )
        observation_matrices = torch.eye(n_dim_obs, n_dim_state, device=device).expand(
            batch_size, n_timesteps, n_dim_obs, n_dim_state
        )
        transition_covariance = torch.eye(n_dim_state, device=device).expand(
            batch_size, n_timesteps - 1, n_dim_state, n_dim_state
        )
        observation_covariance = torch.eye(n_dim_obs, device=device).expand(
            batch_size, n_timesteps, n_dim_obs, n_dim_obs
        )
        transition_offsets = torch.zeros(
            batch_size, n_timesteps - 1, n_dim_state, device=device
        )
        observation_offsets = torch.zeros(
            batch_size, n_timesteps, n_dim_obs, device=device
        )
        initial_state_mean = torch.zeros(batch_size, n_dim_state, device=device)
        initial_state_covariance = torch.eye(n_dim_state, device=device).expand(
            batch_size, n_dim_state, n_dim_state
        )

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
        }

    def test_udu_tensor(self):
        """Test the UDUTensor container class"""
        batch_size = 2
        n_timesteps = 3
        n_dim_state = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create UDUTensor instance
        udu_tensor = UDUTensor(batch_size, n_timesteps, n_dim_state, device)

        # Test initialization
        assert udu_tensor.U.shape == (batch_size, n_timesteps, n_dim_state, n_dim_state)
        assert udu_tensor.D.shape == (batch_size, n_timesteps, n_dim_state)

        # Test getitem
        decomp = udu_tensor[0, 0]
        assert isinstance(decomp, UDUDecomposition)
        assert decomp.U.shape == (n_dim_state, n_dim_state)
        assert decomp.D.shape == (n_dim_state,)

        # Test setitem with UDUDecomposition
        new_decomp = UDUDecomposition(
            torch.eye(n_dim_state, device=device) * 2,
            torch.ones(n_dim_state, device=device) * 3
        )
        udu_tensor[0, 0] = new_decomp
        assert torch.allclose(udu_tensor.U[0, 0], new_decomp.U)
        assert torch.allclose(udu_tensor.D[0, 0], new_decomp.D)

        # Test reconstruct
        full_cov = udu_tensor.reconstruct()
        assert full_cov.shape == (batch_size, n_timesteps, n_dim_state, n_dim_state)
        
        # Test single index reconstruct
        single_cov = udu_tensor.reconstruct((0, 0))
        assert single_cov.shape == (n_dim_state, n_dim_state)

    def test_decorrelate_observations(self, sample_data):
        """Test the _decorrelate_observations method"""
        kf = BiermanKalmanFilter()
        
        (
            obs_matrices2,
            obs_offsets2,
            obs_cov2,
            observations2,
        ) = kf._decorrelate_observations(
            sample_data["observation_matrices"],
            sample_data["observation_offsets"],
            sample_data["observation_covariance"],
            sample_data["observations"],
        )

        # Check shapes remain the same
        assert obs_matrices2.shape == sample_data["observation_matrices"].shape
        assert obs_offsets2.shape == sample_data["observation_offsets"].shape
        assert obs_cov2.shape == sample_data["observation_covariance"].shape
        assert observations2.shape == sample_data["observations"].shape

        # Check observation covariance is now identity
        identity = torch.eye(
            obs_cov2.shape[-1],
            device=obs_cov2.device
        ).expand_as(obs_cov2)
        assert torch.allclose(obs_cov2, identity, atol=1e-6)

    def test_filter_correct_udu(self, sample_data):
        """Test the _filter_correct method with UDU decomposition"""
        kf = BiermanKalmanFilter()

        # Get initial state
        pred_mean = sample_data["initial_state_mean"]
        pred_cov = sample_data["initial_state_covariance"]
        pred_udu = udu(pred_cov)

        # Run correction step
        corrected_mean, corrected_udu = kf._filter_correct(
            sample_data["observation_matrices"][:, 0],
            sample_data["observation_covariance"][:, 0],
            sample_data["observation_offsets"][:, 0],
            pred_mean,
            pred_udu,
            sample_data["observations"][:, 0],
        )

        # Check outputs
        assert isinstance(corrected_udu, UDUDecomposition)
        assert corrected_mean.shape == pred_mean.shape
        assert corrected_udu.U.shape == pred_cov.shape
        assert corrected_udu.D.shape == pred_mean.shape

        # Reconstruct covariance and check properties
        corrected_cov = corrected_udu.reconstruct()
        assert torch.allclose(corrected_cov, corrected_cov.transpose(-2, -1))
        assert torch.all(torch.linalg.eigvalsh(corrected_cov) >= -1e-6)

    def test_filter_predict_udu(self, sample_data):
        kf = BiermanKalmanFilter()

        # Test single prediction step
        current_state_mean = sample_data["initial_state_mean"]
        current_state_covariance = sample_data["initial_state_covariance"]
        transition_matrix = sample_data["transition_matrices"][:, 0]
        transition_offset = sample_data["transition_offsets"][:, 0]
        transition_covariance = sample_data["transition_covariance"][:, 0]

        # Convert to UDU form
        current_udu = udu(current_state_covariance)

        # Predict using UDU
        pred_mean, pred_udu = kf._filter_predict(
            transition_matrix,
            transition_covariance,
            transition_offset,
            current_state_mean,
            current_udu,
        )

        # Check shapes
        assert pred_mean.shape == current_state_mean.shape
        assert isinstance(pred_udu, UDUDecomposition)
        assert pred_udu.U.shape == current_state_covariance.shape
        assert pred_udu.D.shape == current_state_mean.shape


    def test_full_filter_udu(self, sample_data):
        kf = BiermanKalmanFilter()

        # Run full filter
        filtered_means, filtered_covs, predicted_means, predicted_covs = kf._filter(
            **sample_data
        )

        # Check shapes
        assert filtered_means.shape == (
            2,
            3,
            2,
        )  # [batch_size, n_timesteps, n_dim_state]
        assert filtered_covs.shape == (2, 3, 2, 2)
        assert predicted_means.shape == filtered_means.shape
        assert predicted_covs.shape == filtered_covs.shape

        # Check properties for all timesteps
        for t in range(filtered_covs.shape[1]):
            # Symmetry
            assert torch.allclose(
                filtered_covs[:, t].reconstruct(), filtered_covs[:, t].reconstruct().transpose(-2, -1)
            )
            assert torch.allclose(
                predicted_covs[:, t].reconstruct(), predicted_covs[:, t].reconstruct().transpose(-2, -1)
            )

            # Positive semidefinite
            assert torch.all(torch.linalg.eigvalsh(filtered_covs[:, t].reconstruct()) >= -1e-6)
            assert torch.all(torch.linalg.eigvalsh(predicted_covs[:, t].reconstruct()) >= -1e-6)

    def test_smooth_udu(self, sample_data):
        kf = BiermanKalmanFilter()

        # Run filter first
        filtered_means, filtered_covs, predicted_means, predicted_covs = kf.filter(
            **sample_data
        )

        # Run smoother
        smoothed_means, smoothed_covs = kf.smooth(
            filtered_means,
            filtered_covs,
            predicted_means,
            predicted_covs,
            sample_data["transition_matrices"],
        )

        # Check shapes
        assert smoothed_means.shape == filtered_means.shape
        assert smoothed_covs.shape == filtered_covs.shape

        # Check properties for all timesteps
        for t in range(smoothed_covs.shape[1]):
            # Symmetry
            assert torch.allclose(
                smoothed_covs[:, t], smoothed_covs[:, t].transpose(-2, -1)
            )

            # Positive semidefinite
            assert torch.all(torch.linalg.eigvalsh(smoothed_covs[:, t]) >= -1e-6)

    def test_compare_with_standard_kf(self, sample_data):
        from standard import KalmanFilter

        # Initialize both filters
        standard_kf = KalmanFilter()
        bierman_kf = BiermanKalmanFilter()

        # Run both filters
        standard_means, standard_covs = standard_kf(**sample_data, mode="filter")
        bierman_means, bierman_covs = bierman_kf(**sample_data, mode="filter")
        
        # Print detailed comparison information
        print("\nDetailed comparison:")
        print(f"\nStandard means shape: {standard_means.shape}")
        print(f"Bierman means shape: {bierman_means.shape}")
        
        # Compare each timestep
        for t in range(standard_means.shape[1]):
            print(f"\nTimestep {t}:")
            print(f"Standard means:\n{standard_means[:, t]}")
            print(f"Bierman means:\n{bierman_means[:, t]}")
            
            # Calculate differences
            mean_diff = torch.abs(standard_means[:, t] - bierman_means[:, t])
            print(f"Mean absolute difference:\n{mean_diff}")
            
            if not torch.allclose(standard_means[:, t], bierman_means[:, t], rtol=1e-4, atol=1e-4):
                print(f"Large difference at timestep {t}!")

        # Original assertions with more context
        means_close = torch.allclose(standard_means, bierman_means, rtol=1e-4, atol=1e-4)
        covs_close = torch.allclose(standard_covs, bierman_covs, rtol=1e-4, atol=1e-4)
        
        assert means_close, "Means differ significantly between standard and Bierman KF"
        assert covs_close, "Covariances differ significantly between standard and Bierman KF"

    def test_numerical_edge_cases(self):
        # Test with very small and very large values
        batch_size = 2
        n_timesteps = 3
        n_dim_state = 2
        n_dim_obs = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create data with extreme values
        sample_data = {
            "observations": torch.randn(
                batch_size, n_timesteps, n_dim_obs, device=device
            )
            * 1e10,
            "transition_matrices": torch.eye(n_dim_state, device=device).expand(
                batch_size, n_timesteps - 1, n_dim_state, n_dim_state
            )
            * 1e-10,
            "observation_matrices": torch.eye(
                n_dim_obs, n_dim_state, device=device
            ).expand(batch_size, n_timesteps, n_dim_obs, n_dim_state),
            "transition_covariance": torch.eye(n_dim_state, device=device).expand(
                batch_size, n_timesteps - 1, n_dim_state, n_dim_state
            )
            * 1e10,
            "observation_covariance": torch.eye(n_dim_obs, device=device).expand(
                batch_size, n_timesteps, n_dim_obs, n_dim_obs
            )
            * 1e-10,
            "transition_offsets": torch.zeros(
                batch_size, n_timesteps - 1, n_dim_state, device=device
            ),
            "observation_offsets": torch.zeros(
                batch_size, n_timesteps, n_dim_obs, device=device
            ),
            "initial_state_mean": torch.zeros(batch_size, n_dim_state, device=device),
            "initial_state_covariance": torch.eye(n_dim_state, device=device).expand(
                batch_size, n_dim_state, n_dim_state
            ),
        }

        kf = BiermanKalmanFilter()

        # This should run without numerical issues
        filtered_means, filtered_covs = kf(**sample_data, mode="filter")

        # Check results are finite
        assert torch.all(torch.isfinite(filtered_means))
        assert torch.all(torch.isfinite(filtered_covs))

    def test_gradient_flow(self):
        # Create a simple matrix that requires grad
        M = torch.tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)

        # Create UDU decomposition
        decomp = udu(M)

        # Reconstruct and compute loss
        reconstructed = decomp()  # Using forward()
        loss = reconstructed.sum()

        # Check if gradients flow
        loss.backward()
        assert M.grad is not None
        assert torch.all(torch.isfinite(M.grad))

    def test_bierman_filter_gradients(self, sample_data):
        kf = BiermanKalmanFilter()

        # Make some parameters require gradients
        sample_data["transition_matrices"].requires_grad_(True)
        sample_data["observation_matrices"].requires_grad_(True)

        # Run filter
        filtered_means, filtered_covs = kf(**sample_data, mode="filter")

        # Compute loss
        loss = filtered_means.sum() + filtered_covs.sum()

        # Check gradient flow
        loss.backward()

        assert sample_data["transition_matrices"].grad is not None
        assert sample_data["observation_matrices"].grad is not None
        assert torch.all(torch.isfinite(sample_data["transition_matrices"].grad))
        assert torch.all(torch.isfinite(sample_data["observation_matrices"].grad))

    def test_udu_batch_gradients(self):
        # Create batch of matrices requiring gradients
        batch_size = 2
        M = torch.randn(batch_size, 2, 2, requires_grad=True)
        M = torch.matmul(M, M.transpose(-2, -1))  # Make symmetric positive definite

        # Create UDU decomposition
        decomp = udu(M)

        # Reconstruct and compute loss
        reconstructed = decomp()
        loss = reconstructed.sum()

        # Check gradient flow through batched operations
        loss.backward()
        assert M.grad is not None
        assert M.grad.shape == M.shape
        assert torch.all(torch.isfinite(M.grad))


if __name__ == "__main__":
    pytest.main([__file__])
    
