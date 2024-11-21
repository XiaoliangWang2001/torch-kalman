import torch
import pytest
import time
import sys
import os

# add parent folder to search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from standard import KalmanFilter
from specialized_standard import RectIdentityKalmanFilter
from bierman import BiermanKalmanFilter

@pytest.fixture
def test_dimensions():
    return {
        'batch_size': 16,
        'n_timesteps': 100,
        'n_dim_state': 5,
        'n_dim_obs': 3,
    }

@pytest.fixture
def test_data(test_dimensions):
    """Generate test data."""
    batch_size = test_dimensions['batch_size']
    n_timesteps = test_dimensions['n_timesteps']
    n_dim_state = test_dimensions['n_dim_state']
    n_dim_obs = test_dimensions['n_dim_obs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate random transition matrices
    transition_matrices = torch.randn(batch_size, n_timesteps-1, n_dim_state, n_dim_state, device=device)
    
    # Generate positive definite transition covariance
    transition_covariance = torch.randn(batch_size, n_timesteps-1, n_dim_state, n_dim_state, device=device)
    transition_covariance = torch.matmul(transition_covariance, transition_covariance.transpose(-2, -1))
    transition_covariance = (transition_covariance + transition_covariance.transpose(-2, -1)) / 2
    transition_covariance = transition_covariance + torch.eye(n_dim_state, device=device)
    
    # Generate transition offsets
    transition_offsets = torch.randn(batch_size, n_timesteps-1, n_dim_state, device=device)
    
    # Generate initial state
    initial_state_mean = torch.randn(batch_size, n_dim_state, device=device)
    initial_state_covariance = torch.randn(batch_size, n_dim_state, n_dim_state, device=device)
    initial_state_covariance = torch.matmul(initial_state_covariance, initial_state_covariance.transpose(-2, -1))
    initial_state_covariance = (initial_state_covariance + initial_state_covariance.transpose(-2, -1)) / 2
    initial_state_covariance = initial_state_covariance + torch.eye(n_dim_state, device=device)

    # Generate observations
    observations = torch.randn(batch_size, n_timesteps, n_dim_obs, device=device)

    # Create rectangular identity observation matrix
    observation_matrices = torch.zeros(batch_size, n_timesteps, n_dim_obs, n_dim_state, device=device)
    observation_matrices[..., :n_dim_obs, :n_dim_obs] = torch.eye(n_dim_obs, device=device)
    
    # Create identity observation covariance
    observation_covariance = torch.eye(n_dim_obs, device=device).expand(batch_size, n_timesteps, n_dim_obs, n_dim_obs)
    
    # Zero observation offset
    observation_offsets = torch.zeros(batch_size, n_timesteps, n_dim_obs, device=device)

    return {
        'observations': observations,
        'transition_matrices': transition_matrices,
        'observation_matrices': observation_matrices,
        'transition_covariance': transition_covariance,
        'observation_covariance': observation_covariance,
        'transition_offsets': transition_offsets,
        'observation_offsets': observation_offsets,
        'initial_state_mean': initial_state_mean,
        'initial_state_covariance': initial_state_covariance,
    }

@pytest.fixture
def scaled_test_data(test_data):
    """Helper fixture to create test data with specific scales."""
    def _create_test_data_with_scale(scale1, scale2):
        # Create a copy of the base test data
        data = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                for k, v in test_data.items()}
        
        # Scale different components
        data['observations'] = data['observations'] * scale1
        data['transition_matrices'] = data['transition_matrices'] * scale2
        data['initial_state_mean'] = data['initial_state_mean'] * scale1
        
        return data
    return _create_test_data_with_scale

def test_correctness(test_dimensions, test_data):
    """Test that specialized implementation gives same results as standard."""
    specialized = RectIdentityKalmanFilter(
        n_dim_state=test_dimensions['n_dim_state'],
        n_dim_obs=test_dimensions['n_dim_obs']
    )
    standard = KalmanFilter()
    
    # Run both implementations
    spec_means, spec_covs = specialized(**test_data, mode='filter')
    std_means, std_covs = standard(**test_data, mode='filter')
    
    # Print some debug info
    mean_diff = torch.max(torch.abs(spec_means - std_means)).item()
    cov_diff = torch.max(torch.abs(spec_covs - std_covs)).item()
    print(f"\nMeans max difference: {mean_diff}")
    print(f"Covs max difference: {cov_diff}")
    
    # Check first timestep differences
    print("\nFirst timestep means difference:")
    print(spec_means[:, 0] - std_means[:, 0])
    
    # Use looser tolerances since we're using different numerical approaches
    assert torch.allclose(spec_means, std_means, rtol=1e-2, atol=1e-2), \
        f"Means differ by {mean_diff}"
    assert torch.allclose(spec_covs, std_covs, rtol=1e-2, atol=1e-2), \
        f"Covariances differ by {cov_diff}"
    
    # Additional checks for correctness
    assert torch.all(torch.isfinite(spec_means)), "Non-finite values in specialized means"
    assert torch.all(torch.isfinite(spec_covs)), "Non-finite values in specialized covs"
    
    # Check covariance properties
    for t in range(spec_covs.shape[1]):
        cov_t = spec_covs[:, t]
        # Symmetry
        assert torch.allclose(cov_t, cov_t.transpose(-2, -1), rtol=1e-4), \
            f"Covariance not symmetric at timestep {t}"
        # Positive definiteness
        eigvals = torch.linalg.eigvalsh(cov_t)
        assert torch.all(eigvals > -1e-6), \
            f"Non-PD covariance at timestep {t}"

def test_performance_comparison(test_dimensions, test_data):
    """Compare performance between implementations."""
    specialized = RectIdentityKalmanFilter(
        n_dim_state=test_dimensions['n_dim_state'],
        n_dim_obs=test_dimensions['n_dim_obs']
    )
    standard = KalmanFilter()
    bierman = BiermanKalmanFilter()
    
    # Warmup
    for _ in range(5):
        specialized(**test_data, mode='filter')
        standard(**test_data, mode='filter')
        bierman(**test_data, mode='filter')
    
    # Time specialized version
    n_runs = 10
    start_time = time.perf_counter()
    for _ in range(n_runs):
        specialized(**test_data, mode='filter')
    specialized_time = (time.perf_counter() - start_time) / n_runs
    
    # Time standard version
    start_time = time.perf_counter()
    for _ in range(n_runs):
        standard(**test_data, mode='filter')
    standard_time = (time.perf_counter() - start_time) / n_runs
    
    # Time Bierman version
    start_time = time.perf_counter()
    for _ in range(n_runs):
        bierman(**test_data, mode='filter')
    bierman_time = (time.perf_counter() - start_time) / n_runs
    
    print(f"\nPerformance comparison (average over {n_runs} runs):")
    print(f"Specialized: {specialized_time*1000:.2f}ms")
    print(f"Standard: {standard_time*1000:.2f}ms")
    print(f"Bierman: {bierman_time*1000:.2f}ms")
    print(f"Speedup vs standard: {standard_time/specialized_time:.2f}x")
    print(f"Speedup vs Bierman: {bierman_time/specialized_time:.2f}x")

def test_numerical_stability(test_dimensions, scaled_test_data):
    """Test filter with challenging numerical cases."""
    specialized = RectIdentityKalmanFilter(
        n_dim_state=test_dimensions['n_dim_state'],
        n_dim_obs=test_dimensions['n_dim_obs']
    )
    
    # Test cases
    test_cases = [
        ("Large scale differences", 1e5, 1e-5),
        ("Very small values", 1e-10, 1e-10),
        ("Very large values", 1e10, 1e10),
    ]
    
    for name, scale1, scale2 in test_cases:
        print(f"\nTesting {name}")
        
        # Create test data with extreme scales
        data = scaled_test_data(scale1, scale2)
        
        try:
            means, covs = specialized(**data, mode='filter')
            
            # Check results are finite
            assert torch.all(torch.isfinite(means)), "Non-finite values in means"
            assert torch.all(torch.isfinite(covs)), "Non-finite values in covariances"
            
            # Check covariances are symmetric and PD
            for t in range(covs.shape[1]):
                cov_t = covs[:, t]
                assert torch.allclose(cov_t, cov_t.transpose(-2, -1), rtol=1e-4)
                eigvals = torch.linalg.eigvalsh(cov_t)
                assert torch.all(eigvals > -1e-6), "Non-PD covariance detected"
                
            print("✓ Passed")
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            raise


    

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 