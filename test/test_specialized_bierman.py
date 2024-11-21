import torch
import pytest
import time
import sys
import os

# add parent folder to search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bierman import BiermanKalmanFilter
from specialized_bierman import RectIdentityBiermanFilter

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
    """Generate test data that satisfies our assumptions."""
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
    # Ensure perfect symmetry and positive definiteness
    transition_covariance = (transition_covariance + transition_covariance.transpose(-2, -1)) / 2
    transition_covariance = transition_covariance + torch.eye(n_dim_state, device=device) * 1.0  # Increased from 0.1
    
    # Generate transition offsets
    transition_offsets = torch.randn(batch_size, n_timesteps-1, n_dim_state, device=device)
    
    # Generate initial state
    initial_state_mean = torch.randn(batch_size, n_dim_state, device=device)
    initial_state_covariance = torch.randn(batch_size, n_dim_state, n_dim_state, device=device)
    initial_state_covariance = torch.matmul(initial_state_covariance, initial_state_covariance.transpose(-2, -1))
    # Ensure perfect symmetry and positive definiteness
    initial_state_covariance = (initial_state_covariance + initial_state_covariance.transpose(-2, -1)) / 2
    initial_state_covariance = initial_state_covariance + torch.eye(n_dim_state, device=device) * 1.0  # Increased from 0.1

    # Scale down the random components to improve numerical stability
    transition_covariance = transition_covariance * 0.1
    initial_state_covariance = initial_state_covariance * 0.1

    # Generate observations (only first n_dim_obs components)
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

def test_correctness_filter(test_dimensions, test_data):
    """Test that specialized filter gives same results as standard filter."""
    specialized = RectIdentityBiermanFilter(
        n_dim_state=test_dimensions['n_dim_state'],
        n_dim_obs=test_dimensions['n_dim_obs']
    )
    standard = BiermanKalmanFilter()

    # Run both filters
    specialized_results = specialized(**test_data, mode='filter')
    standard_results = standard(**test_data, mode='filter')

    # Check that results match
    for spec, std in zip(specialized_results, standard_results):
        assert torch.allclose(spec, std, rtol=1e-4, atol=1e-4)

def test_correctness_smooth(test_dimensions, test_data):
    """Test that specialized smoother gives same results as standard smoother."""
    specialized = RectIdentityBiermanFilter(
        n_dim_state=test_dimensions['n_dim_state'],
        n_dim_obs=test_dimensions['n_dim_obs']
    )
    standard = BiermanKalmanFilter()

    # Run both smoothers
    specialized_results = specialized(**test_data, mode='smooth')
    standard_results = standard(**test_data, mode='smooth')

    # Check that results match
    for spec, std in zip(specialized_results, standard_results):
        assert torch.allclose(spec, std, rtol=1e-4, atol=1e-4)

# @pytest.mark.parametrize('mode', ['filter', 'smooth'])
# def test_performance_comparison(test_dimensions, test_data, mode, capsys):
#     """Compare performance between specialized and standard Bierman implementations."""
#     import time
    
#     # Initialize filters
#     specialized = RectIdentityBiermanFilter(
#         n_dim_state=test_dimensions['n_dim_state'],
#         n_dim_obs=test_dimensions['n_dim_obs']
#     )
#     standard = BiermanKalmanFilter()

#     # Add observation matrices for standard filter
#     standard_data = test_data.copy()
#     device = test_data['observations'].device
#     batch_size = test_dimensions['batch_size']
#     n_timesteps = test_dimensions['n_timesteps']
#     n_dim_obs = test_dimensions['n_dim_obs']
#     n_dim_state = test_dimensions['n_dim_state']
    
#     # Create rectangular identity observation matrix
#     observation_matrices = torch.zeros(batch_size, n_timesteps, n_dim_obs, n_dim_state, device=device)
#     observation_matrices[..., :n_dim_obs, :n_dim_obs] = torch.eye(n_dim_obs, device=device)
    
#     # Create identity observation covariance
#     observation_covariance = torch.eye(n_dim_obs, device=device).expand(batch_size, n_timesteps, n_dim_obs, n_dim_obs)
    
#     # Zero observation offset
#     observation_offsets = torch.zeros(batch_size, n_timesteps, n_dim_obs, device=device)
    
#     standard_data.update({
#         'observation_matrices': observation_matrices,
#         'observation_covariance': observation_covariance,
#         'observation_offsets': observation_offsets,
#     })
    
#     # Warmup
#     for _ in range(5):
#         specialized(**test_data, mode=mode)
#         standard(**standard_data, mode=mode)
    
#     # Time specialized version
#     n_runs = 10
#     start_time = time.perf_counter()
#     for _ in range(n_runs):
#         specialized(**test_data, mode=mode)
#     specialized_time = (time.perf_counter() - start_time) / n_runs
    
#     # Time standard version
#     start_time = time.perf_counter()
#     for _ in range(n_runs):
#         standard(**standard_data, mode=mode)
#     standard_time = (time.perf_counter() - start_time) / n_runs
    
#     print(f"\nPerformance comparison (average over {n_runs} runs):")
#     print(f"Specialized: {specialized_time*1000:.2f}ms")
#     print(f"Standard: {standard_time*1000:.2f}ms")
#     print(f"Speedup: {standard_time/specialized_time:.2f}x")
    
#     # We should see some speedup
#     assert standard_time > specialized_time, "Specialized version should be faster"

@pytest.mark.parametrize('compile_mode', [True, False])
def test_compilation(test_dimensions, test_data, compile_mode):
    """Test that compilation works and improves performance."""
    specialized = RectIdentityBiermanFilter(
        n_dim_state=test_dimensions['n_dim_state'],
        n_dim_obs=test_dimensions['n_dim_obs'],
        compile_mode=compile_mode
    )
    
    # Warmup
    for _ in range(3):
        specialized(**test_data, mode='filter')
    
    # Time execution
    start_time = time.perf_counter()
    for _ in range(10):
        specialized(**test_data, mode='filter')
    avg_time = (time.perf_counter() - start_time) / 10
    
    print(f"\nCompilation {'enabled' if compile_mode else 'disabled'}:")
    print(f"Average time: {avg_time:.4f}s")

def test_invalid_dimensions():
    """Test that invalid dimensions raise appropriate errors."""
    with pytest.raises(ValueError):
        RectIdentityBiermanFilter(n_dim_state=5, n_dim_obs=6)  # n_dim_obs >= n_dim_state 
        

def test_detailed_performance(test_dimensions, test_data, capsys):
    """Profile where time is spent in both implementations."""
    import cProfile
    import pstats
    from pstats import SortKey
    
    # Setup same as before...
    specialized = RectIdentityBiermanFilter(
        n_dim_state=test_dimensions['n_dim_state'],
        n_dim_obs=test_dimensions['n_dim_obs']
    )
    standard = BiermanKalmanFilter()
    
    # Profile specialized version
    pr = cProfile.Profile()
    pr.enable()
    specialized(**test_data, mode='filter')
    pr.disable()
    
    with capsys.disabled():
        print("\nSpecialized version profile:")
        stats = pstats.Stats(pr).sort_stats(SortKey.TIME)
        stats.print_stats(10)  # Print top 10 time-consuming operations
    
    # Profile standard version
    pr = cProfile.Profile()
    pr.enable()
    standard(**test_data, mode='filter')
    pr.disable()
    
    with capsys.disabled():
        print("\nStandard version profile:")
        stats = pstats.Stats(pr).sort_stats(SortKey.TIME)
        stats.print_stats(10)
        
if __name__ == "__main__":
    pytest.main([__file__])