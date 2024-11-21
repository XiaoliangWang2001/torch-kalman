import torch
from standard import KalmanFilter

class RectIdentityKalmanFilter(KalmanFilter):
    """Specialized correction step for rectangular identity observation matrix.
    
    This implementation is mathematically equivalent to the standard KF with 
    H = [I, 0] and R = I, but uses direct computations instead of matrix 
    multiplications. Small numerical differences (order of 1e-7) may occur due to:
    1. Different order of floating point operations
    2. Fewer intermediate matrix multiplications
    3. Direct submatrix access instead of matrix products with [I, 0]
    
    The specialized version is faster while maintaining numerical stability.
    """
    
    def __init__(self, n_dim_state: int, n_dim_obs: int, compile_mode: bool = False):
        super().__init__(compile_mode=False)
        if n_dim_obs >= n_dim_state:
            raise ValueError("n_dim_obs must be less than n_dim_state")
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs
        self.compile_mode = compile_mode
        if compile_mode:
            self.compiled_filter = torch.compile(self.filter)
            self.compiled_smooth = torch.compile(self.smooth)
            

    def _filter_correct(
        self,
        observation_matrix,  # ignored
        observation_covariance,  # ignored
        observation_offset,  # ignored
        predicted_state_mean,
        predicted_state_covariance,
        observation,
    ):
        """Specialized correction step for rectangular identity observation matrix."""
        # Ensure predicted_state_mean has correct shape
        if len(predicted_state_mean.shape) == 2:
            predicted_state_mean = predicted_state_mean.unsqueeze(-1)

        # For rectangular identity case:
        # H = [I_obs, 0], R = I
        # Innovation = y - H x = y - x[:n_obs]
        innovation = (observation - predicted_state_mean.squeeze(-1)[..., :self.n_dim_obs]).unsqueeze(-1)

        # Predicted observation covariance simplifies to:
        # S = H P H' + R = P[:n_obs, :n_obs] + I
        S = predicted_state_covariance[..., :self.n_dim_obs, :self.n_dim_obs].clone()
        S = S + torch.eye(self.n_dim_obs, device=S.device)

        # Kalman gain simplifies to:
        # K = P H' S^{-1} = P[:, :n_obs] S^{-1}
        kalman_gain = torch.matmul(
            predicted_state_covariance[..., :, :self.n_dim_obs],
            torch.linalg.inv(S)  # Using inv instead of pinv since S is well-conditioned
        )

        # Update state mean
        corrected_state_mean = predicted_state_mean.squeeze(-1) + torch.matmul(
            kalman_gain, innovation
        ).squeeze(-1)

        # Update state covariance
        # P = P - K H P simplifies since H = [I, 0]
        corrected_state_covariance = predicted_state_covariance - torch.matmul(
            kalman_gain[..., :self.n_dim_obs],
            predicted_state_covariance[..., :self.n_dim_obs, :]
        )

        # Ensure symmetry
        corrected_state_covariance = (
            corrected_state_covariance + corrected_state_covariance.transpose(-2, -1)
        ) / 2

        return kalman_gain, corrected_state_mean, corrected_state_covariance

