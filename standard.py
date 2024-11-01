import torch
from typing import Optional
from utils import tensor1d, tensor2d, determine_dimensionality


class KalmanFilter(torch.nn.Module):
    """
    A standard Kalman filter. The model is specified by:
    x_{t+1} = A_t @ x_t + b_t + q_t, q_t ~ N(0, Q_t)
    y_t = C_t @ x_t + d_t + r_t, r_t ~ N(0, R_t)
    """

    def __init__(
        self,
        transition_matrix: torch.Tensor,
        observation_matrix: torch.Tensor,
        transition_covariance: torch.Tensor,
        observation_covariance: torch.Tensor,
        transition_offset: Optional[torch.Tensor] = None,
        observation_offset: Optional[torch.Tensor] = None,
        initial_state_mean: Optional[torch.Tensor] = None,
        initial_state_covariance: Optional[torch.Tensor] = None,
        n_dim_state: Optional[int] = None,
        n_dim_obs: Optional[int] = None,
    ):
        super().__init__()
        n_dim_state = determine_dimensionality(
            [
                (transition_matrix, tensor2d, -2),
                (transition_offset, tensor1d, -1),
                (transition_covariance, tensor2d, -2),
                (initial_state_mean, tensor1d, -1),
                (initial_state_covariance, tensor2d, -2),
                (observation_matrix, tensor2d, -1),
                n_dim_state,
            ],
            None,
        )
        n_dim_obs = determine_dimensionality(
            [
                (observation_matrix, tensor2d, -2),
                (observation_offset, tensor1d, -1),
                (observation_covariance, tensor2d, -2),
                n_dim_obs,
            ],
            None,
        )
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offset = transition_offset
        self.observation_offset = observation_offset
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs

    def forward(self, y: torch.Tensor, mode: str):
        if mode == "filter":
            return self.filter(y)
        elif mode == "smooth":
            return self.smooth(y)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def filter(self, y: torch.Tensor):
        pass

    def smooth(self, y: torch.Tensor):
        pass

    def filter_update(self, x_pred: torch.Tensor, P_pred: torch.Tensor):
        pass
