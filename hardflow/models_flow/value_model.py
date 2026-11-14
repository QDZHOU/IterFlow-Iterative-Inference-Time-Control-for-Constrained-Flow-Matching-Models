"""Original HardFlow Maze2D task and constraint value model."""
import numpy as np
import torch
from torch import nn

class ProxyValueModel(nn.Module):

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        state_dim: int,
        objective: str = "",
        constraint: str = "",
        ellipse_margin: float = 0.0,
        value_objective_scale: float = 1.0,
        value_constraint_scale: float = 1.0,
        normalizer=None,
        dynamics_model=None,
        maze_square_center=(1.5, 4.8),
        maze_circle_center=(4.5, 5.3),
    ):

        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.transition_dim = action_dim + state_dim

        self.objective = objective
        self.constraint = constraint
        self.ellipse_margin = ellipse_margin
        self.maze_square_center = np.asarray(maze_square_center, dtype=float)
        self.maze_circle_center = np.asarray(maze_circle_center, dtype=float)
        if self.maze_circle_center.shape != (2,) or not np.isfinite(self.maze_circle_center).all():
            raise ValueError("maze_circle_center must be finite physical [x, y]")
        if self.maze_square_center.shape != (2,) or not np.isfinite(self.maze_square_center).all():
            raise ValueError("maze_square_center must be finite physical [x, y]")
        self.value_objective_scale = value_objective_scale
        self.value_constraint_scale = value_constraint_scale
        self.normalizer = normalizer
        self.dynamics_model = dynamics_model

        if normalizer is not None:
            self.obs_norm_mins = normalizer.normalizers["observations"].mins
            self.obs_norm_maxs = normalizer.normalizers["observations"].maxs

    def forward(
        self,
        x,
    ):
        """
        x: (batch_size, planning_horizon, transition_dim), should be normalized
        value: (batch_size, 1)
        """

        if self.objective == "":
            objective_value = torch.zeros(
                x.shape[0], 1, device=x.device, requires_grad=True
            )
        elif self.objective == "distance":
            objective_value = self._compute_distance_objective(x)
        else:
            raise NotImplementedError(f"Objective {self.objective} is not implemented.")

        if self.constraint == "":
            constraint_penalty = torch.zeros(
                x.shape[0], 1, device=x.device, requires_grad=True
            )
        elif self.constraint == "ellipses":
            constraint_values = self._compute_ellipse_constraints(x)
            constraint_violations = torch.clamp(-constraint_values, min=0.0)
            constraint_penalty = torch.sum(
                constraint_violations**2, dim=1, keepdim=True
            )
        elif self.constraint == "ellipses_and_dynamics":
            constraint_values = self._compute_ellipse_constraints(x)
            constraint_violations = torch.clamp(-constraint_values, min=0.0)
            constraint_penalty = torch.sum(
                constraint_violations**2, dim=1, keepdim=True
            )

            dynamics_constraint_values = self._compute_dynamics_constraints(x)
            dynamics_penalty = torch.sum(
                dynamics_constraint_values**2, dim=1, keepdim=True
            )
            constraint_penalty += dynamics_penalty
        else:
            raise NotImplementedError(
                f"Constraint {self.constraint} is not implemented."
            )

        value = (
            objective_value * self.value_objective_scale
            - constraint_penalty * self.value_constraint_scale
        )

        return value

    def _compute_distance_objective(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        objective_value: (batch_size, 1)
        """
        batch_size = x.shape[0]
        observations = x[
            :, :, self.action_dim :
        ]  # (batch_size, planning_horizon, state_dim)

        distance_objective = torch.zeros(
            batch_size, 1, device=x.device, requires_grad=True
        )

        # compute distance between consecutive positions
        for i in range(self.horizon - 1):
            current_pos = observations[:, i, :2]  # (batch_size, 2)
            next_pos = observations[:, i + 1, :2]  # (batch_size, 2)
            step_distance = torch.sum(
                (current_pos - next_pos) ** 2, dim=1, keepdim=True
            )  # (batch_size, 1)
            distance_objective = distance_objective + step_distance

        distance_objective = -1.0 * distance_objective

        return distance_objective

    def _compute_ellipse_constraints(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        constraint_values: (batch_size, num_constraints), >= 0 means feasible (outside ellipse)
        """
        batch_size = x.shape[0]
        observations = x[:, :, self.action_dim :]

        if self.normalizer is not None:
            obs_norm_min_0 = float(self.obs_norm_mins[0])
            obs_norm_max_0 = float(self.obs_norm_maxs[0])
            obs_norm_min_1 = float(self.obs_norm_mins[1])
            obs_norm_max_1 = float(self.obs_norm_maxs[1])
        else:
            raise ValueError("Normalizer is required for ellipse constraints.")

        xa = 2.0 * 0.8 / (obs_norm_max_1 - obs_norm_min_1)
        ya = 2.0 * 0.8 / (obs_norm_max_0 - obs_norm_min_0)
        circle_x, circle_y = self.maze_circle_center
        off_xa = 2.0 * (circle_y - obs_norm_min_1) / (obs_norm_max_1 - obs_norm_min_1) - 1.0
        off_ya = 2.0 * (circle_x - obs_norm_min_0) / (obs_norm_max_0 - obs_norm_min_0) - 1.0

        xb = 2.0 * 0.8 / (obs_norm_max_1 - obs_norm_min_1)
        yb = 2.0 * 0.8 / (obs_norm_max_0 - obs_norm_min_0)
        square_x, square_y = self.maze_square_center
        off_xb = 2.0 * (square_y - obs_norm_min_1) / (obs_norm_max_1 - obs_norm_min_1) - 1.0
        off_yb = 2.0 * (square_x - obs_norm_min_0) / (obs_norm_max_0 - obs_norm_min_0) - 1.0

        constraint_values = []

        for i in range(1, self.horizon - 1):

            x_pos = observations[:, i, 1]  # (batch_size,)
            y_pos = observations[:, i, 0]  # (batch_size,)

            ellipse1_constraint = (
                (
                    (1.0 / xa * (x_pos - off_xa)) ** 2
                    + (1.0 / ya * (y_pos - off_ya)) ** 2
                )
                - 1.0
                - self.ellipse_margin
            )  # (batch_size,)

            ellipse2_constraint = (
                (
                    (1.0 / xb * (x_pos - off_xb)) ** 4
                    + (1.0 / yb * (y_pos - off_yb)) ** 4
                )
                - 1.0
                - self.ellipse_margin
            )  # (batch_size,)

            constraint_values.append(
                ellipse1_constraint.unsqueeze(1)
            )  # (batch_size, 1)
            constraint_values.append(
                ellipse2_constraint.unsqueeze(1)
            )  # (batch_size, 1)

        if len(constraint_values) == 0:
            return torch.zeros(batch_size, 0, device=x.device, requires_grad=True)

        constraint_values = torch.cat(constraint_values, dim=1)
        return constraint_values

    def _compute_dynamics_constraints(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        constraint_values: (batch_size, num_constraints), = 0 means feasible
        """
        if self.dynamics_model is None:
            return torch.zeros(x.shape[0], 0, device=x.device, requires_grad=True)

        batch_size = x.shape[0]
        actions = x[:, :, : self.action_dim]
        observations = x[:, :, self.action_dim :]

        A_torch = torch.tensor(self.dynamics_model["A"], device=x.device, dtype=x.dtype)
        B_torch = torch.tensor(self.dynamics_model["B"], device=x.device, dtype=x.dtype)
        c_torch = torch.tensor(self.dynamics_model["c"], device=x.device, dtype=x.dtype)

        constraint_values = []

        for i in range(self.horizon - 3):
            current_state = observations[:, i, :]
            next_state = observations[:, i + 1, :]
            current_action = actions[:, i, :]

            predicted_next_state = (
                torch.matmul(current_state, A_torch.T)
                + torch.matmul(current_action, B_torch.T)
                + c_torch.unsqueeze(0)
            )

            dynamics_constraint = predicted_next_state - next_state
            constraint_values.append(dynamics_constraint)

        if len(constraint_values) == 0:
            return torch.zeros(batch_size, 0, device=x.device, requires_grad=True)

        constraint_values = torch.cat(constraint_values, dim=1)
        return constraint_values
