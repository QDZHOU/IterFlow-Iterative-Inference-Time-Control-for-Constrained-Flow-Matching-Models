from collections import namedtuple
import time
import torch
from torch import nn
from hardflow.config.flow_matching import FlowMatchingEvaluationConfig
from hardflow import utils
from hardflow.models_flow.flow_matcher import (
    apply_conditioning,
    apply_conditioning_from_conditioned_x,
)
from hardflow.utils.arrays import to_torch

import numpy as np

try:
    import casadi as cs
except ImportError:
    cs = None

Trajectories = namedtuple("Trajectories", "actions observations values")


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class ConditionedODESolver(nn.Module):
    def __init__(
        self,
        model,
        conditions,
        action_dim,
        value=None,
        ode_method="euler",
    ):
        super().__init__()
        self.model = model
        self.conditions = conditions
        self.value = value
        self.action_dim = action_dim
        assert ode_method in ["euler"], "Only Euler is supported for now"


    def forward(self, x, t_span, return_chain=False, *args, **kwargs):
        # x: (batch_size, planning_horizon, transition_dim)
        # t_span: at least 2 elements, should be evenly spaced
        assert len(t_span) > 1, "t_span must have at least 2 elements"

        x0 = x.clone()
        dt = t_span[1] - t_span[0]
        x_chain = [x0]
        for t in t_span:


            dx_dt = self.model(x, t)
            dx_dt = apply_conditioning_from_conditioned_x(
                dx_dt, torch.zeros_like(x), self.conditions, self.action_dim
            )
            x = x + dx_dt * dt
            x = x.detach()
            x_chain.append(x)

        if return_chain:
            return x, torch.stack(x_chain, dim=1).detach()
        else:
            return x


class FlowPolicy(nn.Module):
    def __init__(
        self,
        flow_model,
        value_model,
        normalizer,
        action_dim,
        state_dim,
        horizon,
        cfg: FlowMatchingEvaluationConfig,
        constrained_value_fn=None,
        constrained_flow_fn=None,
        dynamics_model=None,
    ):
        super().__init__()
        self.flow_model = flow_model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon

        self.cfg = cfg
        self.value_model = value_model

        self.constrained_value_fn = constrained_value_fn
        self.constrained_flow_fn = constrained_flow_fn
        self.dynamics_model = dynamics_model

        self.obs_norm_mins = self.normalizer.normalizers["observations"].mins
        self.obs_norm_maxs = self.normalizer.normalizers["observations"].maxs

    def sample_initial_noise(self, batch_size):
        shape = (batch_size, self.horizon, self.action_dim + self.state_dim)
        if self.cfg.initial_noise_path is None:
            noise = torch.randn(*shape, device=self.cfg.device)
        else:
            array = np.load(self.cfg.initial_noise_path, allow_pickle=False)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Initial noise must be finite with shape {shape}, got {array.shape}")
            noise = torch.as_tensor(array, dtype=torch.float32, device=self.cfg.device).clone()
        self.last_initial_noise = to_np(noise).copy()
        return noise

    def __call__(self, conditions, batch_size=1, verbose=True):

        if self.cfg.guidance_method == "hardflow_new":
            return self.hardflow_new_forward(conditions, batch_size)
        if self.cfg.guidance_method != "original":
            raise ValueError(f"Unsupported guidance method: {self.cfg.guidance_method}")

        assert batch_size == 1, "batch_size must be 1 for no guidance"

        # normalize the observation
        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        t_start = time.time()

        solver = ConditionedODESolver(
            self.flow_model,
            conditions,
            value=None,
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )

        x = self.sample_initial_noise(batch_size)
        x = apply_conditioning(
            x, to_torch(conditions, device=x.device), self.action_dim
        )

        x, x_chain = solver(
            x,
            t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps + 1, device=x.device
            )[:-1],
            return_chain=True,
        )

        normed_actions = x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(to_np(normed_actions), "actions")

        normed_observations = x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(
            to_np(normed_observations), "observations"
        )

        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)

        trajectories = Trajectories(actions, observations, values)

        action = actions[0, 0]

        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return action, trajectories, x_chain, x1_estimation, info

    def x1_estimate(self, x_chain, conditions):

        batch_size, n_steps, horizon, transition_dim = x_chain.shape

        x1_estimation = torch.zeros_like(x_chain)

        for k in range(n_steps):
            t = to_torch(k / (n_steps - 1), device=x_chain.device)
            current_x = x_chain[:, k, :, :]

            current_v = self.flow_model(current_x, t)
            current_v = apply_conditioning_from_conditioned_x(
                current_v, torch.zeros_like(current_v), conditions, self.action_dim
            )

            predicted_x = current_x + (1.0 - t) * current_v
            x1_estimation[:, k, :, :] = predicted_x

        return x1_estimation

    def unnormalize_chain(self, x_chain):

        normed_x_chain_observation = x_chain[:, :, :, self.action_dim :]
        x_chain_observation = self.normalizer.unnormalize(
            to_np(normed_x_chain_observation), "observations"
        )
        normed_x_chain_action = x_chain[:, :, :, : self.action_dim]
        x_chain_action = self.normalizer.unnormalize(
            to_np(normed_x_chain_action), "actions"
        )
        x_chain = np.concatenate((x_chain_action, x_chain_observation), axis=-1)

        return x_chain

    def _apply_ellipse_constraints(self, opti, X_var, X_index_selector=None):
        obs_norm_min_0 = float(self.obs_norm_mins[0])
        obs_norm_max_0 = float(self.obs_norm_maxs[0])
        obs_norm_min_1 = float(self.obs_norm_mins[1])
        obs_norm_max_1 = float(self.obs_norm_maxs[1])

        # first obstacle
        xa = 2.0 * 0.8 / (obs_norm_max_1 - obs_norm_min_1)
        ya = 2.0 * 0.8 / (obs_norm_max_0 - obs_norm_min_0)
        circle_x, circle_y = getattr(self.cfg, "maze_circle_center", (4.5, 5.3))
        off_xa = 2.0 * (circle_y - obs_norm_min_1) / (obs_norm_max_1 - obs_norm_min_1) - 1.0
        off_ya = 2.0 * (circle_x - obs_norm_min_0) / (obs_norm_max_0 - obs_norm_min_0) - 1.0

        # second obstacle
        square_x, square_y = getattr(self.cfg, "maze_square_center", (1.5, 4.8))
        xb = 2.0 * 0.8 / (obs_norm_max_1 - obs_norm_min_1)
        yb = 2.0 * 0.8 / (obs_norm_max_0 - obs_norm_min_0)
        off_xb = 2.0 * (square_y - obs_norm_min_1) / (obs_norm_max_1 - obs_norm_min_1) - 1.0
        off_yb = 2.0 * (square_x - obs_norm_min_0) / (obs_norm_max_0 - obs_norm_min_0) - 1.0

        # apply constraints for each horizon step (except the first and last)
        for i in range(self.horizon - 2):
            if X_index_selector is None:
                x_pos = X_var[
                    -1,
                    2 * self.action_dim + i * (self.action_dim + self.state_dim) + 1,
                ]
                y_pos = X_var[
                    -1,
                    2 * self.action_dim + i * (self.action_dim + self.state_dim),
                ]
            elif X_index_selector == "flat":
                x_pos = X_var[
                    2 * self.action_dim + i * (self.action_dim + self.state_dim) + 1
                ]
                y_pos = X_var[
                    2 * self.action_dim + i * (self.action_dim + self.state_dim)
                ]
            else:
                raise NotImplementedError

            opti.subject_to(
                (
                    (1.0 / xa * (x_pos - off_xa)) ** 2
                    + (1.0 / ya * (y_pos - off_ya)) ** 2
                )
                - 1.0
                >= self.cfg.ellipse_margin
            )

            opti.subject_to(
                (
                    (1.0 / xb * (x_pos - off_xb)) ** 4
                    + (1.0 / yb * (y_pos - off_yb)) ** 4
                )
                - 1.0
                >= self.cfg.ellipse_margin
            )

    def _apply_dynamics_constraints(self, opti, X_var, X_index_selector=None):

        assert self.dynamics_model is not None, "dynamics model must be loaded"

        for i in range(self.horizon - 3):
            if X_index_selector is None:
                state_index = 2 * self.action_dim + i * (
                    self.action_dim + self.state_dim
                )
                state = X_var[-1, state_index : state_index + self.state_dim]
                next_state_index = state_index + (self.action_dim + self.state_dim)
                next_state = X_var[
                    -1, next_state_index : next_state_index + self.state_dim
                ]
                action_index = state_index - self.action_dim
                action = X_var[-1, action_index : action_index + self.action_dim]
            elif X_index_selector == "flat":
                state_index = 2 * self.action_dim + i * (
                    self.action_dim + self.state_dim
                )
                state = X_var[state_index : state_index + self.state_dim]
                next_state_index = state_index + (self.action_dim + self.state_dim)
                next_state = X_var[next_state_index : next_state_index + self.state_dim]
                action_index = state_index - self.action_dim
                action = X_var[action_index : action_index + self.action_dim]
            else:
                raise NotImplementedError

            opti.subject_to(
                cs.mtimes(self.dynamics_model["A"], cs.reshape(state, -1, 1))
                + cs.mtimes(self.dynamics_model["B"], cs.reshape(action, -1, 1))
                + cs.reshape(self.dynamics_model["c"], -1, 1)
                == cs.reshape(next_state, -1, 1)
            )

        if X_index_selector is None:
            s1_index = 2 * self.action_dim
            s1 = X_var[-1, s1_index : s1_index + self.state_dim]
            a0_index = 0
            a0 = X_var[-1, a0_index : a0_index + self.action_dim]
            sHm1_index = 2 * self.action_dim + (self.horizon - 3) * (
                self.action_dim + self.state_dim
            )
            sHm1 = X_var[-1, sHm1_index : sHm1_index + self.state_dim]
            aHm1_index = sHm1_index - self.action_dim
            aHm1 = X_var[-1, aHm1_index : aHm1_index + self.action_dim]
        elif X_index_selector == "flat":
            s1_index = 2 * self.action_dim
            s1 = X_var[s1_index : s1_index + self.state_dim]
            a0_index = 0
            a0 = X_var[a0_index : a0_index + self.action_dim]
            sHm1_index = 2 * self.action_dim + (self.horizon - 3) * (
                self.action_dim + self.state_dim
            )
            sHm1 = X_var[sHm1_index : sHm1_index + self.state_dim]
            aHm1_index = sHm1_index - self.action_dim
            aHm1 = X_var[aHm1_index : aHm1_index + self.action_dim]
        else:
            raise NotImplementedError

        opti.subject_to(
            cs.mtimes(self.dynamics_model["A"], cs.reshape(self.oc_s0_param, -1, 1))
            + cs.mtimes(self.dynamics_model["B"], cs.reshape(a0, -1, 1))
            + cs.reshape(self.dynamics_model["c"], -1, 1)
            == cs.reshape(s1, -1, 1)
        )
        opti.subject_to(
            cs.mtimes(self.dynamics_model["A"], cs.reshape(sHm1, -1, 1))
            + cs.mtimes(self.dynamics_model["B"], cs.reshape(aHm1, -1, 1))
            + cs.reshape(self.dynamics_model["c"], -1, 1)
            == cs.reshape(self.oc_sH_param, -1, 1)
        )

        # input saturation constraints
        if X_index_selector is None:
            opti.subject_to(
                X_var[-1, 0 : self.action_dim].T >= -np.ones(self.action_dim)
            )
            opti.subject_to(
                X_var[-1, 0 : self.action_dim].T <= np.ones(self.action_dim)
            )
            for i in range(self.horizon - 1):
                action_index = self.action_dim + i * (self.action_dim + self.state_dim)
                opti.subject_to(
                    X_var[-1, action_index : action_index + self.action_dim].T
                    >= -np.ones(self.action_dim)
                )
                opti.subject_to(
                    X_var[-1, action_index : action_index + self.action_dim].T
                    <= np.ones(self.action_dim)
                )
        elif X_index_selector == "flat":
            opti.subject_to(X_var[0 : self.action_dim] >= -np.ones(self.action_dim))
            opti.subject_to(X_var[0 : self.action_dim] <= np.ones(self.action_dim))
            for i in range(self.horizon - 1):
                action_index = self.action_dim + i * (self.action_dim + self.state_dim)
                opti.subject_to(
                    X_var[action_index : action_index + self.action_dim]
                    >= -np.ones(self.action_dim)
                )
                opti.subject_to(
                    X_var[action_index : action_index + self.action_dim]
                    <= np.ones(self.action_dim)
                )
        else:
            raise NotImplementedError

    def _generate_distance_objective(self, opti, X_var, X_index_selector=None):

        self.oc_distance_objective = cs.MX(0.0)

        for i in range(self.horizon - 3):
            if X_index_selector is None:
                state_index = 2 * self.action_dim + i * (
                    self.action_dim + self.state_dim
                )
                state = X_var[
                    -1,
                    state_index : state_index + self.state_dim,
                ]
                next_state_index = state_index + (self.action_dim + self.state_dim)
                next_state = X_var[
                    -1,
                    next_state_index : next_state_index + self.state_dim,
                ]
            elif X_index_selector == "flat":
                state_index = 2 * self.action_dim + i * (
                    self.action_dim + self.state_dim
                )
                state = X_var[state_index : state_index + self.state_dim]
                next_state_index = state_index + (self.action_dim + self.state_dim)
                next_state = X_var[next_state_index : next_state_index + self.state_dim]

            else:
                raise NotImplementedError

            self.oc_distance_objective += cs.sumsqr(state[0:2] - next_state[0:2])

        if X_index_selector is None:
            s1_index = 2 * self.action_dim
            s1 = X_var[-1, s1_index : s1_index + self.state_dim]
            sHm1_index = 2 * self.action_dim + (self.horizon - 3) * (
                self.action_dim + self.state_dim
            )
            sHm1 = X_var[-1, sHm1_index : sHm1_index + self.state_dim]
        elif X_index_selector == "flat":
            s1_index = 2 * self.action_dim
            s1 = X_var[s1_index : s1_index + self.state_dim]
            sHm1_index = 2 * self.action_dim + (self.horizon - 3) * (
                self.action_dim + self.state_dim
            )
            sHm1 = X_var[sHm1_index : sHm1_index + self.state_dim]
        else:
            raise NotImplementedError

        self.oc_distance_objective += cs.sumsqr(
            s1[0:2] - self.oc_s0_param[0:2].T
        ) + cs.sumsqr(sHm1[0:2] - self.oc_sH_param[0:2].T)

        self.oc_distance_objective = -1.0 * self.oc_distance_objective




    def hardflow_formulate(
        self,
        print_level=2,
        constraint="",
        objective="",
    ):
        """
        Operate on the predicted terminal state.
        """

        assert self.cfg.guidance_method in (
            "hardflow",
            "hardflow_new",
        ), f"guidance_method must be hardflow or hardflow_new, but got {self.cfg.guidance_method}"

        self.oc_cs_opti = cs.Opti()

        self.oc_N_steps = self.cfg.ode_t_steps
        self.oc_dof = (
            self.horizon * (self.state_dim + self.action_dim) - self.state_dim * 2
        )

        self.oc_s0_param = self.oc_cs_opti.parameter(1, self.state_dim)
        self.oc_sH_param = self.oc_cs_opti.parameter(1, self.state_dim)

        self.oc_t_param = self.oc_cs_opti.parameter(1)

        self.oc_X_terminal_predicted_ref = self.oc_cs_opti.parameter(self.oc_dof)
        self.oc_X_terminal_predicted = self.oc_cs_opti.variable(self.oc_dof)

        self.oc_control_cost = (
            0.5
            * cs.sumsqr(self.oc_X_terminal_predicted - self.oc_X_terminal_predicted_ref)
            * self.oc_t_param**2
        )

        if constraint == "ellipses":
            self._apply_ellipse_constraints(
                self.oc_cs_opti,
                self.oc_X_terminal_predicted,
                X_index_selector="flat",
            )
        elif constraint == "ellipses_and_dynamics":
            self._apply_ellipse_constraints(
                self.oc_cs_opti,
                self.oc_X_terminal_predicted,
                X_index_selector="flat",
            )
            self._apply_dynamics_constraints(
                self.oc_cs_opti,
                self.oc_X_terminal_predicted,
                X_index_selector="flat",
            )
        else:
            raise ValueError(f"Unsupported constraint: {constraint}")

        if objective == "":
            self.oc_cs_opti.minimize(self.oc_control_cost)
        elif objective == "distance":
            self._generate_distance_objective(
                self.oc_cs_opti,
                self.oc_X_terminal_predicted,
                X_index_selector="flat",
            )
            self.oc_cs_opti.minimize(
                self.oc_control_cost
                - self.oc_distance_objective * self.cfg.hardflow_cost_scale
            )

        solver_opts = {
            "ipopt.print_level": print_level,
            "ipopt.hessian_approximation": "limited-memory",
        }
        self.oc_cs_opti.solver("ipopt", solver_opts)

    def warmstart(self, conditions):

        x = self.sample_initial_noise(self.cfg.warmstart_batch)

        x = apply_conditioning(
            x, to_torch(conditions, device=x.device), self.action_dim
        )
        x_chain = [x.clone()]
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t = k / self.oc_N_steps
            dx_dt = self.flow_model(x, to_torch(t, device=x.device))
            dx_dt = apply_conditioning_from_conditioned_x(
                dx_dt, torch.zeros_like(x), conditions, self.action_dim
            )
            x = x + dx_dt * dt
            x_chain.append(x)

        values = self.value_model(x_chain[-1])
        best_idx = torch.argmax(values, dim=0)

        best_x_chain = torch.stack(
            [x_step[best_idx].squeeze(0) for x_step in x_chain], dim=0
        )  # (oc_N_steps+1, planning_horizon, transition_dim)
        best_x_chain = best_x_chain.reshape(
            self.oc_N_steps + 1, -1
        )  # (oc_N_steps+1, planning_horizon * transition_dim)
        best_s0 = best_x_chain[0, self.action_dim : self.state_dim + self.action_dim]
        best_sH = best_x_chain[0, -self.state_dim :]
        best_dof_chain_part_1 = best_x_chain[:, : self.action_dim]
        best_dof_chain_part_2 = best_x_chain[
            :,
            self.action_dim
            + self.state_dim : self.horizon * (self.action_dim + self.state_dim)
            - self.state_dim,
        ]
        best_dof_chain = torch.cat(
            [best_dof_chain_part_1, best_dof_chain_part_2], dim=-1
        )

        best_s0_np = to_np(best_s0)
        best_sH_np = to_np(best_sH)
        best_dof_chain_np = to_np(best_dof_chain)

        return best_s0_np, best_sH_np, best_dof_chain_np




    def hardflow_new_forward(self, conditions, batch_size=1):
        """
        Variant of hardflow_forward that evaluates the reference flow with
        PyTorch directly instead of going through the l4casadi bridge. The
        per-step QP formulation (`hardflow_formulate`) is unchanged because it
        only references algebraic constraints, so the optimization itself does
        not need the neural network.
        """
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "hardflow_new"
        ), f"guidance_method must be hardflow_new, but got {self.cfg.guidance_method}"

        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        best_s0_np, best_sH_np, best_dof_chain_np = self.warmstart(conditions)

        t_start = time.time()

        device = self.cfg.device
        best_s0_torch = to_torch(best_s0_np, device=device).reshape(1, self.state_dim)
        best_sH_torch = to_torch(best_sH_np, device=device).reshape(1, self.state_dim)

        def flow_eval_np(x_np, t):
            x_torch = to_torch(x_np, device=device).reshape(1, self.oc_dof)
            with torch.no_grad():
                v = self.constrained_flow_fn_torch(
                    x_torch, t, best_s0_torch, best_sH_torch
                )
            return to_np(v).reshape(-1)

        self.oc_cs_opti.set_value(self.oc_s0_param, best_s0_np)
        self.oc_cs_opti.set_value(self.oc_sH_param, best_sH_np)

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]
            v_k = flow_eval_np(x_k, t_k)
            x_next_ref = x_k + v_k * dt

            control_flag = True
            if self.cfg.hardflow_activation == "all":
                pass
            elif self.cfg.hardflow_activation == "late":
                if k < self.oc_N_steps // 2:
                    control_flag = False
            else:
                raise ValueError(
                    f"Unsupported hardflow_activation: {self.cfg.hardflow_activation}"
                )

            if control_flag:
                v_next = flow_eval_np(x_next_ref, t_k + dt)
                x_terminal_predicted_ref = x_next_ref + (1.0 - t_k - dt) * v_next

                self.oc_cs_opti.set_value(self.oc_t_param, t_k + dt)

                self.oc_cs_opti.set_value(
                    self.oc_X_terminal_predicted_ref, x_terminal_predicted_ref
                )
                self.oc_cs_opti.set_initial(
                    self.oc_X_terminal_predicted, x_terminal_predicted_ref
                )

                try:
                    sol = self.oc_cs_opti.solve_limited()
                    x_terminal_predicted = sol.value(self.oc_X_terminal_predicted)
                except RuntimeError as e:
                    print("Solver failed, returning last available value.")
                    x_terminal_predicted = self.oc_cs_opti.debug.value(
                        self.oc_X_terminal_predicted
                    )

                x_next = x_next_ref + (t_k + dt) * (
                    x_terminal_predicted - x_terminal_predicted_ref
                )
            else:
                x_next = x_next_ref

            x_next = np.array(x_next).flatten()

            u_k = (x_next - x_next_ref) / dt
            U_optimized.append(u_k)

            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)
        optimized_x_chain = np.concatenate(
            [
                optimized_dof[:, : self.action_dim],
                np.tile(best_s0_np, (self.oc_N_steps + 1, 1)),
                optimized_dof[:, self.action_dim :],
                np.tile(best_sH_np, (self.oc_N_steps + 1, 1)),
            ],
            axis=1,
        )
        optimized_x_chain = np.reshape(
            optimized_x_chain,
            (
                batch_size,
                self.oc_N_steps + 1,
                self.horizon,
                self.state_dim + self.action_dim,
            ),
        )

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = np.insert(optimized_final_dof, self.action_dim, best_s0_np)
        optimized_final_x = np.append(optimized_final_x, best_sH_np)
        optimized_final_x = to_torch(optimized_final_x, device="cpu")
        optimized_final_x = optimized_final_x.reshape(
            1, self.horizon, self.action_dim + self.state_dim
        )

        normed_actions = optimized_final_x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, "actions")
        actions = to_np(actions)
        normed_observations = optimized_final_x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")
        observations = to_np(observations)
        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)
        trajectories = Trajectories(actions, observations, values)

        action = actions[0, 0]

        x_chain = to_torch(optimized_x_chain, device=self.cfg.device)
        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {
            "computation_time": computation_time,
        }
        return action, trajectories, x_chain, x1_estimation, info



    def constrained_flow_fn_torch(self, dof, t, s0, sH):

        device = dof.device

        assert dof.shape == (1, self.oc_dof)

        full_traj = torch.zeros(
            1, self.horizon * (self.action_dim + self.state_dim), device=device
        )

        full_traj[:, : self.action_dim] = dof[:, : self.action_dim]
        full_traj[:, self.action_dim : self.action_dim + self.state_dim] = s0
        full_traj[:, self.action_dim + self.state_dim : -self.state_dim] = dof[
            :, self.action_dim :
        ]
        full_traj[:, -self.state_dim :] = sH

        full_traj = full_traj.reshape(1, self.horizon, self.action_dim + self.state_dim)

        if isinstance(t, (float, int)):
            t = torch.tensor(t, device=device, dtype=full_traj.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(1)

        flow_output = self.flow_model(full_traj, t)

        flow_output_flat = flow_output.reshape(1, -1)

        dof_flow = torch.zeros(1, dof.shape[1], device=device)
        dof_flow[:, : self.action_dim] = flow_output_flat[:, : self.action_dim]
        dof_flow[:, self.action_dim :] = flow_output_flat[
            :, self.action_dim + self.state_dim : -self.state_dim
        ]

        return dof_flow
