"""Independent Maze2D wrapper of the existing algebraic HardFlow sampler.

The per-layer fallback is inherited from backend.HardFlowBackend, whose equations are also
those in root hardflow/models_flow/flow_policy.py:hardflow_new_forward. Maze2D
requires both endpoint conditions. No original function is replaced.
"""
import os
import numpy as np
from .backend import HardFlowBackend, Coordinates, ROOT, sha
from .core import fingerprint


def apply_action_bounds(opti, variable, coordinates, action_dim):
    """Retain native action limits when the dynamics equalities are disabled."""
    action_mask = np.zeros(coordinates.shape, dtype=bool)
    action_mask[:, :action_dim] = True
    indices = np.flatnonzero(action_mask[coordinates.mask]).tolist()
    opti.subject_to(opti.bounded(-1, variable[indices], 1))


def second_difference_objective(variable,s0,sH,horizon,action_dim,state_dim):
    """Squared second differences of normalized positions, including both fixed endpoints."""
    import casadi as cs
    positions=[cs.reshape(s0[:2],2,1)]
    for frame in range(1,horizon-1):
        offset=action_dim+(frame-1)*(action_dim+state_dim)+action_dim
        positions.append(variable[offset:offset+2])
    positions.append(cs.reshape(sH[:2],2,1))
    p=cs.horzcat(*positions)
    return cs.sumsqr(p[:,2:]-2*p[:,1:-1]+p[:,:-2])


def velocity_smoothness_objective(variable,s0,sH,horizon,action_dim,state_dim):
    """Squared adjacent differences of normalized (vx,vy), including endpoints."""
    import casadi as cs
    velocities=[cs.reshape(s0[2:4],2,1)]
    for frame in range(1,horizon-1):
        offset=action_dim+(frame-1)*(action_dim+state_dim)+action_dim+2
        velocities.append(variable[offset:offset+2])
    velocities.append(cs.reshape(sH[2:4],2,1))
    v=cs.horzcat(*velocities)
    return cs.sumsqr(v[:,1:]-v[:,:-1])


class Maze2D(HardFlowBackend):
    def __init__(self, settings):
        os.environ.setdefault('HARDFLOW_NATIVE_MUJOCO','1')
        import torch
        import casadi as cs
        from hardflow.config.flow_matching import FlowMatchingEvaluationConfig
        from hardflow.models_flow.flow_policy import FlowPolicy
        from hardflow.models_flow.unet import TemporalUnet
        from hardflow.native_maze2d_data import load_compact_data
        from hardflow.models_flow.value_model import ProxyValueModel
        self.settings=settings;self.torch=torch;self.cs=cs
        cfg=FlowMatchingEvaluationConfig()
        for key,value in settings['hardflow'].items():
            if not hasattr(cfg,key):raise ValueError(f'Unknown Maze2D configuration: {key}')
            setattr(cfg,key,value)
        if cfg.device!='cpu' or cfg.guidance_method!='hardflow_new' or cfg.warmstart_batch!=1:
            raise ValueError('Only CPU, hardflow_new, one initial noise supported')
        self.cfg=cfg
        cost_kind=settings.get('terminal_cost_kind','first_difference')
        if cost_kind not in ('first_difference','second_difference','velocity_first_difference'):raise ValueError('Unsupported terminal cost kind')
        if cost_kind in ('second_difference','velocity_first_difference') and cfg.cost!='distance':raise ValueError('Smoothness objective requires an enabled task cost')
        if 'terminal_cost_weight' in settings:
            weight=float(settings['terminal_cost_weight'])
            if not np.isfinite(weight) or weight<=0 or cfg.cost!='distance' or not np.isclose(cfg.hardflow_cost_scale,800*weight):
                raise ValueError('Terminal-cost weight must match the original 800-scale distance objective')
        self.env,_,self.normalizer=load_compact_data(cfg)
        if 'goal' in settings:
            goal=np.asarray(settings['goal'],dtype=float)
            if goal.shape!=(2,) or not np.isfinite(goal).all():
                raise ValueError('Maze2D goal must be a finite [x, y] position')
            self.env._target=goal.copy()
        dynamics=ROOT/settings['dynamics'];checkpoint=ROOT/settings['checkpoint']
        with np.load(dynamics,allow_pickle=True) as data:
            self.dynamics={k:data[k].copy() for k in ('A','B','c')}
            dynamics_normalizer=data['normalizer'].item()
            for key in ('observations','actions'):
                for attr in ('mins','maxs'):
                    if not np.array_equal(getattr(self.normalizer.normalizers[key],attr),getattr(dynamics_normalizer.normalizers[key],attr)):
                        raise ValueError('Dynamics and checkpoint dataset normalization differ')
        model=TemporalUnet(horizon=cfg.horizon,transition_dim=cfg.action_dim+cfg.state_dim,
                           cond_dim=cfg.state_dim,dim=32,dim_mults=(1,4,8),attention=False).to('cpu')
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        self.model=model
        class CapturingPolicy(FlowPolicy):
            def unnormalize_chain(self,chain):
                self.captured_chain=chain.detach().cpu().numpy().copy()
                return super().unnormalize_chain(chain)
        value=ProxyValueModel(cfg.horizon,cfg.action_dim,cfg.state_dim,objective=cfg.value_objective,
                             constraint=cfg.constraint,value_objective_scale=cfg.value_objective_scale,
                             value_constraint_scale=cfg.value_constraint_scale,normalizer=self.normalizer,
                             dynamics_model=self.dynamics,ellipse_margin=cfg.ellipse_margin,
                             maze_square_center=cfg.maze_square_center,maze_circle_center=cfg.maze_circle_center)
        self.policy=CapturingPolicy(model,value,self.normalizer,cfg.action_dim,cfg.state_dim,cfg.horizon,cfg,dynamics_model=self.dynamics)
        self.policy.hardflow_formulate(print_level=cfg.solver_print_level,constraint=cfg.constraint,objective=cfg.cost)
        # Root Maze2D has no hardflow_reg_scale configuration. Apply the requested
        # coefficient to this local NLP instance only, preserving every constraint.
        self.regularizer=float(settings['cost']['coefficient'])
        if settings['cost']['kind']!='hf':raise ValueError('Maze2D wrapper currently supports HF cost only')
        objective=self.regularizer*self.policy.oc_control_cost
        task_expression=None
        if cfg.cost=='distance':
            if cost_kind in ('second_difference','velocity_first_difference'):
                smooth=second_difference_objective if cost_kind=='second_difference' else velocity_smoothness_objective
                task_expression=cfg.hardflow_cost_scale*smooth(self.policy.oc_X_terminal_predicted,self.policy.oc_s0_param,self.policy.oc_sH_param,cfg.horizon,cfg.action_dim,cfg.state_dim)
            else:task_expression=-cfg.hardflow_cost_scale*self.policy.oc_distance_objective
            objective+=task_expression
        elif cfg.cost!='':raise ValueError('Unsupported task cost')
        self.policy.oc_cs_opti.minimize(objective)
        self.grid=np.arange(cfg.ode_t_steps+1,dtype=float)/cfg.ode_t_steps
        if 'tail_steps' in settings:
            from .denoise_schedule import tail_start
            self.control_start=tail_start(self)
        self.conditions={0:np.array(settings['observation'],float),cfg.horizon-1:np.array([*self.env._target,0.,0.])}
        normalized={i:self.normalizer.normalize(o,'observations').astype(np.float32) for i,o in self.conditions.items()}
        self.coordinates=Coordinates(cfg.horizon,cfg.action_dim,cfg.state_dim,normalized)
        if self.coordinates.free_dim!=self.policy.oc_dof:raise ValueError('Free mask differs from native Maze2D sampler')
        self.s0=normalized[0];self.sH=normalized[cfg.horizon-1]
        self.s0_torch=torch.tensor(self.s0).reshape(1,-1);self.sH_torch=torch.tensor(self.sH).reshape(1,-1)
        self.policy.oc_cs_opti.set_value(self.policy.oc_s0_param,self.s0)
        self.policy.oc_cs_opti.set_value(self.policy.oc_sH_param,self.sH)
        opti=self.policy.oc_cs_opti
        if settings.get('action_bounds_without_dynamics', False):
            if cfg.constraint != 'ellipses':
                raise ValueError('Separate action bounds require ellipses without dynamics')
            apply_action_bounds(opti,self.policy.oc_X_terminal_predicted,self.coordinates,cfg.action_dim)
        if 'solver_hessian' in settings:
            if settings['solver_hessian'] not in ('exact', 'limited-memory'):
                raise ValueError('Unsupported solver Hessian setting')
            opti.solver('ipopt',{'print_time':False,'ipopt.print_level':cfg.solver_print_level,
                'ipopt.hessian_approximation':settings['solver_hessian']})
        inputs=[self.policy.oc_X_terminal_predicted,self.policy.oc_s0_param,self.policy.oc_sH_param]
        self.constraint_function=cs.Function('maze_memory_constraints',inputs,[opti.g,opti.lbg,opti.ubg])
        self.task_function=cs.Function('maze_memory_task',inputs,[task_expression]) if task_expression is not None else None
        normalization={k:{'mins':self.normalizer.normalizers[k].mins.tolist(),'maxs':self.normalizer.normalizers[k].maxs.tolist()} for k in ('observations','actions')}
        self.identity={'backend':'maze2d-hardflow-new-local-reg','checkpoint_sha256':sha(checkpoint),'dynamics_sha256':sha(dynamics),
            'source_digest':fingerprint({p:sha(ROOT/p) for p in ('hardflow/models_flow/flow_policy.py','hardflow/models_flow/unet.py','hardflow/native_maze2d_data.py')}),
            'model_configuration':vars(cfg),'regularizer':self.regularizer,'condition':{str(i):o.tolist() for i,o in self.conditions.items()},
            'normalization':normalization,'normalized_condition':{str(i):o.tolist() for i,o in normalized.items()},
            'time_grid':self.grid.tolist(),'full_shape':list(self.coordinates.shape),'free_mask':self.coordinates.mask.tolist(),'free_dim':self.coordinates.free_dim}
        if 'tail_steps' in settings:self.identity['tail_steps']=settings['tail_steps']
        if 'terminal_cost_kind' in settings:self.identity['terminal_cost_kind']=cost_kind
        if settings.get('action_bounds_without_dynamics', False):
            self.identity['action_bounds_without_dynamics']=True
        if 'solver_hessian' in settings:
            self.identity['solver_hessian']=settings['solver_hessian']

    def velocity(self,x,t):
        with self.torch.no_grad():
            v=self.policy.constrained_flow_fn_torch(self.torch.tensor(x,dtype=self.torch.float32).reshape(1,-1),t,self.s0_torch,self.sH_torch)
        return v.cpu().numpy().reshape(-1)

    def terminal(self,x):
        g,lo,hi=(np.array(a).reshape(-1) for a in self.constraint_function(x,self.s0,self.sH))
        violation=np.maximum(np.maximum(lo-g,g-hi),0);equal=np.isfinite(lo)&np.isfinite(hi)&(lo==hi)
        residual={'max':float(np.max(violation,initial=0)),'equality_max':float(np.max(violation[equal],initial=0)),
                  'inequality_max':float(np.max(violation[~equal],initial=0)),'count':len(g)}
        cost=float(self.task_function(x,self.s0,self.sH)) if self.task_function is not None else 0.
        return bool(np.isfinite(x).all() and np.isfinite(g).all() and residual['max']<=self.settings['tolerances']['feasibility']),cost,residual
