"""IterFlow history-ball constraint on the native affine HardFlow successor."""
import time
import numpy as np
from .maze2d import Maze2D
from .compare import physical_observations
from .refinement_backend import RefinementMaze2D

def minimum_ball(points, tolerance=1e-8):
    """SOCP in the exact affine span (SVD); verify primal and dual gap.

    Dual max: sum(alpha*||q||²)-||sum(alpha*q)||² on the simplex.
    The primal upper radius and dual lower radius certify numerical accuracy.
    """
    from scipy.optimize import minimize
    p = np.asarray(points, float)
    if p.ndim != 2 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Invalid history')
    base = p[0]; a = p-base
    _, singular, vt = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(singular > np.finfo(float).eps*max(a.shape)*max(1., singular.max(initial=0))))
    if not rank:
        return base.copy(), 0., dict(rank=0, containment=0., radius_gap=0.)
    basis = vt[:rank]; scale=float(np.max(np.linalg.norm(a,axis=1)))
    q = (a@basis.T)/scale; norms = np.sum(q*q, axis=1)
    def f(w): return float(np.sum((w@q)**2)-w@norms)
    def jac(w): return 2*q@(w@q)-norms
    result = minimize(f, np.full(len(p), 1/len(p)), jac=jac, method='SLSQP',
        bounds=[(0,1)]*len(p), constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(w))}],
        options={'ftol':1e-13,'maxiter':2000})
    w = np.maximum(result.x, 0); w /= w.sum(); center = base+scale*((w@q)@basis)
    radius = float(np.linalg.norm(p-center,axis=1).max())
    lower = scale*np.sqrt(max(0., -f(w))); gap = radius-lower
    refinement=None
    if gap > tolerance*max(1.,radius):
        # Polish the primal convex QCQP. A near-optimal dual objective can stop
        # before its center is accurate enough for a radius certificate.
        import casadi as cs
        opti=cs.Opti();z=opti.variable(rank);r2=opti.variable();constraints=[]
        for point in q:
            constraint=cs.sumsqr(z-cs.DM(point))<=r2
            opti.subject_to(constraint);constraints.append(constraint)
        opti.subject_to(r2>=0);opti.minimize(r2)
        opti.set_initial(z,w@q);opti.set_initial(r2,max(np.sum((q-w@q)**2,axis=1)))
        opti.solver('ipopt',{'print_time':False,'ipopt.print_level':0,'ipopt.tol':1e-12,
            'ipopt.constr_viol_tol':1e-12,'ipopt.acceptable_tol':1e-12,'ipopt.max_iter':1000})
        solution=opti.solve()
        center=base+scale*(np.asarray(solution.value(z)).ravel()@basis)
        radius=float(np.linalg.norm(p-center,axis=1).max())
        weights=np.maximum([float(solution.value(opti.dual(c))) for c in constraints],0)
        weights/=weights.sum();lower=max(lower,scale*np.sqrt(max(0.,-f(weights))))
        gap=radius-lower
        refinement=dict(solver='IPOPT primal convex QCQP',iterations=int(solution.stats()['iter_count']))
    if gap > tolerance*max(1.,radius):
        raise RuntimeError(f'MEB numerical certificate failed: gap={gap}, {result.message}')
    return center, radius, dict(rank=rank, containment=float(np.max(np.linalg.norm(p-center,axis=1)-radius)),
        radius_gap=float(gap), dual_radius=float(lower), iterations=int(result.nit), solver_status=str(result.message),primal_refinement=refinement)

def segment_box_distance(a, b, center, half):
    """Exact minimum Euclidean distance to an axis-aligned box along a segment."""
    a = np.asarray(a)-center; v = np.asarray(b)-center-a
    breaks = [0.,1.]
    for k in range(2):
        if v[k] != 0:
            breaks.extend(float(t) for t in ((-half[k]-a[k])/v[k], (half[k]-a[k])/v[k]) if 0<t<1)
    breaks = sorted(set(breaks)); candidates = list(breaks)
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        mid = a+(lo+hi)/2*v
        active = np.abs(mid)>half
        offset = a-np.sign(mid)*half
        denom = np.sum(v[active]**2)
        if denom: candidates.append(float(np.clip(-np.dot(offset[active],v[active])/denom,lo,hi)))
    return min(float(np.linalg.norm(np.maximum(np.abs(a+t*v)-half,0))) for t in candidates)

def evaluation_distance_cost(full_state,action_dim,scale):
    positions=np.asarray(full_state)[:,action_dim:action_dim+2]
    return float(scale*np.sum(np.diff(positions,axis=0)**2))

class HistoryMaze(Maze2D):
    def __init__(self, settings):
        super().__init__(settings)
        if self.regularizer != 1: raise ValueError('Use coefficient 1: unchanged upstream baseline')
        if tuple(self.cfg.ode_t_span)!=(0,1) or self.cfg.ode_solver!='euler' or self.cfg.flow_matching_type!='cfm':
            raise ValueError('Only existing linear CFM Euler grid supported')
        self.lambda_oc = float(settings.get('control_cost_weight',self.regularizer*(self.grid[1]-self.grid[0])))
        if not np.isfinite(self.lambda_oc) or self.lambda_oc<0:raise ValueError('Control cost weight must be finite and nonnegative')
        if 'control_cost_weight' in settings:self.identity['control_cost_weight']=self.lambda_oc
        self.env.set_state(np.zeros(2),np.zeros(2))
        import mujoco
        model = self.env.model
        particle = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'particle_geom')
        origin = self.env.data.geom_xpos[particle,:2].copy()
        self.walls = []
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
            if name.startswith('wall_'):
                self.walls.append((self.env.data.geom_xpos[g,:2].copy()-origin,model.geom_size[g,:2].copy(),
                    float(model.geom_size[particle,0]+model.geom_margin[particle]+model.geom_margin[g])))
        if not self.walls: raise ValueError('Native wall geometry missing')
        self.wall_geometry = dict(origin=origin.tolist(),walls=[dict(center=c.tolist(),half=h.tolist(),clearance=r) for c,h,r in self.walls],
            units='observation qpos; world wall centers minus zero-qpos particle center')
        self.symbolic_task = self.task_function
        if 'evaluation_distance_cost_scale' in settings:
            scale=float(settings['evaluation_distance_cost_scale'])
            if not np.isfinite(scale) or scale<0:raise ValueError('Invalid evaluation-only distance scale')
            if self.symbolic_task is not None:raise ValueError('Evaluation-only distance requires no distance in the inner optimizer')
            self.identity['evaluation_distance_cost_scale']=scale
        wall_weight=float(settings.get('rollout_wall_cost_weight',0.))
        if not np.isfinite(wall_weight) or wall_weight<0:raise ValueError('Invalid rollout wall-cost weight')
        if wall_weight:
            self.identity['rollout_evaluation']=dict(wall_cost_weight=wall_weight,
                max_steps=settings.get('rollout_max_steps',800),seed=settings.get('rollout_seed',0),
                counting='native particle-wall contact steps',controller='PD 5/1',goal_radius=.5)

    def final_check(self,x):
        explicit,cost,residual = self.terminal(x)
        if 'evaluation_distance_cost_scale' in self.settings:
            cost=evaluation_distance_cost(self.coordinates.full(x),self.cfg.action_dim,self.settings['evaluation_distance_cost_scale'])
        obs = physical_observations(self,x); hits=[]; clearance=float('inf')
        for k,(a,b) in enumerate(zip(obs[:-1,:2],obs[1:,:2])):
            for w,(c,h,r) in enumerate(self.walls):
                gap=segment_box_distance(a,b,c,h)-r; clearance=min(clearance,gap)
                if gap < -1e-8: hits.append([k,w,float(gap)])
        # Continuous versions of the SAME two explicit obstacles, evaluated separately.
        extra = RefinementMaze2D.validate_path(self,x)
        start=float(np.linalg.norm(obs[0]-self.conditions[0])); goal=float(np.linalg.norm(obs[-1]-self.conditions[self.cfg.horizon-1]))
        strict = explicit and not hits and extra['feasible'] and max(start,goal)<=1e-5
        tolerance = getattr(self,'wall_tolerance',0.)
        violation = max(0.,-clearance)
        accepted = explicit and violation<=tolerance+1e-8 and extra['feasible'] and max(start,goal)<=1e-5
        geometry_accepted=accepted
        mode=getattr(self,'admission_mode','tolerant_geometry')
        if mode=='original_explicit':accepted=explicit and max(start,goal)<=1e-5
        if mode=='soft_cost':accepted=np.isfinite(x).all() and np.isfinite(cost) and max(start,goal)<=1e-5
        return dict(safe=bool(accepted),strict_safe=bool(strict),admission_mode=mode,geometry_admission_pass=bool(geometry_accepted),wall_tolerance=tolerance,wall_max_violation=violation,
            explicit_feasible=explicit,explicit_residual=residual,C=cost,
            wall_collision=bool(hits),wall_segment_hits=hits,min_wall_clearance=clearance,
            continuous_extra_obstacles=extra,start_error=start,goal_error=goal)

    def solve_step(self,i,x,center,radius,history,rng,max_iter):
        cs=self.cs; dt=self.grid[i+1]-self.grid[i]; t=float(self.grid[i+1]); ref=self.natural(i,x)
        pred=ref+(1-t)*self.velocity(ref,t)
        opti=cs.Opti(); y=opti.variable(self.coordinates.free_dim)
        actual=ref+t*(y-pred)  # EXACT native Algorithm 1 affine map F_t(y).
        g,lower,upper=self.constraint_function(y,self.s0,self.sH)
        _,lo,hi=self.constraint_function(pred,self.s0,self.sH)
        lo=np.asarray(lo).ravel();hi=np.asarray(hi).ravel()
        for k in range(len(lo)):
            if lo[k]==hi[k]: opti.subject_to(g[k]==lo[k])
            else:
                if np.isfinite(lo[k]):opti.subject_to(g[k]>=lo[k])
                if np.isfinite(hi[k]):opti.subject_to(g[k]<=hi[k])
        enforce_history_ball=getattr(self,'enforce_history_ball',True)
        if enforce_history_ball:
            opti.subject_to(cs.sumsqr(actual-center)<=radius**2)
        control_bound=None
        if getattr(self,'control_norm_bounds',None) is not None:
            control_bound=float(self.control_norm_bounds[i])
            if not np.isfinite(control_bound) or control_bound<0:raise ValueError('Invalid control norm bound')
            # u_i=(F(y)-natural_step)/dt: constrain it inside the surrogate NLP.
            opti.subject_to(cs.sumsqr(actual-ref)<=(dt*control_bound)**2)
        task=self.symbolic_task(y,self.s0,self.sH) if self.symbolic_task else 0
        objective=task+self.regularizer*.5*t*t*cs.sumsqr(y-pred)
        opti.minimize(objective)
        opti.solver('ipopt',{'print_time':False,'ipopt.print_level':0,'ipopt.hessian_approximation':self.settings.get('solver_hessian','limited-memory'),
            'ipopt.max_iter':max_iter,'ipopt.tol':1e-8,'ipopt.constr_viol_tol':1e-8})
        starts=[np.asarray(history[-1]).copy()]
        if enforce_history_ball:
            for _ in range(2):
                direction=rng.normal(size=len(x));direction/=np.linalg.norm(direction)
                starts.append(center+radius*rng.random()**(1/len(x))*direction)
        else:
            # No ball is fitted or used even for initialization in this ablation.
            starts.extend([ref.copy(),ref+.01*rng.normal(size=len(x))])
        attempts=[]; candidates=[]
        for k,initial in enumerate(starts):
            opti.set_initial(y,pred+(initial-ref)/t)
            strategy='history_reference' if k==0 else ('uniform_ball' if enforce_history_ball else ('natural_reference' if k==1 else 'gaussian_perturbation'))
            begun=time.perf_counter();detail=dict(initial_strategy=strategy,initial_actual=initial.tolist(),history_ball_enforced=enforce_history_ball)
            try:
                solution=opti.solve_limited();stats=solution.stats()
                yy=np.asarray(solution.value(y)).ravel(); xx=ref+t*(yy-pred)
                valid,_,residual=self.terminal(yy);ball=float(np.linalg.norm(xx-center)-radius) if enforce_history_ball else 0.
                value=float(solution.value(objective));condition=0. # P is free coordinates; reinjection is exact.
                control_norm=float(np.linalg.norm((xx-ref)/dt))
                control_violation=0. if control_bound is None else max(0.,control_norm-control_bound)
                feasible=bool(valid and ball<=self.settings['tolerances']['feasibility'] and np.isfinite(value)
                    and control_violation<=self.settings['tolerances']['feasibility'])
                detail.update(status=stats.get('return_status'),solver_success=bool(stats.get('success')),iterations=int(stats.get('iter_count',0)),
                    feasible=feasible,predicted_residual=residual,ball_violation=max(0.,ball),conditioning_violation=condition,objective=value,
                    control_norm=control_norm,control_norm_bound=control_bound,control_bound_violation=control_violation)
                if feasible:candidates.append((value,xx,yy,k))
            except RuntimeError as error:
                stats=opti.stats(); detail.update(feasible=False,status=stats.get('return_status'),iterations=int(stats.get('iter_count',0)),error=str(error))
            detail['seconds']=time.perf_counter()-begun;attempts.append(detail)
        if not candidates:return None,dict(attempts=attempts,reason='no_feasible_multistart_solution',ref=ref.tolist())
        value,xx,yy,k=min(candidates,key=lambda a:a[0])
        return (xx,(xx-ref)/dt,ref,yy),dict(attempts=attempts,selected_start=k,objective=value,reason='feasible',
            predicted_residual=self.terminal(yy)[2],history_ball_enforced=enforce_history_ball,
            ball_violation=max(0.,float(np.linalg.norm(xx-center)-radius)) if enforce_history_ball else 0.,
            control_norm=float(np.linalg.norm((xx-ref)/dt)),control_norm_bound=control_bound,
            control_bound_violation=0. if control_bound is None else max(0.,float(np.linalg.norm((xx-ref)/dt))-control_bound))

def metrics(backend,run,original,seconds,rollout_path=None):
    from .denoise_schedule import tail_steps
    active=tail_steps(backend)
    check=backend.final_check(run['X'][-1]);dt=np.diff(backend.grid);energy=.5*np.sum(run['U']**2,axis=1)*dt
    rms=float(np.sqrt(np.mean((physical_observations(backend,run['X'][-1])[:,:2]-physical_observations(backend,original['X'][-1])[:,:2])**2)))
    raw_j=check['C']+backend.lambda_oc*float(energy.sum());soft_j=raw_j+getattr(backend,'constraint_weight',1.)*check.get('constraint_penalty',0.)
    result=dict(**check,E=float(energy.sum()),E_tail=float(energy[-active:].sum()),E_start=float(energy[-active-1]) if len(energy)>active else 0.,J=soft_j,J_without_constraint_penalty=raw_j,
        control_norms=np.linalg.norm(run['U'],axis=1).tolist(),RMS_physical_position=rms,seconds=seconds)
    weight=getattr(backend,'settings',{}).get('terminal_cost_weight')
    if weight is not None:
        result.update(terminal_cost_weight=weight,C_unweighted=check['C']/weight)
    from .rollout_wall_cost import apply_rollout_cost
    return apply_rollout_cost(result,backend,run,rollout_path)

def save_run(path,backend,run,noise,predictions=None):
    np.savez_compressed(path,X=run['X'],U=run['U'],refs=run['refs'],noise=noise,time_grid=backend.grid,
        full_X=np.stack([backend.coordinates.full(x) if np.isfinite(x).all() else np.full(backend.coordinates.shape,np.nan) for x in run['X']]),
        physical_final=physical_observations(backend,run['X'][-1]),predicted_endpoints=np.asarray([] if predictions is None else predictions))
