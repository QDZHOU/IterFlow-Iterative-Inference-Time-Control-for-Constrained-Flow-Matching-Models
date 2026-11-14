"""Diverse initial-noise sampling with hard HF constraints and fixed known tail balls."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from .history_ball_surrogate import HistoryMaze,minimum_ball,save_run,metrics
from .compare import NaturalFlow
from .core import sample,canonical
from .io import write_json
from .denoise_schedule import tail_start
from .history_ball_noise import (diverse_noise, cost_improves, fit_history_balls, reachable_noise, exploitation_probability, search_region, compact_report, resume_settings_match)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--source-rounds',type=int,default=400)
    p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=50)
    p.add_argument('--noise-radius',type=float,default=1.)
    p.add_argument('--pool-size',type=int,default=64);p.add_argument('--expand-after',type=int,default=10)
    p.add_argument('--growth',type=float,default=1.5);p.add_argument('--seed',type=int,default=20261013)
    p.add_argument('--max-iter',type=int,default=300)
    p.add_argument('--require-improvement',action='store_true')
    p.add_argument('--improvement-tolerance',type=float,default=1e-8)
    p.add_argument('--epsilon-scale',type=float,default=1.)
    p.add_argument('--update-balls',action='store_true')
    p.add_argument('--spread-between-samples',action='store_true')
    p.add_argument('--entry-screen',action='store_true')
    p.add_argument('--prefer-far-from-center',action='store_true',help='Prioritize distance from initial noise among screened reachable candidates')
    p.add_argument('--noise-max-radius',type=float,default=float('inf'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--mcmc-start',type=int,default=151)
    p.add_argument('--temperature',type=float,default=20.)
    p.add_argument('--walk-sigma',type=float,default=.1)
    p.add_argument('--initial-reference',choices=['hardflow'],default='hardflow')
    p.add_argument('--control-norm-ratio',type=float,help='Per-step u norm cap relative to latest admitted trajectory, e.g. 0.99')
    a=p.parse_args()
    if a.control_norm_ratio is not None and (not np.isfinite(a.control_norm_ratio) or not 0<a.control_norm_ratio<=1):raise ValueError('Control norm ratio must be in (0,1]')
    if a.noise_radius<=0 or not np.isfinite(a.noise_radius) or a.growth<=1 or a.expand_after<1 or a.pool_size<1 or a.rounds<1 or not np.isfinite(a.epsilon_scale) or a.epsilon_scale<=0 or a.improvement_tolerance<0 or a.noise_max_radius<a.noise_radius or (a.prefer_far_from_center and not a.entry_screen):raise ValueError('Invalid sampling settings')
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and not a.resume:raise ValueError('Output already initialized')
    settings=json.loads((source/'config.json').read_text())['settings']
    backend=HistoryMaze(settings);backend.torch.set_num_threads(int(os.environ.get('HARDFLOW_TORCH_THREADS','4')))
    backend.admission_mode='original_explicit';backend.wall_tolerance=.06
    if canonical(backend.spec())!=canonical(json.loads((source/'spec.json').read_text())):raise ValueError('Source identity mismatch')
    experiment=dict(vars(a),constraint_mode='hard',admission_mode='original_explicit',wall_tolerance=.06,
        variable_initial_noise=True,tail_balls_frozen=not a.update_balls,noise_strategy='uniform-volume pool maximin',
        cost_metric='J=0.1E' if settings['hardflow']['cost']=='' else 'J=C+0.1E')
    if 'terminal_cost_weight' in settings:
        experiment['cost_metric']=f'J={settings["terminal_cost_weight"]:g}C_base+0.1E'
    if settings.get('terminal_cost_kind')=='second_difference':experiment['cost_metric']=f'J={settings["hardflow"]["hardflow_cost_scale"]:g}*sum_squared_second_position_differences+0.1E'
    if settings.get('terminal_cost_kind')=='velocity_first_difference':experiment['cost_metric']=f'J={settings["hardflow"]["hardflow_cost_scale"]:g}*sum_squared_adjacent_velocity_differences+0.1E'
    if backend.lambda_oc==0:experiment['cost_metric']=experiment['cost_metric'].replace('+0.1E','')
    elif backend.lambda_oc!=.1:experiment['cost_metric']=experiment['cost_metric'].replace('0.1E',f'{backend.lambda_oc:g}E')
    if settings.get('rollout_wall_cost_weight',0):
        experiment['cost_metric']+=f'+{settings["rollout_wall_cost_weight"]:g}*rollout_wall_contact_steps'
    if a.prefer_far_from_center:experiment['noise_strategy']='farthest from initial center among exact-prefix reachable stratified samples; previous-sample distance breaks ties'
    # JSON is strict; an omitted radius cap is represented by null.
    if not np.isfinite(experiment['noise_max_radius']):experiment['noise_max_radius']=None
    if a.resume:
        stored=json.loads((out/'config.json').read_text())
        import copy
        comparison=copy.deepcopy(experiment);stored_comparison=copy.deepcopy(stored)
        for key in ('mcmc_start','temperature','walk_sigma'):
            comparison.pop(key,None);stored_comparison['experiment'].pop(key,None)
        if not resume_settings_match(stored_comparison,settings,comparison):raise ValueError('Resume settings differ beyond the requested MCMC phase')
    write_json(out/'config.json',dict(settings=settings,experiment=experiment));write_json(out/'spec.json',backend.spec());write_json(out/'wall_geometry.json',backend.wall_geometry)
    n=len(backend.grid)-1;s=tail_start(backend);levels=np.arange(s,n+1);history=[]
    for name in ('original','hardflow'):
        with np.load(source/f'{name}.npz') as data:
            run={k:data[k].copy() for k in ('X','U','refs')};noise=data['noise'].copy()
        save_run(out/f'{name}.npz',backend,run,noise)
        if name==a.initial_reference:
            if not backend.final_check(run['X'][-1])['safe']:raise ValueError('Initial reference is not explicitly feasible')
            history.append(run['X'][s:].copy());center=backend.free_noise(noise).astype(float)
            best=metrics(backend,run,run,0.)['J']
            reference_control_norms=np.linalg.norm(run['U'],axis=1)
    best_noise=center.copy();best_noise_round=0;imported_rounds=[];imported_reference=None
    for r in range(1,a.source_rounds+1):
        m=json.loads((source/f'round_{r:02d}_metrics.json').read_text())
        if not m['safe']:continue
        if a.require_improvement and not cost_improves(True,m['J'],best,a.improvement_tolerance):continue
        with np.load(source/f'round_{r:02d}.npz') as data:chain=data['X'].copy()
        if not backend.terminal(chain[-1])[0]:raise ValueError('Non-hard-feasible source record')
        if a.control_norm_ratio is not None:
            with np.load(source/f'round_{r:02d}.npz') as data:imported_norms=np.linalg.norm(data['U'],axis=1)
            if np.any(imported_norms[s:]>a.control_norm_ratio*reference_control_norms[s:]+settings['tolerances']['feasibility']):continue
        history.append(chain[s:])
        imported_rounds.append(r);imported_reference=source/f'round_{r:02d}.npz'
        if a.require_improvement:best_noise=chain[0].copy();best_noise_round=r
        if a.control_norm_ratio is not None:
            with np.load(source/f'round_{r:02d}.npz') as data:reference_control_norms=np.linalg.norm(data['U'],axis=1)
        if a.require_improvement:best=m['J']
    if imported_reference is not None and a.require_improvement:
        with np.load(imported_reference) as data:
            initial_run={key:data[key].copy() for key in ('X','U','refs')};initial_noise=data['noise'].copy()
        save_run(out/'initial_history_reference.npz',backend,initial_run,initial_noise)
        initial_metrics=metrics(backend,initial_run,initial_run,0.,rollout_path=out/'initial_history_reference_rollout.npz')
        np.testing.assert_allclose(initial_metrics['J'],best,rtol=0,atol=1e-8)
        initial_metrics.update(imported_round=imported_rounds[-1],imported_rounds=imported_rounds)
        write_json(out/'initial_history_reference_metrics.json',initial_metrics)
    points=np.asarray(history);eps=a.epsilon_scale*np.asarray(json.loads((source/'epsilon.json').read_text())['epsilon'])
    centers,rho,radii,certificates=fit_history_balls(points,eps)
    np.savez_compressed(out/'known_balls.npz',levels=levels,centers=centers,rho=rho,epsilon=eps,effective_radius=radii,history=points,noise_center=center)
    write_json(out/'known_balls.json',dict(source=str(source),source_rounds=a.source_rounds,records=len(points),levels=levels.tolist(),effective_radius=radii.tolist(),certificates=certificates,frozen_during_test=not a.update_balls,initial_best_cost=best,epsilon_scale=a.epsilon_scale))
    previous=[center.copy()];radius=a.noise_radius;failed_streak=0;reports={};accepted=[];times=[];begun=time.perf_counter();completed=0
    if a.resume:
        for r in range(1,a.rounds+1):
            file=out/f'round_{r:02d}_metrics.json'
            if not file.exists():break
            m=json.loads(file.read_text())
            with np.load(out/f'round_{r:02d}.npz') as data:
                previous.append(data['X'][0].copy());chain=data['X'][s:].copy()
            if m['safe']:
                if m['J']<best-a.improvement_tolerance:
                    best_noise=previous[-1].copy();best_noise_round=r
                accepted.append(chain)
                if a.update_balls:history.append(chain)
                if a.control_norm_ratio is not None:
                    with np.load(out/f'round_{r:02d}.npz') as data:reference_control_norms=np.linalg.norm(data['U'],axis=1)
            best=m['best_cost_after'];failed_streak=m['noise_sample']['consecutive_failures']
            if m['noise_sample']['expanded_after_attempt']:failed_streak=0
            radius=m['noise_sample'].get('next_radius',m['noise_sample']['radius'])
            times.append(m['round_wall_seconds']);reports[f'round_{r}']=compact_report(m);completed=r
    tolerance=settings['tolerances']['feasibility']
    if a.mcmc_start<1 or a.temperature<=0 or a.walk_sigma<=0:raise ValueError('Invalid hybrid MCMC settings')
    # Reconstruct the state at the phase boundary from committed records.
    chain_noise=center.copy();chain_cost=json.loads((source/'hardflow_metrics.json').read_text())['J']
    for rr in range(1,completed+1):
        mm=reports[f'round_{rr}']
        if (rr<a.mcmc_start and mm['safe'] and mm['J']<chain_cost-a.improvement_tolerance) or (rr>=a.mcmc_start and mm.get('chain_accepted')):
            chain_noise=previous[rr].copy();chain_cost=mm.get('chain_cost_after',mm['J'])
    for r in range(completed+1,a.rounds+1):
        start=time.perf_counter();rng=np.random.default_rng(np.random.SeedSequence([a.seed,r]))
        previous_control_norms=reference_control_norms.copy()
        backend.control_norm_bounds=None if a.control_norm_ratio is None else a.control_norm_ratio*previous_control_norms
        if a.update_balls:points=np.asarray(history);centers,rho,radii,certificates=fit_history_balls(points,eps)
        np.savez_compressed(out/f'balls_{r:02d}.npz',centers=centers,rho=rho,epsilon=eps,effective_radius=radii,history=points)
        comparison=previous[1:] if a.spread_between_samples and len(previous)>1 else previous
        screened_prefix=None
        policy_file=out/'search_policy.json'
        policy=json.loads(policy_file.read_text()) if policy_file.exists() else None
        policy_rng=np.random.default_rng(np.random.SeedSequence([a.seed,r,991]))
        search_center,search_radius,branch=search_region(center,best_noise,radius,policy_rng,policy,r)
        mcmc=r>=a.mcmc_start
        if mcmc:
            initial=chain_noise+a.walk_sigma*rng.normal(size=len(chain_noise));policy=None
            noise_detail=dict(strategy='chain-centered symmetric Gaussian MH',radius=radius,walk_sigma=a.walk_sigma,temperature=a.temperature,pool_size=1,center_distance=float(np.linalg.norm(initial-center)))
        elif a.entry_screen:initial,noise_detail,screened_prefix=reachable_noise(search_center,search_radius,comparison,rng,backend,centers[0],radii[0],a.pool_size,prefer_far_from_center=a.prefer_far_from_center and branch!='local_best',domain_center=center if branch=='local_best' else None,domain_radius=radius)
        else:initial,noise_detail=diverse_noise(center,radius,comparison,rng,a.pool_size)
        if policy:
            if not a.entry_screen:raise ValueError('Mixture search requires exact-prefix entry screening')
            noise_detail.update(search_branch=branch,search_policy=policy,sampling_radius=search_radius,
                exploit_probability_used=exploitation_probability(policy,r),
                radius=radius,sampling_center_round=best_noise_round if branch=='local_best' else 0,
                global_center_distance=float(np.linalg.norm(initial-center)),best_region_cost=best)
        previous.append(initial.copy())
        noise=backend.coordinates.full(initial)[None]
        X=np.full((n+1,len(initial)),np.nan);U=np.full((n,len(initial)),np.nan);refs=U.copy();X[0]=initial
        if screened_prefix is not None:X[:s+1]=screened_prefix;refs[:s]=screened_prefix[1:];U[:s]=0
        else:
            for i in range(s):refs[i]=backend.natural(i,X[i]);X[i+1]=refs[i];U[i]=0
        prefix=X[:s+1].copy();steps=[];predictions=[]
        entry_excess=float(np.linalg.norm(X[s]-centers[0])-radii[0]);failure=None
        if mcmc and np.linalg.norm(initial-center)>a.noise_max_radius:failure='outside_global_noise_ball'
        elif entry_excess>tolerance:failure=f'x{s}_outside_known_ball'
        else:
            for i in range(s,n):
                solution,detail=backend.solve_step(i,X[i],centers[i+1-s],radii[i+1-s],points[:,i+1-s],rng,a.max_iter)
                steps.append(dict(step=i,**detail))
                if solution is None:failure=f'step_{i}: '+detail['reason'];break
                X[i+1],U[i],refs[i],yy=solution;predictions.append(yy)
                np.testing.assert_allclose(X[i+1],refs[i]+backend.grid[i+1]*(yy-(refs[i]+(1-backend.grid[i+1])*backend.velocity(refs[i],backend.grid[i+1]))),rtol=0,atol=1e-10)
        np.testing.assert_array_equal(X[:s+1],prefix)
        run=dict(X=X,U=U,refs=refs)
        if failure:m=dict(safe=False,failure=failure,J=None,C=None,E=None,strict_safe=False)
        else:
            original=sample(NaturalFlow(backend),noise,backend.spec())
            save_run(out/f'original_round_{r:02d}.npz',backend,original,noise)
            m=metrics(backend,run,original,time.perf_counter()-start,rollout_path=out/f'round_{r:02d}_rollout.npz')
            m['failure']=None if m['safe'] else 'final_explicit_constraints'
        if settings.get('rollout_wall_cost_weight',0):
            if failure:m.update(rollout_evaluated=False,rollout_skipped_reason='incomplete candidate')
            else:print(f'Rollout {r}: wall_contacts={m["rollout"]["wall_contact_steps"]}, reached={m["rollout"]["target_reached"]}, J_plan={m["J_plan"]:.6f}, J_eval={m["J"]:.6f}',flush=True)
        candidate_control_norms=np.linalg.norm(U,axis=1)
        control_excess=None
        if a.control_norm_ratio is not None and not failure:
            control_excess=float(np.max(candidate_control_norms[s:]-backend.control_norm_bounds[s:]))
            if control_excess>tolerance:m.update(safe=False,failure='control_norm_bound')
        feasible=bool(m['safe']);best_before=best
        if mcmc:
            from .history_ball_mcmc import log_acceptance
            chain_before=chain_cost;accepted_chain=False;logalpha=None
            if feasible:
                logalpha=log_acceptance(chain_cost,m['J'],chain_noise,initial,best_noise,a.temperature,0.,a.walk_sigma,a.walk_sigma)
                accepted_chain=bool(np.log(max(rng.random(),np.finfo(float).tiny))<logalpha)
                if accepted_chain:chain_noise=initial.copy();chain_cost=m['J']
            m.update(phase='mcmc',chain_accepted=accepted_chain,chain_cost_before=chain_before,chain_cost_after=chain_cost,log_acceptance=logalpha,accepted_uphill=bool(accepted_chain and m['J']>chain_before))
        else:
            m['phase']='best_mixture'
            if feasible and (not a.require_improvement or m['J']<best-a.improvement_tolerance):chain_noise=initial.copy();chain_cost=m['J']
        if feasible and a.require_improvement and not cost_improves(True,m['J'],best,a.improvement_tolerance):
            m.update(safe=False,failure='cost_not_lower_than_best')
        # An accepted record must satisfy every tail ball, including the entry state.
        excess=float(np.max(np.linalg.norm(X[s:]-centers,axis=1)-radii)) if np.isfinite(X[s:]).all() else max(0.,entry_excess)
        if m['safe'] and excess>tolerance:raise AssertionError('Accepted candidate escaped known balls')
        if m['safe']:
            best_noise=initial.copy();best_noise_round=r
            accepted.append(X[s:].copy())
            if a.update_balls:history.append(X[s:].copy())
            best=min(best,m['J'])
            if a.control_norm_ratio is not None:reference_control_norms=candidate_control_norms.copy()
        failed_streak=0 if m['safe'] else failed_streak+1
        noise_detail.update(consecutive_failures=failed_streak,expanded_after_attempt=False)
        if failed_streak>=a.expand_after:
            old_radius=radius;radius=min(radius*a.growth,a.noise_max_radius);failed_streak=0
            noise_detail.update(expanded_after_attempt=True,next_radius=radius,expansion_capped=radius==old_radius)
        m.update(round=r,steps=steps,noise_sample=noise_detail,entry_level=s,entry_ball_excess=entry_excess,**({'x5_ball_excess':entry_excess} if s==5 else {}),ball_max_violation=max(0.,excess),prefix_frozen_per_noise=True,duplicate=False,seconds=time.perf_counter()-start,
            constraint_feasible=feasible,best_cost_before=best_before,best_cost_after=best,history_size_after=len(history))
        if a.control_norm_ratio is not None:
            m.update(control_norm_ratio=a.control_norm_ratio,control_reference_norms=previous_control_norms.tolist(),
                control_norm_bounds=backend.control_norm_bounds.tolist(),
                candidate_control_norms=[float(v) if np.isfinite(v) else None for v in candidate_control_norms],
                control_bound_max_violation=None if control_excess is None else max(0.,control_excess),
                control_reference_after=reference_control_norms.tolist())
        save_run(out/f'round_{r:02d}.npz',backend,run,noise,predictions)
        m['round_wall_seconds']=time.perf_counter()-start;reports[f'round_{r}']=compact_report(m)
        write_json(out/f'round_{r:02d}_metrics.tmp.json',m);(out/f'round_{r:02d}_metrics.tmp.json').replace(out/f'round_{r:02d}_metrics.json')
        times.append(m['round_wall_seconds']);remaining=(a.rounds-r)*float(np.mean(times[-20:]))
        progress=dict(status='running',rounds_completed=r,rounds_requested=a.rounds,admitted=len(accepted),failures=r-len(accepted),constraint_feasible=sum(m.get('constraint_feasible',m['safe']) for m in reports.values()),cost_rejected=sum(m['failure']=='cost_not_lower_than_best' for m in reports.values()),entry_rejected=sum(m['failure']==f'x{s}_outside_known_ball' for m in reports.values()),best_cost=best,noise_radius_next=radius,history_size=len(history),estimated_remaining_seconds=remaining,elapsed_seconds=time.perf_counter()-begun,
            baseline_cost=json.loads((out/'known_balls.json').read_text())['initial_best_cost'],recent_mean_round_seconds=float(np.mean(times[-20:])))
        progress['updated_unix_seconds']=time.time()
        progress.update(sampling_phase='mcmc' if mcmc else 'best_mixture',mcmc_start=a.mcmc_start,chain_cost=chain_cost,
            mcmc_chain_accepted=sum(bool(v.get('chain_accepted')) for v in reports.values()))
        write_json(out/'progress.tmp.json',progress);(out/'progress.tmp.json').replace(out/'progress.json')
        write_json(out/'cost_history.tmp.json',[dict(round=value['round'],J=value['J'],best=value['best_cost_after'],admitted=value['safe']) for value in reports.values()])
        (out/'cost_history.tmp.json').replace(out/'cost_history.json')
        print(f'Round {r}/{a.rounds}: admitted={m["safe"]}, failure={m["failure"]}, noise_radius={noise_detail["radius"]}, next_radius={radius}, seconds={m["seconds"]:.2f}, ETA_minutes={remaining/60:.1f}',flush=True)
    np.savez_compressed(out/'sampled_noise.npz',samples=np.asarray(previous[1:]),center=center)
    np.savez_compressed(out/'accepted_history.npz',X=np.asarray(accepted),levels=levels)
    np.savez_compressed(out/'history.npz',X=np.asarray(history),levels=levels,epsilon=eps)
    write_json(out/'summary.json',dict(status='complete',attempts_completed=a.rounds,safe_success=len(accepted),failures=a.rounds-len(accepted),metrics=reports,constraint_mode='hard',variable_initial_noise=True,known_balls_frozen=not a.update_balls,best_cost=best,admission='Original explicit hard constraints and endpoints'+('; strictly decreasing best J required' if a.require_improvement else '')+'; geometric collisions separately recorded'))
    progress.update(status='complete',estimated_remaining_seconds=0.);write_json(out/'progress.json',progress)


if __name__=='__main__':main()
