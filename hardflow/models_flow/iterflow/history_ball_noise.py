"""Initial-noise proposals, exact-prefix screening and history-ball fitting."""
import numpy as np
from .history_ball_surrogate import minimum_ball
from .core import canonical
from .denoise_schedule import tail_start

def diverse_noise(center,radius,previous,rng,pool_size=64):
    """Maximin selection from uniform-volume draws; never move outside the ball."""
    directions=rng.normal(size=(pool_size,len(center)))
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    candidates=center+radius*rng.random(pool_size)[:,None]**(1/len(center))*directions
    nearest=np.full(pool_size,np.inf)
    for old in previous:nearest=np.minimum(nearest,np.linalg.norm(candidates-old,axis=1))
    index=int(np.argmax(nearest))
    return candidates[index],dict(pool_size=pool_size,nearest_previous_distance=float(nearest[index]),
        center_distance=float(np.linalg.norm(candidates[index]-center)),radius=float(radius))


def cost_improves(feasible,value,best,tolerance=1e-8):
    return bool(feasible and value is not None and np.isfinite(value) and value<best-tolerance)


def fit_history_balls(points,epsilon):
    fits=[minimum_ball(points[:,j]) for j in range(points.shape[1])]
    centers=np.asarray([item[0] for item in fits]);rho=np.asarray([item[1] for item in fits])
    return centers,rho,rho+epsilon,[item[2] for item in fits]


def reachable_noise(center,radius,previous,rng,backend,entry_center,entry_radius,pool_size=256,bins=8,probes_per_bin=2,prefer_far_from_center=False,domain_center=None,domain_radius=None):
    """Stratified radii, maximin ranking and exact natural-prefix entry screening."""
    directions=rng.normal(size=(pool_size,len(center)))
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    strata=np.arange(pool_size)%bins
    radial=radius*(strata+rng.random(pool_size))/bins
    candidates=center+radial[:,None]*directions
    if domain_center is not None:
        keep=np.linalg.norm(candidates-domain_center,axis=1)<=domain_radius
        candidates=candidates[keep];radial=radial[keep];strata=strata[keep]
        if not len(candidates):
            if np.linalg.norm(center-domain_center)>domain_radius+1e-9:raise ValueError('Local center outside global noise ball')
            candidates=np.asarray([center.copy()]);radial=np.zeros(1);strata=np.zeros(1,dtype=int)
    old=np.asarray(previous);nearest2=np.full(len(candidates),np.inf)
    # Gram distances avoid allocating (pool,history,dimension) or a Python loop
    # over thousands of full-dimensional records.
    norms=np.sum(candidates*candidates,axis=1)
    for offset in range(0,len(old),256):
        chunk=old[offset:offset+256]
        distances=norms[:,None]+np.sum(chunk*chunk,axis=1)[None,:]-2*candidates@chunk.T
        nearest2=np.minimum(nearest2,np.maximum(distances,0).min(axis=1))
    indices=[]
    for bucket in range(bins):
        members=np.flatnonzero(strata==bucket)
        if not len(members):continue
        ranking=radial if prefer_far_from_center else nearest2
        selected=members[np.argsort(ranking[members])[::-1]][:probes_per_bin].tolist()
        # Also test the inner-most point of each stratum: a thin reachable
        # interval must not disappear just because its outer points rank higher.
        selected.append(int(members[np.argmin(radial[members])]))
        indices.extend(dict.fromkeys(selected))
    s=tail_start(backend);tested=[];valid=[]
    for index in indices:
        prefix=[candidates[index].copy()]
        for i in range(s):prefix.append(backend.natural(i,prefix[-1]))
        excess=float(np.linalg.norm(prefix[-1]-entry_center)-entry_radius)
        tested.append(dict(index=index,radial_distance=float(radial[index]),nearest_previous_distance=float(np.sqrt(nearest2[index])),entry_excess=excess))
        if np.isfinite(prefix[-1]).all() and excess<=backend.settings['tolerances']['feasibility']:
            valid.append((nearest2[index],index,np.asarray(prefix)))
    chosen=max(valid,key=lambda item:(radial[item[1]],item[0]) if prefer_far_from_center else (item[0],)) if valid else None
    if chosen is None:
        index=min(tested,key=lambda item:item['entry_excess'])['index']
        prefix=[candidates[index].copy()]
        for i in range(s):prefix.append(backend.natural(i,prefix[-1]))
        prefix=np.asarray(prefix)
    else:_,index,prefix=chosen
    return candidates[index],dict(pool_size=pool_size,nearest_previous_distance=float(np.sqrt(nearest2[index])),
        center_distance=float(radial[index]),radius=float(radius),strategy='stratified_radius_farthest_center_entry_screen' if prefer_far_from_center else 'stratified_radius_maximin_entry_screen',
        prefix_probes=tested,entry_reachable=chosen is not None),prefix


def exploitation_probability(policy,round_index=None):
    if 'schedule' in policy:
        schedule=policy['schedule']
        if schedule.get('kind')!='linear' or round_index is None:raise ValueError('Linear schedule requires a round index')
        first=int(schedule['first_round']);last=int(schedule['last_round'])
        start=float(schedule['start_exploit_probability']);end=float(schedule['end_exploit_probability'])
        if first<1 or last<=first or not np.isfinite([start,end]).all() or not 0<=start<=1 or not 0<=end<=1:
            raise ValueError('Invalid probability schedule')
        progress=np.clip((round_index-first)/(last-first),0.,1.)
        return float(start+(end-start)*progress)
    return float(policy.get('exploit_probability',.8))


def search_region(center,best_noise,radius,rng,policy,round_index=None):
    """Choose a local incumbent neighborhood or the unchanged global ball."""
    if not policy:return center,radius,'global'
    if policy.get('mode')!='best_mixture':raise ValueError('Unknown search policy')
    probability=exploitation_probability(policy,round_index)
    fraction=float(policy.get('local_radius_fraction',.25))
    cap=float(policy.get('local_radius_max',5.))
    if not 0<=probability<=1 or not np.isfinite([probability,fraction,cap]).all() or fraction<=0 or cap<=0:
        raise ValueError('Invalid mixture search settings')
    if rng.random()<probability:return best_noise.copy(),min(fraction*radius,cap),'local_best'
    return center,radius,'global_exploration'


def compact_report(report):
    return {key:value for key,value in report.items() if key not in ('steps','wall_segment_hits','noise_sample')}


def resume_settings_match(stored,settings,experiment):
    old=stored['experiment'].copy();new=experiment.copy()
    old.pop('resume',None);new.pop('resume',None)
    old.setdefault('prefer_far_from_center',False)
    old.setdefault('control_norm_ratio',None);new.setdefault('control_norm_ratio',None)
    if new['rounds']<old['rounds']:return False
    old['rounds']=new['rounds']
    return old==new and stored['settings']==settings
