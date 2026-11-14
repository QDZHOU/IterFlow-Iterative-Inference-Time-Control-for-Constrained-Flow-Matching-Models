"""Native execution used by outer-loop admission, not by the surrogate NLP."""
import json
from pathlib import Path
import numpy as np


def evaluate_wall_contacts(plan, start, goal, seed=0, max_steps=800, env_factory=None):
    from .contact_recorder import ContactRecorder
    if env_factory is None:
        from hardflow.envs.maze2d import MazeEnv
        env_factory=MazeEnv
    plan=np.asarray(plan,float);start=np.asarray(start,float);goal=np.asarray(goal,float)
    if plan.ndim!=2 or plan.shape[1]!=4 or len(plan)<2 or not np.isfinite(plan).all():
        raise ValueError('Rollout requires a complete finite physical state plan')
    np.testing.assert_allclose(plan[0],start,rtol=0,atol=2e-6)
    env=env_factory()
    try:
        if not 1<=max_steps<=env.max_episode_steps:raise ValueError('Invalid rollout step limit')
        env._target=goal.copy()
        def reset():
            env.seed(seed);env.reset();env.set_state(start[:2],start[2:])
        reset();recorder=ContactRecorder(env);observations=[env._get_obs().copy()];actions=[]
        for k in range(max_steps):
            obs=observations[-1]
            if np.linalg.norm(obs[:2]-goal)<=.5:break
            desired=plan[min(k,len(plan)-1)]
            action=np.clip(5*(desired[:2]-obs[:2])+desired[2:]-obs[2:],-1,1)
            nxt,_,done,_=recorder.step(action)
            observations.append(nxt.copy());actions.append(action.copy())
            if done:break
        observations=np.asarray(observations);actions=np.asarray(actions).reshape(-1,2)
        assert np.isfinite(observations).all() and np.isfinite(actions).all()
        reset();replay=[env._get_obs().copy()]
        for action in actions:replay.append(env.step(action)[0].copy())
        np.testing.assert_array_equal(replay,observations)
        contacts=np.asarray([bool(c) for c in recorder.contacts]);distance=float(np.linalg.norm(observations[-1,:2]-goal))
        report=dict(steps=len(actions),wall_contact_steps=int(contacts.sum()),
            wall_contact_pairs=sum(len(c) for c in recorder.contacts),
            wall_max_penetration=float(max(recorder.penetrations,default=0.)),
            target_reached=distance<=.5,goal_distance=distance,
            final_observation=observations[-1].tolist(),action_replay_exact=True,
            seed=seed,max_steps=max_steps,controller='native PD Kp=5 Kd=1, actions clipped [-1,1]',
            counting='one per env.step with at least one native particle-wall contact; ground excluded')
        data=dict(observations=observations,actions=actions,planned_observations=plan,
                  wall_contact_steps=contacts,wall_penetrations=np.asarray(recorder.penetrations))
        return data,report
    finally:env.close()


def apply_rollout_cost(result,backend,run,output=None):
    weight=float(backend.settings.get('rollout_wall_cost_weight',0.))
    if weight==0 and not backend.settings.get('evaluate_rollout',False):return result
    from .compare import physical_observations
    data,report=evaluate_wall_contacts(physical_observations(backend,run['X'][-1]),
        backend.settings['observation'],backend.settings['goal'],
        seed=backend.settings.get('rollout_seed',0),max_steps=backend.settings.get('rollout_max_steps',800))
    penalty=weight*report['wall_contact_steps']
    result.update(J_plan=result['J'],rollout_evaluated=True,rollout_wall_cost=penalty,rollout_wall_cost_weight=weight,
                  rollout=report,J=result['J']+penalty)
    if output is not None:
        output=Path(output);np.savez_compressed(output,**data)
        output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        result['rollout_file']=str(output)
    return result
