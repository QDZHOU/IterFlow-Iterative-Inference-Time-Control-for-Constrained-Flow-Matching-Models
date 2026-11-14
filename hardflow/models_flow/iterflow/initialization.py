"""Prepare a same-noise original/HardFlow history seed, without IterFlow attempts."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from .history_ball_surrogate import HistoryMaze, metrics, save_run
from .compare import NaturalFlow
from .core import sample
from .denoise_schedule import tail_start
from .io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    settings=json.loads(a.config.read_text())
    write_json(out/'config.json',dict(settings=settings,experiment=dict(seed=a.seed,baseline_only=True,initial_reference='hardflow')))
    b=HistoryMaze(settings);b.torch.set_num_threads(int(os.environ.get('HARDFLOW_TORCH_THREADS','4')));b.admission_mode='original_explicit';b.wall_tolerance=.06
    try:
        write_json(out/'spec.json',b.spec());write_json(out/'wall_geometry.json',b.wall_geometry)
        noise=b.noise(a.seed);noise.flags.writeable=False;runs={};reports={}
        for name,policy in [('original',NaturalFlow(b)),('hardflow',b)]:
            started=time.perf_counter();run=sample(policy,noise,b.spec());runs[name]=run
            m=metrics(b,run,runs['original'],time.perf_counter()-started,rollout_path=out/f'{name}_rollout.npz')
            m['native_solver_errors']=run['report']['errors']
            save_run(out/f'{name}.npz',b,run,noise);write_json(out/f'{name}_metrics.json',m);reports[name]=m
            print(f'{name}: J={m["J"]:.8f}, E={m["E"]:.8f}, explicit={m["explicit_feasible"]}',flush=True)
        repeated=sample(b,noise,b.spec())
        difference=float(np.max(np.abs(repeated['X']-runs['hardflow']['X'])))
        np.testing.assert_allclose(repeated['X'],runs['hardflow']['X'],rtol=0,atol=2e-5)
        s=tail_start(b);d=b.coordinates.free_dim
        epsilon=.1*np.maximum(np.linalg.norm(np.diff(runs['hardflow']['X'],axis=0)[s-1:],axis=1),1e-3*np.sqrt(d))
        write_json(out/'epsilon.json',dict(levels=list(range(s,len(b.grid))),epsilon=epsilon.tolist(),eta=.1,
            lambda_oc=b.lambda_oc,reference='hardflow',formula='eta*max(norm(P(x_reference[j]-x_reference[j-1])),1e-3*sqrt(d_free))',override=None))
        write_json(out/'validation.json',dict(paired_initial_noise=True,same_tensor_baseline_reproduction_max_error=difference))
        write_json(out/'summary.json',dict(status='baseline_complete' if reports['hardflow']['safe'] else 'blocked',baseline=reports,seed=a.seed,lmpc_attempts=0))
        if not reports['hardflow']['safe']:raise RuntimeError('HardFlow baseline fails explicit constraints; no history search started')
    finally:b.env.close()


if __name__=='__main__':main()
