"""Short checks on the real CUDA device; never starts Isaac or a full training."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from learning import MotorPolicy,load_demonstrations,dataset_experiment,observation,feedback_action
from torch_backend import resolve_device
from train import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output and args.output.exists():p.error('Validation report already exists')
    device=resolve_device(args.device)
    if device.type!='cuda':p.error('This check requires an actual CUDA device')
    repo=Path(__file__).resolve().parents[2];base=repo/'logs/grasp_v2'
    dataset=base/'wall_demos_bc1.npz';prior=base/'wall_bc2/policy.npz'
    data,meta=load_demonstrations(dataset);experiment=dataset_experiment(meta,prior)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    with tempfile.TemporaryDirectory(prefix='peg_wall_cuda_check_') as directory:
        output=Path(directory)/'model'
        manifest=train(dataset,output,updates=100,batch_size=1024,initialize_from=prior,device=str(device))
        numpy_policy=MotorPolicy(output/'policy.npz')
        gpu_policy=MotorPolicy(output/'policy.npz',experiment,device=str(device))
        indices=np.flatnonzero(data['episode']==meta['episodes'][-1])
        held=data['observations'][indices]
        reference=np.stack([numpy_policy.predict(obs) for obs in held])
        prediction=gpu_policy.torch_policy.predict_tensor(held)
        assert prediction.device==device
        actual=prediction.cpu().numpy()
        np.testing.assert_allclose(actual,reference,atol=5e-4,rtol=5e-4)
        numpy_mse=float(np.mean((reference-data['actions'][indices])**2))
        assert abs(numpy_mse-manifest['validation']['normalized_action_mse'])<2e-5
        trace=base/'wall_psm1_2026-10-05_16-11-03-991776.trace.jsonl'
        rows=[]
        for line in trace.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
        weak=next(row for row in reversed(rows) if row['stage']=='close' and
                  min(row['finger_normal_force_n'])<.30 and row['jaw_target_rad']>0.)
        obs=observation(weak,'close',weak['phase_target_w_m'],weak['tool_quaternion_wxyz'],
                        weak['command_position_w_m'],weak['jaw_target_rad'])
        recovered=gpu_policy.predict(obs)
        assert recovered[3]<0.,'Weak pressure must keep closing instead of stopping'
        report=dict(device=str(device),gpu_name=torch.cuda.get_device_name(device),cuda=torch.version.cuda,
            checks_passed=['real-data training and sampling on CUDA','checkpoint export/reload',
                           'CUDA/NumPy action parity','CUDA inference closes on recorded weak pressure'],
            updates=100,batch_size=1024,seconds=time.monotonic()-started,
            peak_torch_memory_mb=torch.cuda.max_memory_allocated(device)/1024**2,
            validation_mse=manifest['validation']['normalized_action_mse'],numpy_validation_mse=numpy_mse,
            max_action_difference=float(np.max(np.abs(reference-actual))),
            original_weak_pressure_n=weak['finger_normal_force_n'],new_jaw_action=float(recovered[3]),
            experiment=experiment,source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
            production_checkpoint_created=False,isaac_sim_launched=False,neural_physical_success_verified=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as file:json.dump(report,file,indent=2);file.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
