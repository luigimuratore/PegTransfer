"""Check spring bounds, arm transfer, damping and removal on the requested device."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
from peg_support import PegSupport


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required, no fallback')
    path=Path(__file__).resolve().parents[2]/'third_party/IsaacLab/source/isaaclab/isaaclab/utils/math.py'
    spec=importlib.util.spec_from_file_location('support_sdk_math',path)
    sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk)
    def t(x):return torch.tensor(x,device=device,dtype=torch.float32)
    class Object:
        def __init__(self):
            self.data=SimpleNamespace(root_pos_w=t([[0,0,.05]]),root_quat_w=t([[1,0,0,0]]),
                                      root_lin_vel_w=t([[0,0,0]]),root_ang_vel_w=t([[0,0,0]]))
            self.writes=0;self.wrenches=[]
        def write_data_to_sim(self):self.writes+=1
        def set_external_force_and_torque(self,force,torque,**kwargs):
            assert kwargs['is_global'];self.wrenches.append((force.clone(),torque.clone()))
    obj=Object();frame=SimpleNamespace(target_pos_w=t([[[0,0,.06]]]),target_quat_w=t([[[1,0,0,0]]]))
    frame2=SimpleNamespace(target_pos_w=t([[[.01,0,.06]]]),target_quat_w=t([[[1,0,0,0]]]))
    env=SimpleNamespace(device=device,physics_dt=.0025,scene={'object':obj,
                         'ee_1_frame':SimpleNamespace(data=frame),'ee_2_frame':SimpleNamespace(data=frame2)})
    with patch.dict(sys.modules,{'isaaclab.utils.math':sdk}):
        support=PegSupport(env,.5);obj.write_data_to_sim();assert obj.writes==1 and support.steps==0
        support.acquire(1);obj.write_data_to_sim();assert float(support.force.norm())==0.
        obj.data.root_pos_w[0,0]-=.02
        obj.data.root_quat_w[:]=t([[2**-.5,0,2**-.5,0]])
        before=obj.data.root_pos_w.clone();beforeq=obj.data.root_quat_w.clone()
        obj.write_data_to_sim();assert 0<float(support.force.norm())<=.075001
        assert 0<float(support.torque.norm())<=.001001
        assert support.force[0,0,0]>0 and support.torque[0,0,1]<0
        torch.testing.assert_close(obj.data.root_pos_w,before);torch.testing.assert_close(obj.data.root_quat_w,beforeq)
        support.acquire(2);obj.write_data_to_sim();assert float(support.force.norm())<1e-6
        obj.data.root_lin_vel_w[0,0]=.02;obj.write_data_to_sim();assert support.force[0,0,0]<0
        support.release();obj.write_data_to_sim();assert not support.holder
        assert float(obj.wrenches[-1][0].norm())==float(obj.wrenches[-1][1].norm())==0.
        zero=PegSupport(env,0.);zero.acquire(1);assert not zero.holder
    print(json.dumps(dict(device=str(device),gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                         checks=['inactive before acquisition','force/torque limits','restoring sign','no pose writes',
                                 'continuous anchor transfer','velocity damping','zero force after release','zero support disabled'],
                         evidence=support.evidence()),indent=2))


if __name__=='__main__':main()
