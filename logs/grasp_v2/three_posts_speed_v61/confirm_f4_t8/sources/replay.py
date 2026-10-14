"""Replay a frozen grasp checkpoint with the front view; optionally record one episode."""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[2]
FRONT_EYE=(.025,.32,.25)
FRONT_TARGET=(.025,0.,.035)


def checkpoint_sources(checkpoint):
    with np.load(checkpoint,allow_pickle=False) as file:
        metadata=json.loads(str(file['metadata']))
    folder=checkpoint.parent/'sources'
    names=('geometry.py','runtime.py','motion.py','grip.py','learning.py','probe.py','peg_profile.json','robot_kinematics.json')
    if metadata['schema']=='peg-wall-motor-bc-3':names+=('torch_backend.py',)
    digest=hashlib.sha256(b''.join((folder/name).read_bytes() for name in names)).hexdigest()
    if digest!=metadata['experiment']['code_sha256']:
        raise ValueError('Saved checkpoint sources do not match their experiment hash')
    return folder,metadata


class EpisodeVideo:
    """RTX frames at control-time 25 Hz, encoded by NVIDIA NVENC."""
    def __init__(self,env,path,on_closed=None):
        import omni.replicator.core as rep
        self.env=env;self.path=path;self.frames=0;self.steps=0;self.closed=False
        self.on_closed=on_closed
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('Video already exists')
        ffmpeg=shutil.which('ffmpeg')
        if ffmpeg is None:raise RuntimeError('ffmpeg with h264_nvenc is required')
        self.log_path=path.with_suffix('.encoder.log');self.log=self.log_path.open('x')
        command=[ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                 '-video_size','1280x720','-framerate','25','-i','pipe:0','-an',
                 '-c:v','h264_nvenc','-preset','p4','-cq','22','-b:v','0',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=self.log)
        self.product=rep.create.render_product(env.cfg.viewer.cam_prim_path,(1280,720))
        self.annotator=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu')
        self.annotator.attach([self.product])
        real_step=env.step;real_close=env.close
        def step(action):
            result=real_step(action);self.steps+=1
            if self.steps%4==0:
                env.sim.render()
                frame=self.annotator.get_data()
                if frame.size==0:raise RuntimeError('RTX render product returned an empty video frame')
                self.process.stdin.write(np.ascontiguousarray(frame[:,:,:3]).tobytes())
                self.frames+=1
            return result
        def close():
            try:self.close()
            finally:real_close()
        env.step=step;env.close=close

    def close(self):
        if self.closed:return
        self.closed=True
        self.process.stdin.close();code=self.process.wait(timeout=30);self.log.close()
        self.annotator.detach([self.product.path]);self.product.destroy()
        if code or not self.frames:
            raise RuntimeError(f'Video encoding failed ({code}); see {self.log_path}')
        if self.on_closed:self.on_closed(self)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--video_path',type=Path)
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--episodes',type=int)
    parser.add_argument('--camera_eye',type=float,nargs=3,default=FRONT_EYE)
    parser.add_argument('--camera_target',type=float,nargs=3,default=FRONT_TARGET)
    args,remaining=parser.parse_known_args()
    if args.video and args.episodes not in (None,1):parser.error('Record exactly one episode')
    checkpoint=args.policy
    if args.video and checkpoint is None:checkpoint=REPO/'logs/grasp_v2/wall_bc3/policy.npz'
    if checkpoint:
        checkpoint=checkpoint.resolve();folder,metadata=checkpoint_sources(checkpoint)
    else:
        folder=Path(__file__).parent;metadata=None
    sys.path.insert(0,str(folder))
    probe=importlib.import_module('probe');probe.REPO=REPO
    original_camera=probe.configure_camera
    def camera(env,eye,lookat):
        original_camera(env,eye,lookat)
        if env.viewport_camera_controller is None:env.sim.set_camera_view(eye=eye,target=lookat)
        if args.video:
            for _ in range(4):env.sim.render()
    probe.configure_camera=camera
    passed=remaining+['--camera_eye',*map(str,args.camera_eye),'--camera_target',*map(str,args.camera_target)]
    if checkpoint:passed+=['--policy',str(checkpoint)]
    if args.episodes:passed+=['--episodes',str(args.episodes)]
    if args.video:
        passed+=['--headless','--enable_cameras','--episodes','1']
        if '--mode' not in passed:passed+=['--mode','wall']
    video_path=(args.video_path or REPO/'logs/grasp_v2/videos'/f'wall_bc3_front_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4').resolve()
    recorders=[]
    def save_video_metadata(recorder):
        # Isaac may terminate Python in app.close(), so save before closing it.
        summary=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     original_experiment=metadata['experiment'],frozen_sources=str(folder),
                     camera=dict(eye=args.camera_eye,target=args.camera_target),video=str(video_path),
                     video_sha256=hashlib.sha256(video_path.read_bytes()).hexdigest(),
                     encoder='h264_nvenc',fps=25,frames=recorder.frames,duration_s=recorder.frames/25,
                     resolution=[1280,720])
        with video_path.with_suffix('.json').open('x') as file:json.dump(summary,file,indent=2);file.write('\n')
        print(f'[GRASP VIDEO] {video_path}; {recorder.frames} frames',flush=True)
    if args.video:
        if video_path.exists():parser.error('Video already exists')
        from isaaclab import app as launcher_module
        real_launcher=launcher_module.AppLauncher
        def launch(options):
            launcher=real_launcher(options)
            runtime=importlib.import_module('runtime');make=runtime.make_env
            def make_recording_env(*a,**kw):
                env=make(*a,**kw)
                try:recorders.append(EpisodeVideo(env,video_path,save_video_metadata))
                except Exception:
                    env.close();raise
                return env
            runtime.make_env=make_recording_env
            return launcher
        launcher_module.AppLauncher=launch
    sys.argv=[str(folder/'probe.py'),*passed]
    code=probe.main()
    if args.video and recorders:
        recorder=recorders[0]
        if not recorder.closed:recorder.close()
    return code


if __name__=='__main__':raise SystemExit(main())
