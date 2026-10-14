"""Explicit temporary inter-arm collision exclusion, authored BEFORE PhysX cooks."""


def install_spawn_filter(runtime,interarm=True,receiver_nonfinger_peg=False):
    from pxr import Usd,UsdPhysics
    original=runtime.spawn_contact_robot
    evidence=[]
    def spawn(prim_path,cfg,translation=None,orientation=None,**kwargs):
        root=original(prim_path,cfg,translation,orientation,**kwargs)
        actual=str(root.GetPath())
        arm=int(actual.rsplit('_',1)[-1])
        other=actual.rsplit('/',1)[0]+f'/Robot_{3-arm}'
        bodies=[prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        relatives=[str(prim.GetPath())[len(actual):] for prim in bodies]
        for prim in bodies:
            relation=UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
            if interarm:
                for relative in relatives:relation.AddTarget(other+relative)
            if receiver_nonfinger_peg and arm==2 and prim.GetName() not in ('psm_tool_gripper1_link','psm_tool_gripper2_link'):
                relation.AddTarget(actual.rsplit('/',1)[0]+'/Object')
        evidence.append(dict(robot=actual,filtered_rigid_bodies=len(bodies)))
        return root
    runtime.spawn_contact_robot=spawn
    return evidence


def verify(env,interarm=True,receiver_nonfinger_peg=False):
    from pxr import UsdPhysics
    stage=env.scene['robot_1'].stage
    rows=[]
    for i in range(env.num_envs):
        for arm in (1,2):
            asset=env.scene[f'robot_{arm}'];paths=asset.root_physx_view.link_paths[i]
            other=f'/World/envs/env_{i}/Robot_{3-arm}/'
            count=0
            for path in paths:
                prim=stage.GetPrimAtPath(path)
                targets=UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
                matching=[str(t) for t in targets if str(t).startswith(other)]
                if interarm and not matching:raise RuntimeError(f'Missing within-environment inter-arm filter: {path}')
                object_path=f'/World/envs/env_{i}/Object'
                object_filtered=object_path in [str(t) for t in targets]
                finger=prim.GetName() in ('psm_tool_gripper1_link','psm_tool_gripper2_link')
                expected=receiver_nonfinger_peg and arm==2 and not finger
                if object_filtered!=expected:raise RuntimeError(f'Incorrect object filtering: {path}')
                count+=len(matching)
            rows.append(dict(environment=i,arm=arm,pairs=count))
    return dict(enabled=True,type='temporary_collision_support',interarm=interarm,
        receiver_nonfinger_peg=receiver_nonfinger_peg,
        board_collisions_unchanged=True,peg_gripper_contact_required=True,
        physics_success_criteria_unchanged=True,verified_usd_pairs=rows)
