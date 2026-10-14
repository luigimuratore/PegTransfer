"""Read contact reports for every PSM link; never changes collisions or forces."""
import torch


class LinkContacts:
    def __init__(self,env):
        self.env=env
        sim=env.scene['contact_2_1']._physics_sim_view
        self.views={}
        for arm in (1,2):
            view=sim.create_rigid_contact_view(f'/World/envs/env_*/Robot_{arm}/*',
                filter_patterns=['/World/envs/env_*/Object',f'/World/envs/env_*/Robot_{3-arm}/*'],
                max_contact_data_count=env.num_envs*4096)
            self.views[arm]=view

    def read(self):
        rows={}
        for arm,view in self.views.items():
            matrix=view.get_contact_force_matrix(dt=self.env.physics_dt)
            norm=matrix.norm(dim=-1)
            active=(norm.amax(-1)>.001).nonzero().flatten().tolist()
            rows[str(arm)]=[dict(path=view.sensor_paths[i],force_vectors_n=matrix[i].tolist(),force_n=norm[i].tolist()) for i in active]
        return rows
