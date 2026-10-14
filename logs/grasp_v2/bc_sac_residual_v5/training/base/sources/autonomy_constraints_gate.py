"""Gates for explicit grasp/placement constraints; zero support is native only."""
from autonomy_full_recovery import native_full_gate
from autonomy_support import SUPPORT_PROFILES


def assisted_gate(report, strength, episodes=16, threshold=.9, profile='firm', placement=True):
    if strength == 0:
        return native_full_gate(report, episodes, threshold) and report.get('programmed_task_controller') is False
    settings = SUPPORT_PROFILES[profile]
    config = report.get('grasp_support', {})
    rows = report.get('cases', [])
    if (report.get('complete') is not True or report.get('skill') != 'full'
        or report.get('episodes') != episodes or len(rows) != episodes
        or report.get('teacher_at_inference') is not False or report.get('programmed_task_controller') is not False
        or report.get('assisted_grasp') is not True or config.get('strength') != strength
        or config.get('profile') != profile or config.get('placement_guide_enabled') is not placement
        or config.get('force_limit_n') != settings['force_limit_n']*strength
        or config.get('torque_limit_nm') != settings['torque_limit_nm']*strength):
        return False
    successes = [r for r in rows if r.get('success') and r.get('full_success') and r.get('handover')]
    if len(successes) != report.get('success') or len(successes) != report.get('full_success') or len(successes)/episodes < threshold:
        return False
    for row in successes:
        physics = row.get('terminal_physics', {})
        support = row.get('grasp_support', {})
        if (row.get('assisted_grasp') is not True or row.get('failure') or row.get('timeout')
            or row.get('peak_force_n', float('inf')) > 1 or row.get('hole_error_mm', float('inf')) > 1.5
            or not .008 <= physics.get('bottom_m', float('inf')) <= .012
            or support.get('strength') != strength or support.get('profile') != profile
            or support.get('applied_physics_steps', 0) < 1 or support.get('holder') != 0
            or support.get('placement_guide_enabled') is not placement or support.get('placement_guide_active') is not False
            or support.get('force_peak_n', float('inf')) > settings['force_limit_n']*strength+1e-6
            or support.get('torque_peak_nm', float('inf')) > settings['torque_limit_nm']*strength+1e-6
            or support.get('placement_force_peak_n', float('inf')) > .15*strength+1e-6
            or support.get('placement_torque_peak_nm', float('inf')) > .002*strength+1e-6
            or support.get('object_pose_writes') is not False or support.get('object_velocity_writes') is not False):
            return False
        forces = physics.get('forces_n', [])
        if len(forces) != 2 or any(len(pair) != 2 or any(abs(f) > .01 for f in pair) for pair in forces):
            return False
    return True
