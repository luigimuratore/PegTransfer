"""Metre geometry for the SAC task; importable without Isaac Sim.

Bounds include all USD parent transforms and the 0.011 spawn scale.
The original lobed hole is approximated by a smaller, explicit polygonal hole.
"""

import math

import torch

ASSET_SCALE = 0.011
PEG_MIN = (-0.000926772, -0.009282851, -0.014613020)
PEG_MAX = (0.015828740, 0.009283128, 0.002601549)
HOLE_CENTER = (0.00578638, 0.0)
HOLE_RADIUS = 0.0035
HOLE_INRADIUS = HOLE_RADIUS * math.cos(math.pi / 32)
POST_RADIUS = 0.002
POST_BOTTOM = 0.0105
POST_TOP = 0.0355
BOARD_TOP = 0.010
REST_Z = BOARD_TOP - PEG_MIN[2]
RESET_Z = REST_Z + 0.0004
CLEAR_Z = POST_TOP - PEG_MIN[2] + 0.001
CONTACT_MARGIN = 0.0002
THREAD_TOLERANCE = HOLE_INRADIUS - POST_RADIUS - CONTACT_MARGIN
TARGET = (0.042 - HOLE_CENTER[0], 0.015, REST_Z)
SOURCE_POSTS = ((0.0, 0.0), (0.0, -0.030))
POSTS = ((-0.042, 0.030), (-0.042, 0.0), (-0.042, -0.030),
         (0.0, 0.030), *SOURCE_POSTS, (0.042, -0.015), (0.042, 0.015),
         (0.084, 0.035), (0.084, -0.035), (0.126, -0.015), (0.126, 0.015))
TARGET_POST = (0.042, 0.015)
# Keep the original source labels stable; append unique non-target board posts.
SOURCE_POSTS = ((0.0, 0.0), (0.0, -0.030), *(post for post in POSTS
    if post != TARGET_POST and post not in ((0.0, 0.0), (0.0, -0.030))))
SOURCE_LABELS = ('L5', 'L6', *(f'P{i}' for i in range(1, len(SOURCE_POSTS) - 1)))
GRASP1 = (0.003, -0.009, -0.006)
GRASP2 = (0.014, 0.0, -0.006)
GRASP_RADIUS = 0.004
LIFT_Z = 0.055
HANDOVER = (0.020, -0.0075, 0.065)
ABOVE_TARGET = (*TARGET[:2], 0.065)


def source_label(peg_root_xy):
    """Label a possibly jittered source by its nearest nominal board post."""
    xy = peg_root_xy + peg_root_xy.new_tensor(HOLE_CENTER)
    posts = xy.new_tensor(SOURCE_POSTS)
    nearest = torch.linalg.vector_norm(posts[None] - xy[:, None], dim=2).argmin(dim=1)
    return [SOURCE_LABELS[int(i)] for i in nearest]


def collider_sectors():
    """Convex prisms with a guaranteed open hole, outer rectangle and exact z bounds.

    Extra rays through rectangle corners keep each prism convex and the union closed.
    Points are in metres relative to the rigid-body root, before USD scaling.
    """
    cx, cy = HOLE_CENTER
    angles = [2 * math.pi * i / 32 for i in range(32)]
    angles += [math.atan2(y - cy, x - cx) % (2 * math.pi)
               for x in (PEG_MIN[0], PEG_MAX[0]) for y in (PEG_MIN[1], PEG_MAX[1])]
    angles = sorted(set(angles))

    def ray(a):
        dx, dy = math.cos(a), math.sin(a)
        tx = ((PEG_MAX[0] if dx >= 0 else PEG_MIN[0]) - cx) / dx if abs(dx) > 1e-12 else math.inf
        ty = ((PEG_MAX[1] if dy >= 0 else PEG_MIN[1]) - cy) / dy if abs(dy) > 1e-12 else math.inf
        r = min(tx, ty)
        return (cx + r * dx, cy + r * dy)

    result = []
    for a, b in zip(angles, angles[1:] + [angles[0] + 2 * math.pi]):
        polygon = [(cx + HOLE_RADIUS * math.cos(a), cy + HOLE_RADIUS * math.sin(a)),
                   ray(a), ray(b),
                   (cx + HOLE_RADIUS * math.cos(b), cy + HOLE_RADIUS * math.sin(b))]
        result.append([(x, y, z) for z in (PEG_MIN[2], PEG_MAX[2]) for x, y in polygon])
    return result


def upright_path_safe(start, end):
    """Conservative swept test against all posts and board for upright assisted poses.

    Callers limit displacement to 4 mm; samples at <=0.125 mm plus a 0.2 mm
    margin prevent jumping across a post wall. No contact impulses are simulated here.
    """
    t = torch.linspace(0, 1, 33, device=start.device, dtype=start.dtype)
    p = start[:, None, :] + t[None, :, None] * (end - start)[:, None, :]
    posts = start.new_tensor(POSTS)
    rel = posts[None, None, :, :] - p[:, :, None, :2]
    lo, hi = start.new_tensor(PEG_MIN[:2]), start.new_tensor(PEG_MAX[:2])
    rectangle_distance = torch.linalg.vector_norm(torch.maximum(lo - rel, rel - hi).clamp_min(0), dim=-1)
    outside = rectangle_distance > POST_RADIUS + CONTACT_MARGIN
    threaded = torch.linalg.vector_norm(rel - start.new_tensor(HOLE_CENTER), dim=-1) < THREAD_TOLERANCE
    bottom, top = p[:, :, 2] + PEG_MIN[2], p[:, :, 2] + PEG_MAX[2]
    overlap_z = (bottom < POST_TOP + CONTACT_MARGIN) & (top > POST_BOTTOM - CONTACT_MARGIN)
    post_safe = (~overlap_z[:, :, None]) | outside | threaded
    # Permit recovery from PhysX contact slop, always finishing above the board.
    board_safe = (bottom >= BOARD_TOP - CONTACT_MARGIN).all(dim=1)
    board_safe &= end[:, 2] + PEG_MIN[2] >= BOARD_TOP - 1e-6
    return post_safe.all(dim=(1, 2)) & board_safe
