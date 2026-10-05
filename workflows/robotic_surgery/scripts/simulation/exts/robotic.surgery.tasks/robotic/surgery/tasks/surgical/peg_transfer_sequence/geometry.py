"""Sequence-only grasp calibration; legacy board, peg and post geometry retained."""
from ..peg_transfer.sac_geometry import (
    ABOVE_TARGET, CLEAR_Z, GRASP1, GRASP_RADIUS, HANDOVER, PEG_MAX, PEG_MIN,
    REST_Z, TARGET, source_label, upright_path_safe,
)

# Approach the opposite Y edge, parallel to the jaw opening instead of pushing
# the block from +X with the leading open finger. Same 4 mm capture radius.
# PSM1's proven grasp lies at Y=-9 mm; this receiver point lies at Y=+9 mm.
GRASP2 = (.014, .009, -.006)
RECEIVER_OUTSIDE = (0., .015, 0.)
RECEIVER_CAPTURE = (0., .0035, 0.)
