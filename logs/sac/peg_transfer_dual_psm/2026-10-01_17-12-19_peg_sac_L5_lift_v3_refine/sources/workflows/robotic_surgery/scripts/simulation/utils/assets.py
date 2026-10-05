# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

ASSET_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.5.0/132c82d/"

# Docker mounts this directory at /workspace/i4h/data. Resolve it from the
# checkout so the scene also works in a local Isaac Lab installation.
_LOCAL_ASSET_ROOT = Path(__file__).resolve().parents[5] / "data/Isaac/Healthcare/0.5.0/132c82d"


def _asset_url(relative_path: str) -> str:
    for root in (_LOCAL_ASSET_ROOT, Path("/workspace/i4h/data/Isaac/Healthcare/0.5.0/132c82d")):
        local_path = root / relative_path
        if local_path.is_file():
            return str(local_path)
    return ASSET_PATH + relative_path

DVRK_PSM_USD = ASSET_PATH + "Robots/dVRK/PSM/psm.usd"
BLOCK_USD = _asset_url("Props/PegBlock/block.usd")
TABLE_USD = _asset_url("Props/Table/table.usd")
