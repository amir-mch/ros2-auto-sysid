#!/usr/bin/env python3
"""ROS 2 Executable Script for SysId Excitation Node."""

import sys
from pathlib import Path

# Ensure package root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_excite.excitation_node import main

if __name__ == "__main__":
    main()
