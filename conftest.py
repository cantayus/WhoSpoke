"""Test configuration for pytest.

This file adjusts ``sys.path`` so that the package under development can be
imported from the ``src`` directory without installing it. By adding the
project's ``src`` directory to the front of ``sys.path``, imports such as
``from WhoSpoke import SpeakerVideoAnalyzer`` will resolve correctly.
"""

import sys
from pathlib import Path

# Determine the path to the repository root (parent of the 'tests' directory)
_here = Path(__file__).resolve()
_project_root = _here.parents[1]
_src_path = _project_root / "src"

# Insert the src directory at the beginning of sys.path
sys.path.insert(0, str(_src_path))