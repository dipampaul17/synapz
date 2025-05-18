"""
Synapz: Adaptive learning system for neurodiverse students.

This package provides tools to build and evaluate an adaptive learning system
that tailors educational content to different cognitive profiles.

Core components:
- Database with WAL journaling for concurrent access
- System prompts for adaptive and control teaching
- Cognitive profiles for ADHD, dyslexic, and visual learners
- Content adaptation for different learning styles
- Experiment tracking and analysis
"""

import os
from pathlib import Path

# Setup package paths
PACKAGE_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PACKAGE_ROOT / "data"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"

# Version information
__version__ = "0.2.0"
__author__ = "Synapz Team" 