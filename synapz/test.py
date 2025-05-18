#!/usr/bin/env python3
"""Launcher for Synapz test harness."""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The above line adds the 'synapz/synapz' directory to the path.
# For `from synapz.test_harness` to work when running synapz/test.py,
# the project root (/Users/dipampaul_/Downloads/synapz) which contains the top-level synapz package
# needs to be in sys.path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Import here to ensure path is set up correctly and .env is loaded
    from synapz.test_harness import main
    main() 