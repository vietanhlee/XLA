"""
Forwarder entry point for Standard ZED Training.
Redirects to root train.py to eliminate code duplication.
"""
import sys
from pathlib import Path

# Add root dir to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from train import main

if __name__ == "__main__":
    main()
