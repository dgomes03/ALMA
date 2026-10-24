import sys
import os

# Put the repository root on sys.path so `core`, `components`, ... import
# from anywhere pytest is invoked.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
