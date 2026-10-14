"""Environment check: confirms every required library is installed.

    python test.py
"""
import sys

import gymnasium as gym
import numpy as np
import matplotlib
import pandas as pd

print(f"Python     {sys.version.split()[0]}")
print(f"gymnasium  {gym.__version__}")
print(f"numpy      {np.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print(f"pandas     {pd.__version__}")
print("All good")
