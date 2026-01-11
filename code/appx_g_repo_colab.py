"""
Python & AI in Asset Management
Appendix G · Repository and Colab Guide

(c) Dr. Yves J. Hilpisch
AI-Powered by GPT 5.1
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

# %% cell 4
import os
import sys
from pathlib import Path

print('Python', sys.version)
print('CWD   ', os.getcwd())

# %% cell 6
root = Path('..').resolve()
for path in sorted(root.iterdir()):
    if path.name.startswith('.'):
        continue
    print(path)

# %% cell 8
data_path = Path('data/pyaiam_eod.csv')
print('Data exists:', data_path.exists(), '->', data_path)
