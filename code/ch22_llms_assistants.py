"""
Python & AI in Asset Management
Chapter 22 · LLMs as Research and Coding Assistants

(c) Dr. Yves J. Hilpisch
AI-Powered by GPT 5.1
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

# %% cell 4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.family": "serif", "figure.dpi": 300})

DATA_PATH = Path("data/pyaiam_eod.csv")
if not DATA_PATH.exists():
    DATA_PATH = "https://hilpisch.com/pyaiam_eod.csv"

import json
import textwrap

# %% cell 6
prices = pd.read_csv(DATA_PATH,
parse_dates=['Date']).set_index('Date').sort_index().ffill()

# %% cell 8
PROMPTS = {
    'experiment_summary': textwrap.dedent('''
        You are an assistant for quantitative research. Summarize the
        following experiment metadata in 3 bullet points for a weekly lab
        meeting. Emphasize model type, data span, and key risk notes.

        Metadata: {metadata}
    '''),
    'code_review': textwrap.dedent('''
        You are helping a junior quant understand a code snippet.

        Explain what this code does and mention two potential pitfalls:

        {code}
    '''),
}

def render_prompt(name: str, **kwargs) -> str:
    template = PROMPTS[name]
    return template.format(**kwargs)

# %% cell 10
class MockLLMClient:
    def complete(self, prompt: str) -> str:
        preview = prompt.replace('', ' ')[:200]
        return '[MOCK LLM RESPONSE] ' + preview + ' ...'

llm = MockLLMClient()

# %% cell 12
log_path = Path('../reports/model_risk_rf.json')
if log_path.exists():
    metadata = json.loads(log_path.read_text())
else:
    metadata = {
        'model': 'RandomForestRegressor',
        'train_start': '2020-01-01',
        'train_end': '2025-01-01',
    }
prompt = render_prompt('experiment_summary', metadata=json.dumps(metadata, indent=2))
print(llm.complete(prompt))

# %% cell 14
snippet = """prices = pd.read_csv(DATA_PATH,
parse_dates=['Date']).set_index('Date').sort_index().ffill()
log_rets = np.log(prices / prices.shift(1)).dropna()
rolling_vol = log_rets.rolling(63).std() * np.sqrt(252)
"""

prompt = render_prompt('code_review', code=snippet)
print(llm.complete(prompt))
