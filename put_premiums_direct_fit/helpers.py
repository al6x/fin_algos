import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
import os
from scipy.optimize import brentq
import os, pickle
from datetime import datetime
import matplotlib.cm as cm
import re

def parse_inits_bounds(init_params):
  inits, bounds = [], []
  for p in init_params:
    if isinstance(p, tuple):
      init, lower, upper = p
      inits.append(init); bounds.append((lower, upper))
    else:
      inits.append(p); bounds.append((None, None))
  return inits, bounds

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def logit(y):
  return np.log(y / (1.0 - y))

report_path = f'readme.md'
report_first_call = True
def report(msg, print_=True):
  global report_first_call

  def turn2space_intent_into4(text):
    lines = text.splitlines()
    fixed = []
    for line in lines:
      if line.startswith("  ") and not line.startswith("    "):
        fixed.append("  " + line)
      else:
        fixed.append(line)
    return "\n".join(fixed)

  def replace_h1_with_h3(text):
    lines = text.splitlines()
    fixed = [
      line.replace('# ', '### ', 1) if line.lstrip().startswith('# ') else line
      for line in lines
    ]
    return '\n'.join(fixed)

  if report_first_call:
    report_first_call = False
    if os.path.exists(report_path):
      os.remove(report_path)

  if print_:
    print(msg)
  # os.makedirs(os.path.dirname(report_path), exist_ok=True)
  with open(report_path, "a") as f:
    f.write(turn2space_intent_into4(replace_h1_with_h3(msg)).rstrip() + "\n\n")

def save_asset(obj, name):
  def safe_name(s):
    s = re.sub(r'[^a-zA-Z0-9]', '-', s)
    s = re.sub(r'-+', '-', s)
    return s.strip('-').lower()

  # path = f'{report_path}/{name}'
  base_path, _ = os.path.splitext(report_path)
  path = f'{base_path}/{safe_name(name)}.png'
  os.makedirs(os.path.dirname(path), exist_ok=True)
  if hasattr(obj, "savefig"):
    obj.savefig(path)
  elif isinstance(obj, str):
    with open(path, "w") as f:
      f.write(obj)
  else:
    raise ValueError("Unsupported asset type: expected figure or string")
  report(f'{name}\n\n![{name}]({path})')