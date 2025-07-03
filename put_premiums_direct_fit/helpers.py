import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
import os
from scipy.optimize import brentq
import os, pickle
from datetime import datetime
import matplotlib.cm as cm

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

  if report_first_call:
    report_first_call = False
    if os.path.exists(report_path):
      os.remove(report_path)

  if print_:
    print(msg)
  # os.makedirs(os.path.dirname(report_path), exist_ok=True)
  with open(report_path, "a") as f:
    f.write(msg.rstrip() + "\n\n")

def save_asset(obj, name):
  # path = f'{report_path}/{name}'
  base_path, _ = os.path.splitext(report_path)
  path = f'{base_path}/{name}'
  os.makedirs(os.path.dirname(path), exist_ok=True)
  if hasattr(obj, "savefig"):
    obj.savefig(path)
  elif isinstance(obj, str):
    with open(path, "w") as f:
      f.write(obj)
  else:
    raise ValueError("Unsupported asset type: expected figure or string")
  report(f'{name}\n![{name}]({path})')