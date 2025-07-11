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

  def turn2space_indent_into4(text):
    return text.replace('\n  ', '\n    ')

  # def replace_h1_with_h3(text):
  #   lines = text.splitlines()
  #   fixed = [
  #     line.replace('\n# ', '\n### ', 1) if line.lstrip().startswith('# ') else line
  #     for line in lines
  #   ]
  #   return '\n'.join(fixed)
  def replace_h1_with_h3(text):
    return re.sub(r'(^|\n)# ', r'\1### ', text)

  def dedent(s):
    lines = s.strip('\n').splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
      return ''
    min_indent = min(len(line) - len(line.lstrip(' ')) for line in non_empty)
    return '\n'.join(line[min_indent:] if len(line) >= min_indent else line for line in lines)

  if report_first_call:
    report_first_call = False
    if os.path.exists(report_path):
      os.remove(report_path)

  msg = dedent(msg)

  if print_:
    indented_msg = "\n".join("  " + line for line in msg.splitlines())
    print(indented_msg + "\n")

  # os.makedirs(os.path.dirname(report_path), exist_ok=True)
  with open(report_path, "a") as f:
    msg = turn2space_indent_into4(replace_h1_with_h3(msg)).rstrip()
    f.write(msg + "\n\n")

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
  # report(f'{name}\n\n![{name}]({path})')
  report(f'![{name}]({path})')

def cached(key, get):
  path = f"./tmp/cache/{key}-{datetime.now():%Y-%m-%d}.pkl"
  if os.path.exists(path):
    print(f"  {key} loaded from cache")
    with open(path, 'rb') as f: return pickle.load(f)
  print(f"  {key} calculating")
  result = get()
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'wb') as f: pickle.dump(result, f)
  return result

