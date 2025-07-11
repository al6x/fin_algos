from scipy.optimize import minimize
from itertools import combinations
import numpy as np
from scipy.stats import norm

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

def prune_params(loss, init, fit, min_params):
  """
  Exhaustive best-subset selection.

  Args:
    loss       : (P) -> float, where P is a full-length list of params
    fit        : (loss, params) -> { x, fun }
    init       : full-length list or array of initial guesses for P
    min_params : minimum number of nonzero parameters to consider (inclusive)

  Returns:
    results: dict mapping k -> (best_loss, kept_indices, best_P_full)
  """
  n, results = len(init), {}

  # helper: build a loss that only optimizes the entries in `kept`
  def make_masked_loss(kept):
    free_idxs = list(kept)
    def loss_(x_free):
      # reconstruct a full-length P, plugging x_free into kept positions, zeros elsewhere
      P_full = [0.0] * n
      for idx, val in zip(free_idxs, x_free):
        P_full[idx] = val
      return loss(P_full)
    # also give back the corresponding slice of init
    x0_free = [init[i] for i in free_idxs]
    return loss_, x0_free

  # loop over sub-model sizes k = min_params…n
  for k in range(min_params, n+1):
    best_loss, best_kept, best_P = float('inf'), None, None

    # try every combination of k indices to keep
    print(f'  fitting {k} combinations')
    for kept in combinations(range(n), k):
      loss_sub, x0_free = make_masked_loss(kept)

      # fit only the k free parameters
      # res = minimize(loss_sub, x0=x0_free, method='Powell')
      print(f'  .')
      res = fit(loss_sub, x0_free)

      if res.fun < best_loss:
        best_loss, best_kept = res.fun, kept

        # rebuild the winning full-length P
        P_full = [0.0] * n
        for idx, val in zip(kept, res.x):
          P_full[idx] = val
        best_P = P_full

    print(f"  Best {k}-param model loss={best_loss:.4f}, kept={best_kept}")
    results[k] = (best_loss, best_kept, best_P)

  return results

def test_prune_params():
  # true targets for a 3‐param quadratic loss
  target = [1.0, 2.0, 3.0]
  loss = lambda P: sum((P[i] - target[i])**2 for i in range(len(target)))
  init = [0.0, 0.0, 0.0]

  def fit(loss, x0):
    return minimize(loss, x0=x0, method='Powell')

  # exhaustively find best submodels of size 1–4
  results = prune_params(loss=loss, fit=fit, init=init, min_params=1)

  assert results[3][1] == (0, 1, 2) # size=3 should keep all indices
  assert results[2][1] == (1, 2) # size=2 best drops the smallest‐target dim (0), keeps (1,2)
  assert results[1][1] == (2,) # size=1 best keeps the largest‐target dim (2)

def fit_multi_init(loss, inits, fit):
  """
  Try multiple initial guesses for parameter optimization.

  Args:
    loss  : (P) -> float, loss function over full-length parameter list P
    inits : list of initial guesses (each a full-length list or array)
    fit   : (loss, params) -> { x, fun }, optimizer function

  Returns:
    result: (best_loss, best_P, all_results)
      best_loss: lowest loss found
      best_P   : parameter vector achieving best_loss
      all_results: list of tuples (loss, P) for each init
  """
  best_loss, best_P = float('inf'), None
  all_results = []

  for i, init in enumerate(inits):
    res = fit(loss, init)
    loss_val = res.fun
    print(f"  Init #{i+1}: {loss_val:.4f}, {init}")
    P_val    = res.x
    all_results.append((loss_val, P_val))

    if loss_val < best_loss:
      best_loss, best_P = loss_val, P_val

  print(f"  Best loss={best_loss:.6f} at params={best_P}")
  return best_P

def test_fit_multi_init():
  # quadratic loss with known minimum at [1, 2, 3]
  target = [1.0, 2.0, 3.0]
  loss = lambda P: sum((P[i] - target[i])**2 for i in range(len(P)))

  # multiple initial guesses
  inits = [
    [0.0, 0.0, 0.0],
    [5.0, -1.0, 2.0],
    [1.0, 1.0, 1.0]
  ]

  def fit(loss, x0):
    return minimize(loss, x0=x0, method='Powell')

  best_loss, best_P, all_results = fit_multi_init(loss, inits, fit)

  # Check that the best fit is very close to the true target
  assert np.allclose(best_P, target, atol=1e-3)
  assert abs(best_loss) < 1e-6
  print("  test_fit_multi_init passed")

def posterior(v, sigma, fn, shifts=(-1, 0, +1)):
  v = np.asarray(v)
  sigma = np.asarray(sigma)

  weights = norm.pdf(shifts)
  weights /= weights.sum()

  result = 0
  for shift, weight in zip(shifts, weights):
    shifted_v = v + shift * sigma
    result += weight * fn(shifted_v)

  return result

def posterior2(a, sigma_a, b, sigma_b, fn, shifts=(-1, 0, +1)):
  a, b = np.asarray(a), np.asarray(b)
  sigma_a, sigma_b = np.asarray(sigma_a), np.asarray(sigma_b)

  weights = norm.pdf(shifts)
  weights /= weights.sum()

  result = 0
  for i, w_i in enumerate(weights):
    for j, w_j in enumerate(weights):
      shifted_a = a + shifts[i] * sigma_a
      shifted_b = b + shifts[j] * sigma_b
      w_ij = w_i * w_j
      result += w_ij * fn(shifted_a, shifted_b)
  return result

if __name__ == "__main__":
  test_prune_params()
  test_fit_multi_init()