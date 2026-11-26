"""Autoregressive-network localizer of Biazzo, Braunstein, Dall'Asta and Mazza (Sci. Rep. 2022),
run through their library `annfore` (github.com/ocadni/annfore, commit ee39c69), with the
settings of their script `nn_run_redo.py` (github.com/ocadni/annfore-results) unless noted.

The method needs the start time and the transmission probability (as in their patient-zero
experiments), so it is always *told* them here.

Time convention (annfore `SirModel.energy_i`): x_i^t is node i's state at time t, and a contact
(t, j -> i) with i S and j I at time t infects i at t + 1 with probability lambda. This is ours with
time k = window tau + k, tau the true start. So:
  - a contact in our window t becomes (t - tau, u, v, p) in both directions;
  - an observer infected at y is I at y - tau, and S at y - tau - 1 if y > tau;
  - an observer healthy at t_obs (not infected before t_obs) is S at t_obs - 1 - tau;
  - the source is the node that is I at time 0, so P(s is the source) = P(x_s^0 = I).
mu = 0 (SI): as in their script for mu = 0, every node also gets the net-only observation
(i, I, t_limit + 10), which forces "never recovers".
p_source = 1e-4 (their hospital patient-zero runs used 1e-5; both make several sources
negligible), p_sus = 0.5 (equal prior weight on every final state, i.e. no extra bias),
p_mc = p_obs = exp(log p_source - t_limit) (their defaults).
"""
from __future__ import annotations

import tempfile
import time

import numpy as np


def ann_source_probs(graph, outbreak, tau: int, p: float, n_beta_steps: int, num_samples: int,
                     seed: int = 0, p_source: float = 1e-4, threads: int = 1) -> tuple[np.ndarray, float]:
    """P(s is the source) for every node (normalized to sum to 1), and the seconds it took."""
    import torch
    from annfore.learn.losses import loss_fn_coeff
    from annfore.learn.opt import make_opt
    from annfore.learn.train import make_training_step_local, train_beta
    from annfore.models import sir_model_N_obs
    from annfore.net import nn_sir_path_obs
    from annfore.utils.graph import find_neighs

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    start = time.time()
    n, t_obs = graph.n_nodes, outbreak.t_observe
    t_limit = t_obs - tau  # states at times 0 .. t_limit
    rows = []
    for t in range(tau, t_obs):
        for u, v in graph.windows[t]:
            rows += [(t - tau, u, v, p), (t - tau, v, u, p)]
    # annfore reads N = max node + 1 and T = max time + 2 from the contacts, so if node n - 1 or time
    # t_limit - 1 is missing, add one zero-probability contact (log(1 - 0) = 0: no effect). It must
    # use a pair with no contact at that time, because annfore overwrites (not adds) a repeated pair.
    if not any(r[0] == t_limit - 1 for r in rows) or not any(r[1] == n - 1 for r in rows):
        taken = {(r[1], r[2]) for r in rows if r[0] == t_limit - 1}
        other = next(j for j in range(n - 1) if (n - 1, j) not in taken)
        rows += [(t_limit - 1, n - 1, other, 0.0), (t_limit - 1, other, n - 1, 0.0)]
    contacts = np.array(rows, dtype=float)

    obs_model, obs_net = [], []  # annfore wants (node, time, state) and (node, state, time)
    for node, y in zip(outbreak.observers, outbreak.observed_times.astype(int)):
        node = int(node)
        if y < t_obs:
            obs_model.append((node, y - tau, 1))
            obs_net.append((node, 1, y - tau))
            if y > tau:
                obs_model.append((node, y - tau - 1, 0))
                obs_net.append((node, 0, y - tau - 1))
        else:
            obs_model.append((node, t_obs - 1 - tau, 0))
            obs_net.append((node, 0, t_obs - 1 - tau))
    obs_net += [(i, 1, t_limit + 10) for i in range(n)]  # mu = 0: never recovers (their script)

    neighs = find_neighs(contacts, N=n, only_minor=True, next_near_neigh=True)
    net = nn_sir_path_obs.SIRPathColdObs(neighs, t_limit + 1, obs_list=obs_net, hidden_layer_spec=[-2] * 3,
                                         dtype=torch.float, device="cpu", in_func=torch.nn.LeakyReLU(),
                                         bias=True, lin_scale_power=2.0, layer_norm=False)
    net.init(method="xavier")
    p_mc = float(np.exp(np.log(p_source) - t_limit))
    model = sir_model_N_obs.SirModel(contacts, mu=0.0, device="cpu", p_source=p_source, p_obs=p_mc, p_sus=0.5, p_w=p_mc)
    model.set_obs(obs_model)
    optimizer = [make_opt(net.params_i[i], lr=1e-3) if len(net.params_i[i]) else [] for i in range(n)]
    betas = np.linspace(0, 1, n_beta_steps + 1)[:-1]
    with tempfile.TemporaryDirectory() as folder:
        train_beta(net, optimizer, model, folder + "/run", loss_fn_coeff, 0, num_samples=num_samples,
                   train_step=make_training_step_local, betas=betas, save_every=10**9, print_log=False)
        # their script then keeps training at beta = 1 while averaging marginals (train_saving_marginals,
        # 100 iterations); here: 100 more steps at beta = 1, then marginals from 10 x num_samples samples
        train_beta(net, optimizer, model, folder + "/end", loss_fn_coeff, 0, num_samples=num_samples,
                   train_step=make_training_step_local, betas=np.ones(100), save_every=10**9, print_log=False)
    marg = net.marginals(num_samples=10 * num_samples, batch_size=1000).cpu().numpy()
    prob = marg[:, 0, 1].astype(float)  # P(I at time 0)
    total = prob.sum()
    return (prob / total if total > 0 else prob), time.time() - start
