import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import h5py
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "This runner requires PyTorch. Install with `pip install torch` before running."
    ) from exc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimization.experiments_settings import (
    codes,
    load_tanner_graph,
    noise_levels,
    parse_edgelist,
    path_to_initial_codes,
    tanner_graph_to_parity_check_matrix,
    textfiles,
)
from optimization.logical_guided_search.logical_guided_eval import (
    evaluate_mc,
    get_code_parameters_and_matrices,
)
from optimization.logical_guided_search.logical_guided_search_core import (
    find_low_weight_classical_codeword,
    generate_logical_guided_candidates,
)


@dataclass
class EnvConfig:
    p: float
    episode_steps: int = 20
    num_candidates: int = 16
    logical_max_comb_order: int = 5
    proposal_max_tries: int = 200
    require_detectable: bool = True
    require_distance_non_decrease: bool = False
    reward_distance: float = 1.0
    reward_logical_weight: float = 0.2
    reward_mc_ler: float = 0.0
    mc_budget: int = 0
    mc_workers: int = 1
    mc_batch_size: int = 5000
    candidate_workers: int = 1


class HGPCandidateEnv:
    """Environment with dynamic candidate actions generated from the current Tanner graph."""

    def __init__(self, initial_state: nx.MultiGraph, cfg: EnvConfig):
        self.initial_state = nx.MultiGraph(initial_state)
        self.cfg = cfg
        self.max_actions = cfg.num_candidates

        self.state = None
        self.state_params = None
        self.state_distance = None
        self.state_logical_weight = None
        self.state_ler = None

        self.step_idx = 0
        self.best_state = None
        self.best_distance = -np.inf
        self.best_ler = np.inf
        self.best_score = -np.inf

        self._candidates = []

    def _state_features(self):
        n_cl = max(float(self.state_params["n_classical"]), 1.0)
        n_q = max(float(self.state_params["n_quantum"]), 1.0)

        d_cl = float(self.state_params["d_classical"])
        d_t = float(self.state_params["d_T_classical"])
        d_q = float(self.state_params["d_quantum"])
        k_q = float(self.state_params["k_quantum"])

        log_w = float(self.state_logical_weight) if self.state_logical_weight is not None else n_cl
        ler_val = float(self.state_ler) if self.state_ler is not None else 1.0

        feat = np.array(
            [
                d_cl / n_cl,
                d_t / n_cl if np.isfinite(d_t) else 1.0,
                d_q / n_cl if np.isfinite(d_q) else 1.0,
                k_q / n_q,
                log_w / n_cl,
                np.log1p(log_w),
                ler_val,
                np.log1p(ler_val),
                self.step_idx / max(self.cfg.episode_steps, 1),
                1.0,
            ],
            dtype=np.float32,
        )
        return feat

    def _candidate_features(self):
        feat_dim = 8
        features = np.zeros((self.max_actions, feat_dim), dtype=np.float32)
        mask = np.zeros((self.max_actions,), dtype=np.bool_)

        n_cl = max(float(self.state_params["n_classical"]), 1.0)
        n_q = max(float(self.state_params["n_quantum"]), 1.0)
        old_d = float(self.state_distance)
        old_lw = float(self.state_logical_weight) if self.state_logical_weight is not None else n_cl

        for i, cand in enumerate(self._candidates[: self.max_actions]):
            new_d = float(cand["distance_after"])
            d_q = float(cand["params"]["d_quantum"])
            k_q = float(cand["params"]["k_quantum"])
            lw = float(cand.get("logical_weight", old_lw))

            features[i] = np.array(
                [
                    (new_d - old_d) / n_cl,
                    new_d / n_cl,
                    d_q / n_cl if np.isfinite(d_q) else 1.0,
                    k_q / n_q,
                    lw / n_cl,
                    (old_lw - lw) / max(old_lw, 1.0),
                    float(cand["distance_before"]) / n_cl,
                    1.0,
                ],
                dtype=np.float32,
            )
            mask[i] = True

        return features, mask

    @staticmethod
    def _estimate_logical_weight(state: nx.MultiGraph):
        H = tanner_graph_to_parity_check_matrix(state)
        vec, w = find_low_weight_classical_codeword(csr_matrix(H, dtype=np.uint8), max_comb_order=5)
        if vec is None or w is None:
            return None
        return float(w)

    def _refresh_candidates(self):
        self._candidates = generate_logical_guided_candidates(
            state=self.state,
            get_code_parameters_and_matrices=get_code_parameters_and_matrices,
            num_candidates=self.cfg.num_candidates,
            proposal_max_tries=self.cfg.proposal_max_tries,
            logical_max_comb_order=self.cfg.logical_max_comb_order,
            require_detectable=self.cfg.require_detectable,
            require_distance_non_decrease=self.cfg.require_distance_non_decrease,
            candidate_workers=self.cfg.candidate_workers,
            verbose=False,
        )

    def reset(self):
        self.state = nx.MultiGraph(self.initial_state)
        self.state_params, _, _ = get_code_parameters_and_matrices(self.state)
        self.state_distance = min(self.state_params["d_classical"], self.state_params["d_T_classical"])
        self.state_logical_weight = self._estimate_logical_weight(self.state)
        self.state_ler = None

        # Provide an MC-LER baseline so reward_mc_ler can contribute from the first step.
        if self.cfg.mc_budget > 0:
            _, Hx, Hz = get_code_parameters_and_matrices(self.state)
            mc_result = evaluate_mc(
                Hx,
                Hz,
                self.cfg.p,
                self.cfg.mc_budget,
                run_label="rl_screen_reset",
                workers=self.cfg.mc_workers,
                batch_size=self.cfg.mc_batch_size,
            )
            self.state_ler = float(mc_result["ler"])

        self.step_idx = 0
        self.best_state = nx.MultiGraph(self.state)
        self.best_distance = float(self.state_distance)
        self.best_ler = np.inf
        self.best_score = self.best_distance

        self._refresh_candidates()
        obs = self._state_features()
        action_features, action_mask = self._candidate_features()
        return obs, action_features, action_mask

    def step(self, action_idx: int):
        if action_idx < 0 or action_idx >= len(self._candidates):
            raise ValueError(f"Invalid action index {action_idx} for {len(self._candidates)} candidates")

        cand = self._candidates[action_idx]
        prev_distance = float(self.state_distance)
        prev_log_w = self.state_logical_weight
        prev_ler = self.state_ler

        self.state = nx.MultiGraph(cand["state"])
        self.state_params = cand["params"]
        self.state_distance = float(cand["distance_after"])
        self.state_logical_weight = self._estimate_logical_weight(self.state)

        self.state_ler = None
        if self.cfg.mc_budget > 0:
            _, Hx, Hz = get_code_parameters_and_matrices(self.state)
            mc_result = evaluate_mc(
                Hx,
                Hz,
                self.cfg.p,
                self.cfg.mc_budget,
                run_label=f"rl_screen_step_{self.step_idx}",
                workers=self.cfg.mc_workers,
                batch_size=self.cfg.mc_batch_size,
            )
            self.state_ler = float(mc_result["ler"])

        rew_dist = (self.state_distance - prev_distance)

        rew_logw = 0.0
        if prev_log_w is not None and self.state_logical_weight is not None:
            rew_logw = (prev_log_w - self.state_logical_weight) / max(prev_log_w, 1.0)

        rew_ler = 0.0
        if self.cfg.reward_mc_ler != 0.0 and prev_ler is not None and self.state_ler is not None:
            rew_ler = (prev_ler - self.state_ler) / max(prev_ler, 1e-12)

        reward = (
            self.cfg.reward_distance * rew_dist
            + self.cfg.reward_logical_weight * rew_logw
            + self.cfg.reward_mc_ler * rew_ler
        )

        score = self.state_distance + 0.01 * rew_logw
        if score > self.best_score:
            self.best_score = score
            self.best_state = nx.MultiGraph(self.state)

        if self.state_distance > self.best_distance:
            self.best_distance = self.state_distance

        if self.state_ler is not None and self.state_ler < self.best_ler:
            self.best_ler = self.state_ler

        self.step_idx += 1
        done = self.step_idx >= self.cfg.episode_steps

        if not done:
            self._refresh_candidates()
            if len(self._candidates) == 0:
                done = True

        if done:
            obs = np.zeros(10, dtype=np.float32)
            action_features = np.zeros((self.max_actions, 8), dtype=np.float32)
            action_mask = np.zeros((self.max_actions,), dtype=np.bool_)
        else:
            obs = self._state_features()
            action_features, action_mask = self._candidate_features()

        info = {
            "distance": float(self.state_distance),
            "logical_weight": self.state_logical_weight,
            "ler": self.state_ler,
            "num_candidates": len(self._candidates),
            "best_distance": float(self.best_distance),
            "best_ler": float(self.best_ler),
        }
        return obs, action_features, action_mask, float(reward), done, info


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + action_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action_features, action_mask):
        # obs: [B, obs_dim], action_features: [B, A, action_feat_dim], action_mask: [B, A]
        h = self.obs_encoder(obs)
        B, A, _ = action_features.shape

        h_expand = h.unsqueeze(1).expand(B, A, h.shape[-1])
        logits = self.action_head(torch.cat([h_expand, action_features], dim=-1)).squeeze(-1)

        invalid = ~action_mask
        logits = logits.masked_fill(invalid, -1e9)
        value = self.value_head(h).squeeze(-1)
        return logits, value


def masked_categorical_sample(logits, action_mask):
    probs = torch.softmax(logits, dim=-1)
    probs = probs * action_mask.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return action, log_prob, entropy


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def save_best_state_hdf5(code_name: str, run_name: str, state: nx.MultiGraph, attrs: dict):
    output_file = "optimization/results/rl_hgp_search.hdf5"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, "a") as f:
        grp = f.require_group(code_name)
        run_grp = grp.require_group(run_name)

        for k, v in attrs.items():
            run_grp.attrs[k] = v

        best_edges = parse_edgelist(state).astype(np.uint32)
        if "best_state" in run_grp:
            del run_grp["best_state"]
        run_grp.create_dataset("best_state", data=best_edges[np.newaxis, :], dtype=np.uint32)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="RL (PPO-style) runner for HGP code optimization")
    parser.add_argument("-C", default=0, type=int, help="Code family index")
    parser.add_argument("-p", default=None, type=float, help="Physical error rate")
    parser.add_argument("--episodes", default=200, type=int)
    parser.add_argument("--episode_steps", default=20, type=int)
    parser.add_argument("--num_candidates", default=50, type=int)
    parser.add_argument("--logical_max_comb_order", default=5, type=int)
    parser.add_argument("--proposal_max_tries", default=200, type=int)
    parser.add_argument("--candidate_workers", default=1, type=int, help="Thread workers for candidate evaluation before MC")
    parser.add_argument("--mc_budget", default=0, type=int, help="MC budget during RL steps (0 disables MC reward)")
    parser.add_argument("--mc_workers", default=1, type=int, help="Parallel workers for MC in reset/step reward")
    parser.add_argument("--mc_batch_size", default=5000, type=int, help="Batch size per MC worker for reset/step reward")
    parser.add_argument("--reward_distance", default=1.0, type=float)
    parser.add_argument("--reward_logical_weight", default=0.2, type=float)
    parser.add_argument("--reward_mc_ler", default=0.0, type=float)
    parser.add_argument("--final_eval_budget", default=100000, type=int)
    parser.add_argument("--final_eval_workers", default=1, type=int, help="Parallel workers for final precise MC")
    parser.add_argument("--final_eval_batch_size", default=5000, type=int, help="Batch size per worker for final precise MC")
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--gae_lambda", default=0.95, type=float)
    parser.add_argument("--clip_eps", default=0.2, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--ppo_epochs", default=4, type=int)
    parser.add_argument("--entropy_coef", default=0.01, type=float)
    parser.add_argument("--value_coef", default=0.5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.reward_mc_ler != 0.0 and args.mc_budget <= 0:
        raise ValueError("Set --mc_budget > 0 when using nonzero --reward_mc_ler")
    if args.mc_workers < 1:
        raise ValueError("--mc_workers must be >= 1")
    if args.final_eval_workers < 1:
        raise ValueError("--final_eval_workers must be >= 1")
    if args.candidate_workers < 1:
        raise ValueError("--candidate_workers must be >= 1")
    if args.mc_batch_size < 1:
        raise ValueError("--mc_batch_size must be >= 1")
    if args.final_eval_batch_size < 1:
        raise ValueError("--final_eval_batch_size must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    C = args.C
    code_name = codes[C]
    p = noise_levels[C] if args.p is None else args.p

    print("\n--- RL HGP OPTIMIZATION (PPO-STYLE) ---")
    print(f"Code family: {code_name}")
    print(f"Noise level p = {p}")
    print(f"Episodes = {args.episodes}, Episode steps = {args.episode_steps}")
    print(f"Candidate actions per step = {args.num_candidates}")
    print(f"Candidate evaluation workers = {args.candidate_workers}")
    print(
        f"MC reward eval: budget={args.mc_budget}, workers={args.mc_workers}, "
        f"batch_size={args.mc_batch_size}"
    )
    print(
        f"Final eval: budget={args.final_eval_budget}, workers={args.final_eval_workers}, "
        f"batch_size={args.final_eval_batch_size}"
    )

    initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    env_cfg = EnvConfig(
        p=p,
        episode_steps=args.episode_steps,
        num_candidates=args.num_candidates,
        logical_max_comb_order=args.logical_max_comb_order,
        proposal_max_tries=args.proposal_max_tries,
        reward_distance=args.reward_distance,
        reward_logical_weight=args.reward_logical_weight,
        reward_mc_ler=args.reward_mc_ler,
        mc_budget=args.mc_budget,
        mc_workers=args.mc_workers,
        mc_batch_size=args.mc_batch_size,
        candidate_workers=args.candidate_workers,
    )
    env = HGPCandidateEnv(initial_state=initial_state, cfg=env_cfg)

    obs_dim = 10
    act_feat_dim = 8
    model = PolicyValueNet(obs_dim=obs_dim, action_feat_dim=act_feat_dim, hidden_dim=128).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_best_state = nx.MultiGraph(initial_state)
    global_best_distance = -np.inf

    start_t = time.time()

    for ep in range(1, args.episodes + 1):
        obs, act_feats, act_mask = env.reset()

        rollout_obs = []
        rollout_act_feats = []
        rollout_masks = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        ep_return = 0.0

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            act_feats_t = torch.tensor(act_feats, dtype=torch.float32, device=args.device).unsqueeze(0)
            act_mask_t = torch.tensor(act_mask, dtype=torch.bool, device=args.device).unsqueeze(0)

            with torch.no_grad():
                logits, value = model(obs_t, act_feats_t, act_mask_t)
                action, log_prob, _ = masked_categorical_sample(logits, act_mask_t)

            action_idx = int(action.item())
            next_obs, next_act_feats, next_act_mask, reward, done, info = env.step(action_idx)

            rollout_obs.append(obs)
            rollout_act_feats.append(act_feats)
            rollout_masks.append(act_mask)
            rollout_actions.append(action_idx)
            rollout_log_probs.append(float(log_prob.item()))
            rollout_rewards.append(float(reward))
            rollout_dones.append(done)
            rollout_values.append(float(value.item()))

            ep_return += reward
            obs, act_feats, act_mask = next_obs, next_act_feats, next_act_mask

        values_np = np.asarray(rollout_values, dtype=np.float32)
        rewards_np = np.asarray(rollout_rewards, dtype=np.float32)
        dones_np = np.asarray(rollout_dones, dtype=np.bool_)
        advantages, returns = compute_gae(rewards_np, values_np, dones_np, args.gamma, args.gae_lambda)

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs_b = torch.tensor(np.asarray(rollout_obs), dtype=torch.float32, device=args.device)
        act_feats_b = torch.tensor(np.asarray(rollout_act_feats), dtype=torch.float32, device=args.device)
        masks_b = torch.tensor(np.asarray(rollout_masks), dtype=torch.bool, device=args.device)
        actions_b = torch.tensor(np.asarray(rollout_actions), dtype=torch.long, device=args.device)
        old_log_probs_b = torch.tensor(np.asarray(rollout_log_probs), dtype=torch.float32, device=args.device)
        adv_b = torch.tensor(advantages, dtype=torch.float32, device=args.device)
        ret_b = torch.tensor(returns, dtype=torch.float32, device=args.device)

        for _ in range(args.ppo_epochs):
            logits, values = model(obs_b, act_feats_b, masks_b)
            probs = torch.softmax(logits, dim=-1)
            probs = probs * masks_b.float()
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            dist = torch.distributions.Categorical(probs=probs)

            new_log_probs = dist.log_prob(actions_b)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_b)
            pg_loss_1 = ratio * adv_b
            pg_loss_2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_b
            policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()

            value_loss = nn.functional.mse_loss(values, ret_b)
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

        if env.best_distance > global_best_distance:
            global_best_distance = env.best_distance
            global_best_state = nx.MultiGraph(env.best_state)

        if ep % 10 == 0 or ep == 1:
            print(
                f"Episode {ep:4d} | return={ep_return:8.4f} | "
                f"best_d(ep)={env.best_distance:5.1f} | best_d(global)={global_best_distance:5.1f}"
            )

    print("\nRunning final precise MC evaluation on global best state...")
    _, Hx_best, Hz_best = get_code_parameters_and_matrices(global_best_state)
    final_mc = evaluate_mc(
        Hx_best,
        Hz_best,
        p,
        args.final_eval_budget,
        run_label="rl_hgp_final_eval",
        workers=args.final_eval_workers,
        batch_size=args.final_eval_batch_size,
    )

    run_name = (
        f"rl_hgp_ppo_C{C}_p{p}_ep{args.episodes}_steps{args.episode_steps}_"
        f"cand{args.num_candidates}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    output_file = save_best_state_hdf5(
        code_name=code_name,
        run_name=run_name,
        state=global_best_state,
        attrs={
            "algorithm": "ppo_masked_candidates",
            "episodes": args.episodes,
            "episode_steps": args.episode_steps,
            "num_candidates": args.num_candidates,
            "p": p,
            "reward_distance": args.reward_distance,
            "reward_logical_weight": args.reward_logical_weight,
            "reward_mc_ler": args.reward_mc_ler,
            "mc_budget": args.mc_budget,
            "final_eval_budget": args.final_eval_budget,
            "final_ler": final_mc["ler"],
            "final_stderr": final_mc["stderr"],
            "final_runtime": final_mc["runtime"],
            "final_failures": final_mc["failures"],
            "final_completed_runs": final_mc["completed_runs"],
            "final_early_stopped": int(final_mc["early_stopped"]),
            "seed": args.seed,
        },
    )

    ckpt_dir = "optimization/results"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "code_name": code_name,
            "best_distance": float(global_best_distance),
            "final_mc": final_mc,
            "run_name": run_name,
        },
        ckpt_path,
    )

    elapsed = time.time() - start_t
    print("\n=== RL HGP RESULT ===")
    print(f"Best distance found: {global_best_distance}")
    print(f"Final LER: {final_mc['ler']:.6f} ± {final_mc['stderr']:.6f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"HDF5 output: {output_file} (group: {code_name}/{run_name})")
    print(f"Total time: {elapsed // 3600:.0f}h {(elapsed % 3600) // 60:.0f}m {elapsed % 60:.2f}s")


if __name__ == "__main__":
    main()
