
from optimization.logical_guided_search.logical_guided_rl_policy import CandidateActorCritic
from optimization.logical_guided_search.logical_guided_rl_env import LogicalGuidedSwapEnv
from optimization.experiments_settings import (
    load_tanner_graph,
    from_edgelist,
    codes,
    path_to_initial_codes,
    textfiles,
)
import argparse
import os
import sys
import time
import random
from datetime import datetime

import torch
import torch.nn.functional as F
import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def obs_to_tensors(obs, device):
    return (
        torch.tensor(obs["global"], dtype=torch.float32,
                     device=device).unsqueeze(0),
        torch.tensor(obs["candidates"], dtype=torch.float32,
                     device=device).unsqueeze(0),
        torch.tensor(obs["mask"], dtype=torch.float32,
                     device=device).unsqueeze(0),
    )


def compute_returns_advantages(rewards, values, dones, gamma=0.99):
    returns = []
    G = 0.0

    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0.0
        G = r + gamma * G
        returns.append(G)

    returns = list(reversed(returns))
    advantages = np.array(returns, dtype=np.float32) - \
        np.array(values, dtype=np.float32)

    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

    return np.array(returns, dtype=np.float32), advantages.astype(np.float32)


def ppo_update(
    model,
    optimizer,
    batch,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    epochs=4,
):
    device = next(model.parameters()).device

    global_obs = torch.tensor(
        np.stack(batch["global"]), dtype=torch.float32, device=device)
    cand_obs = torch.tensor(
        np.stack(batch["candidates"]), dtype=torch.float32, device=device)
    mask = torch.tensor(
        np.stack(batch["mask"]), dtype=torch.float32, device=device)

    actions = torch.tensor(batch["actions"], dtype=torch.long, device=device)
    old_logprobs = torch.tensor(
        batch["logprobs"], dtype=torch.float32, device=device)
    returns = torch.tensor(
        batch["returns"], dtype=torch.float32, device=device)
    advantages = torch.tensor(
        batch["advantages"], dtype=torch.float32, device=device)

    for _ in range(epochs):
        logits, values = model(global_obs, cand_obs, mask)
        dist = torch.distributions.Categorical(logits=logits)

        new_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        ratio = torch.exp(new_logprobs - old_logprobs)

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_eps,
                              1.0 + clip_eps) * advantages

        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return float(loss.item())


def collect_episode(env, model, device):
    obs = env.reset()

    data = {
        "global": [],
        "candidates": [],
        "mask": [],
        "actions": [],
        "logprobs": [],
        "values": [],
        "rewards": [],
        "dones": [],
        "infos": [],
    }

    done = False

    while not done:
        if obs["mask"].sum() == 0:
            break

        action, logprob, value = model.act(obs)

        print(f"Step {len(data['actions'])} | Action: {action} | LogProb: {logprob:.4f} | Value: {value:.4f}")
        next_obs, reward, done, info = env.step(action)

        data["global"].append(obs["global"])
        data["candidates"].append(obs["candidates"])
        data["mask"].append(obs["mask"])
        data["actions"].append(action)
        data["logprobs"].append(logprob)
        data["values"].append(value)
        data["rewards"].append(float(reward))
        data["dones"].append(done)
        data["infos"].append(info)

        obs = next_obs

        if obs is None:
            break

    returns, advantages = compute_returns_advantages(
        data["rewards"],
        data["values"],
        data["dones"],
    )

    data["returns"] = returns
    data["advantages"] = advantages

    return data


def load_initial_state_from_hdf5(
    input_file: str,
    input_code: str,
    input_run_name: str | None = None,
    input_dataset: str = "best_state",
):
    with h5py.File(input_file, "r") as f:
        grp = f[input_code]

        if input_run_name is not None:
            grp = grp[input_run_name]

        edge_list = np.asarray(grp[input_dataset][:])

    if edge_list.ndim == 2 and edge_list.shape[0] == 1:
        edge_list = edge_list[0]

    return from_edgelist(edge_list)


def merge_episode_batches(episodes):
    batch = {
        "global": [],
        "candidates": [],
        "mask": [],
        "actions": [],
        "logprobs": [],
        "values": [],
        "returns": [],
        "advantages": [],
    }

    infos = []
    rewards = []

    for ep in episodes:
        if len(ep["actions"]) == 0:
            continue

        for key in batch:
            batch[key].extend(ep[key])

        infos.extend(ep["infos"])
        rewards.extend(ep["rewards"])

    return batch, infos, rewards


def evaluate_policy_episode(env, model, greedy: bool = True):
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    final_info = None

    done = False

    while not done:
        if obs["mask"].sum() == 0:
            break

        if greedy:
            device = next(model.parameters()).device
            global_obs = torch.tensor(
                obs["global"], dtype=torch.float32, device=device).unsqueeze(0)
            cand_obs = torch.tensor(
                obs["candidates"], dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.tensor(
                obs["mask"], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(global_obs, cand_obs, mask)
                action = int(torch.argmax(logits, dim=-1).item())
        else:
            action, _, _ = model.act(obs)

        obs, reward, done, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        final_info = info

        if obs is None:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "final_info": final_info,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-C", default=2, type=int, help="Code family index")
    parser.add_argument("--episodes", default=50, type=int)
    parser.add_argument("--rollout-episodes", default=4, type=int)
    parser.add_argument("--steps", default=8, type=int)
    parser.add_argument("--trials-per-step", default=100, type=int)
    parser.add_argument("--candidates", default=16, type=int)
    parser.add_argument("--score-beta", default=0.3, type=float)
    parser.add_argument("--score-window", default=2, type=int)

    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--ppo-epochs", default=4, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--clip-eps", default=0.2, type=float)
    parser.add_argument("--vf-coef", default=0.5, type=float)
    parser.add_argument("--ent-coef", default=0.01, type=float)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--candidate-order", default="mixed",
                        choices=["ranked", "shuffle", "mixed"])
    parser.add_argument("--elite-frac", default=0.25, type=float)

    parser.add_argument(
        "--require-k-preserved",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--input-file", default=None, type=str)
    parser.add_argument("--input-code", default=None, type=str)
    parser.add_argument("--input-run-name", default=None, type=str)
    parser.add_argument("--input-dataset", default="best_state", type=str)

    parser.add_argument(
        "--output-dir", default="optimization/results/ppo", type=str)
    parser.add_argument("--save-every", default=10, type=int)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    C = args.C
    input_code = args.input_code if args.input_code is not None else codes[C]

    if args.input_file is not None:
        print("Loading initial state from HDF5")
        print(f"  input_file={args.input_file}")
        print(f"  input_code={input_code}")
        print(f"  input_run_name={args.input_run_name}")
        print(f"  input_dataset={args.input_dataset}")

        initial_state = load_initial_state_from_hdf5(
            input_file=args.input_file,
            input_code=input_code,
            input_run_name=args.input_run_name,
            input_dataset=args.input_dataset,
        )
    else:
        print("Loading initial state from text file")
        print(f"  code={codes[C]}")
        print(f"  file={path_to_initial_codes + textfiles[C]}")

        initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        args.output_dir,
        f"ppo_C{C}_K{args.candidates}_S{args.steps}_seed{args.seed}_{timestamp}.pt",
    )

    print("\n--- LOGICAL-GUIDED PPO TRAINING ---")
    print(f"Code family: {codes[C]}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Rollout episodes/update: {args.rollout_episodes}")
    print(f"Steps/episode: {args.steps}")
    print(f"Candidate slots K: {args.candidates}")
    print(f"Candidate order: {args.candidate_order}")
    print(f"Elite frac: {args.elite_frac}")
    print(f"Score beta: {args.score_beta}")
    print(f"Score window: {args.score_window}")
    print(f"Require k preserved: {args.require_k_preserved}")
    print(f"Model path: {model_path}")

    # Based on your current env:
    # global features = 8
    # candidate features = 14
    model = CandidateActorCritic(
        global_dim=8,
        candidate_dim=14,
        hidden=args.hidden,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_eval_reward = -float("inf")
    best_eval_info = None

    start_time = time.time()
    update_idx = 0

    for episode_start in range(0, args.episodes, args.rollout_episodes):
        collected = []

        print(
            f"\n=== Update {update_idx:04d} | Episodes {episode_start}-{min(episode_start + args.rollout_episodes - 1, args.episodes - 1)} ===")

        for j in range(args.rollout_episodes):
            print(f"\n--- Collecting episode {episode_start + j} ---")
            ep_idx = episode_start + j
            if ep_idx >= args.episodes:
                break

            env = LogicalGuidedSwapEnv(
                initial_state=initial_state,
                max_steps=args.steps,
                candidates_per_step=args.candidates,
                score_beta=args.score_beta,
                score_window=args.score_window,
                require_k_preserved=args.require_k_preserved,
                candidate_order=args.candidate_order,
                elite_frac=args.elite_frac,
                seed=args.seed + ep_idx,
                trials_per_step=args.trials_per_step,
            )

            ep_data = collect_episode(env, model, device)

            if len(ep_data["actions"]) > 0:
                collected.append(ep_data)

        batch, infos, rewards = merge_episode_batches(collected)

        if len(batch["actions"]) == 0:
            print(f"Update {update_idx}: no valid rollout data, skipping.")
            continue

        loss = ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            clip_eps=args.clip_eps,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            epochs=args.ppo_epochs,
        )

        update_idx += 1

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        sum_reward = float(np.sum(rewards)) if rewards else 0.0

        final_infos = [ep["infos"][-1]
                       for ep in collected if len(ep["infos"]) > 0]
        final_ds = [info["d_quantum"] for info in final_infos]
        final_scores = [info["score"] for info in final_infos]

        mean_final_d = float(np.mean(final_ds)) if final_ds else float("nan")
        max_final_d = int(np.max(final_ds)) if final_ds else -1
        min_final_score = float(
            np.min(final_scores)) if final_scores else float("nan")

        best_weight_patterns = None
        if final_infos:
            best_idx = int(np.argmin(final_scores))
            best_weight_patterns = final_infos[best_idx].get(
                "weight_patterns", None)

        print(
            f"update={update_idx:04d} | "
            f"episodes={episode_start}-{episode_start + len(collected) - 1} | "
            f"steps={len(batch['actions'])} | "
            f"loss={loss:+.4f} | "
            f"mean_step_reward={mean_reward:+.4f} | "
            f"sum_reward={sum_reward:+.4f} | "
            f"mean_final_d={mean_final_d:.2f} | "
            f"max_final_d={max_final_d} | "
            f"min_final_score={min_final_score:.6g} | "
            f"best_weight_patterns={best_weight_patterns}"
        )

        # Evaluation with greedy action under current policy.
        if update_idx % args.save_every == 0:
            eval_env = LogicalGuidedSwapEnv(
                initial_state=initial_state,
                max_steps=args.steps,
                candidates_per_step=args.candidates,
                score_beta=args.score_beta,
                score_window=args.score_window,
                require_k_preserved=args.require_k_preserved,
                candidate_order="mixed",
                elite_frac=args.elite_frac,
                seed=args.seed + 100000 + update_idx,
                trials_per_step=args.trials_per_step,
            )

            eval_result = evaluate_policy_episode(eval_env, model, greedy=True)
            eval_reward = eval_result["total_reward"]
            eval_info = eval_result["final_info"]

            if eval_info is not None:
                print(
                    f"  eval | reward={eval_reward:+.4f} | "
                    f"steps={eval_result['steps']} | "
                    f"d={eval_info['d_quantum']} | "
                    f"k={eval_info['k_quantum']} | "
                    f"rank={eval_info['rank_H']} | "
                    f"score={eval_info['score']:.6g} | "
                    f"patterns={eval_info.get('weight_patterns', {})}"
                )

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_eval_info = eval_info

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "best_eval_reward": best_eval_reward,
                        "best_eval_info": best_eval_info,
                    },
                    model_path,
                )

                print(f"  saved best model to {model_path}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "best_eval_reward": best_eval_reward,
            "best_eval_info": best_eval_info,
        },
        model_path,
    )

    elapsed = time.time() - start_time
    print("\nTraining done.")
    print(
        f"Runtime: {elapsed // 3600:.0f}h {(elapsed % 3600) // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Final model saved to: {model_path}")

    if best_eval_info is not None:
        print("\nBest eval info:")
        print(best_eval_info)


if __name__ == "__main__":
    main()
