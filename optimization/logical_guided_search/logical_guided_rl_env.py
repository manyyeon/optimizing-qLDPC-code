import numpy as np
import networkx as nx

from optimization.experiments_settings import parse_edgelist, tanner_graph_to_parity_check_matrix
from optimization.logical_guided_search.logical_guided_eval import (
    get_code_parameters_and_matrices,
    compute_weighted_low_weight_score,
)
from optimization.logical_guided_search.logical_guided_search_core import (
    generate_logical_guided_candidates,
)


class LogicalGuidedSwapEnv:
    def __init__(
        self,
        initial_state: nx.MultiGraph,
        max_steps: int = 50,
        trials_per_step: int | None = None,
        candidates_per_step: int = 32,
        score_beta: float = 0.3,
        score_window: int = 2,
        require_k_preserved: bool = True,
        candidate_order: str = "mixed",
        elite_frac: float = 0.25,
        seed: int | None = None,
    ):
        self.initial_state = nx.MultiGraph(initial_state)
        self.max_steps = max_steps
        self.K = candidates_per_step
        self.score_beta = score_beta
        self.score_window = score_window
        self.require_k_preserved = require_k_preserved

        self.candidate_order = candidate_order
        self.elite_frac = elite_frac
        self.rng = np.random.default_rng(seed)
        self.trials_per_step = trials_per_step if trials_per_step is not None else self.K * 5

        self.state = None
        self.params = None
        self.score_info = None
        self.candidates = []
        self.step_count = 0

    def reset(self):
        self.state = nx.MultiGraph(self.initial_state)
        self.step_count = 0

        self.params, _, _ = get_code_parameters_and_matrices(self.state)
        self.score_info = self._score(self.state, self.params)

        return self._make_observation()

    def _score(self, state, params=None):
        return compute_weighted_low_weight_score(
            state=state,
            params=params,
            beta=self.score_beta,
            max_weight_offset=self.score_window,
        )

    def _generate_candidates(self):
        raw = generate_logical_guided_candidates(
            state=self.state,
            get_code_parameters_and_matrices=get_code_parameters_and_matrices,
            max_trials=self.trials_per_step,
            logical_max_comb_order=5,
            require_detectable=True,
            require_distance_non_decrease=True,
            verbose=False,
            seen_keys=None,
            score_candidate_fn=self._score,
        )

        # raw is already sorted by beam_rank_key inside generate_logical_guided_candidates.
        filtered = []

        for cand in raw:
            if self.require_k_preserved:
                if int(cand["params"]["k_quantum"]) != int(self.params["k_quantum"]):
                    continue
                if int(cand["params"]["rank_H"]) != int(self.params["rank_H"]):
                    continue

            filtered.append(cand)

        if not filtered:
            return []

        if self.candidate_order == "ranked":
            # Greedy / beam-style behavior.
            candidates = filtered[: self.K]

        elif self.candidate_order == "shuffle":
            # Pure exploration: random K candidates from all valid candidates.
            idx = self.rng.permutation(len(filtered))
            candidates = [filtered[i] for i in idx[: self.K]]

        elif self.candidate_order == "mixed":
            # Recommended for PPO:
            # keep some top-ranked candidates, but fill the rest randomly.
            elite_count = int(np.ceil(self.K * self.elite_frac))
            elite_count = min(elite_count, len(filtered), self.K)

            elite = filtered[:elite_count]
            rest = filtered[elite_count:]

            if rest:
                idx = self.rng.permutation(len(rest))
                rest_sample = [rest[i] for i in idx[: self.K - elite_count]]
            else:
                rest_sample = []

            candidates = elite + rest_sample

            # Important: shuffle final slot order so action 0 is not always elite/best.
            idx = self.rng.permutation(len(candidates))
            candidates = [candidates[i] for i in idx]

        else:
            raise ValueError(
                f"Unknown candidate_order={self.candidate_order!r}. "
                "Use 'ranked', 'shuffle', or 'mixed'."
            )

        return candidates

    def _global_features(self):
        components = self.score_info.get("components", {})
        d = int(self.params["d_quantum"])

        return np.array(
            [
                float(self.params["d_quantum"]),
                float(self.params["k_quantum"]),
                float(self.params["rank_H"]),
                np.log1p(float(self.score_info["score"])),
                float(components.get(d, 0)),
                float(components.get(d + 1, 0)),
                float(components.get(d + 2, 0)),
                float(self.step_count) / float(self.max_steps),
            ],
            dtype=np.float32,
        )

    def _candidate_features(self, candidates):
        feats = np.zeros((self.K, 14), dtype=np.float32)

        old_score = float(self.score_info["score"])
        old_d = int(self.params["d_quantum"])
        old_deg = self._degree_summary_from_state(self.state)

        for i, cand in enumerate(candidates[: self.K]):
            new_score = float(cand["score_info"]["score"])
            new_d = int(cand["params"]["d_quantum"])
            components = cand["score_info"].get("components", {})
            new_deg = self._degree_summary_from_state(cand["state"])

            duplicate_add_count = self._duplicate_add_count(cand)

            feats[i] = np.array(
                [
                    float(new_d - old_d),
                    np.log1p(old_score),
                    np.log1p(new_score),
                    np.log1p(old_score) - np.log1p(new_score),
                    float(cand.get("logical_weight", 0.0)),

                    float(components.get(new_d, 0)),
                    float(components.get(new_d + 1, 0)),
                    float(components.get(new_d + 2, 0)),

                    duplicate_add_count,
                    new_deg["num_edges"] - old_deg["num_edges"],
                    new_deg["max_check_weight"] - old_deg["max_check_weight"],
                    new_deg["avg_check_weight"] - old_deg["avg_check_weight"],
                    new_deg["max_var_degree"] - old_deg["max_var_degree"],
                    new_deg["avg_var_degree"] - old_deg["avg_var_degree"],
                ],
                dtype=np.float32,
            )

        return feats

    def _make_observation(self):
        self.candidates = self._generate_candidates()

        mask = np.zeros(self.K, dtype=np.float32)
        mask[: len(self.candidates)] = 1.0

        return {
            "global": self._global_features(),
            "candidates": self._candidate_features(self.candidates),
            "mask": mask,
        }
    
    def _choose_greedy_action(self):
        best_action = None
        best_reward = -float("inf")

        old_params = self.params
        old_score_info = self.score_info

        for i, cand in enumerate(self.candidates):
            reward = self._reward(
                old_score_info=old_score_info,
                new_score_info=cand["score_info"],
                old_params=old_params,
                new_params=cand["params"],
            )

            if reward > best_reward:
                best_reward = reward
                best_action = i

        return best_action

    def _reward(self, old_score_info, new_score_info, old_params, new_params):
        old_score = float(old_score_info["score"])
        new_score = float(new_score_info["score"])

        old_d = int(old_params["d_quantum"])
        new_d = int(new_params["d_quantum"])

        # Distance is primary.
        distance_reward = 5.0 * float(new_d - old_d)

        # Score is most meaningful when distance is unchanged.
        if new_d == old_d:
            score_reward = np.log1p(old_score) - np.log1p(new_score)
        elif new_d > old_d:
            # Do not punish the new score too much because score window changed.
            score_reward = -0.05 * np.log1p(new_score)
        else:
            score_reward = -5.0 * float(old_d - new_d)

        k_penalty = 0.0
        if int(new_params["k_quantum"]) != int(old_params["k_quantum"]):
            k_penalty += 20.0
        if int(new_params["rank_H"]) != int(old_params["rank_H"]):
            k_penalty += 20.0

        # Optional soft LDPC penalty. Set to 0.0 if you really do not care.
        old_deg = self._degree_summary_from_state(self.state)
        new_deg = self._degree_summary_from_state(self.candidates[0]["state"])  # don't use this line directly

        return float(distance_reward + score_reward - k_penalty)

    def step(self, action: int):
        if action < 0 or action >= len(self.candidates):
            return self._make_observation(), -10.0, True, {"invalid_action": True}

        old_params = self.params
        old_score_info = self.score_info

        chosen = self.candidates[action]

        self.state = chosen["state"]
        self.params = chosen["params"]
        self.score_info = chosen["score_info"]
        self.step_count += 1

        reward = self._reward(
            old_score_info=old_score_info,
            new_score_info=self.score_info,
            old_params=old_params,
            new_params=self.params,
        )

        done = self.step_count >= self.max_steps

        info = {
            "score": float(self.score_info["score"]),
            "weight_patterns": self.score_info.get("components", {}),
            "d_quantum": int(self.params["d_quantum"]),
            "k_quantum": int(self.params["k_quantum"]),
            "rank_H": int(self.params["rank_H"]),
            "edges_to_add": chosen["edges_to_add"],
            "edges_to_remove": chosen["edges_to_remove"],
            "logical_weight": chosen.get("logical_weight", -1),
        }

        obs = None if done else self._make_observation()
        return obs, reward, done, info
    
    def _degree_summary_from_state(self, state):
        H = np.asarray(tanner_graph_to_parity_check_matrix(state), dtype=np.uint8)
        row_w = H.sum(axis=1)
        col_w = H.sum(axis=0)

        return {
            "num_edges": float(H.sum()),
            "max_check_weight": float(row_w.max()) if row_w.size else 0.0,
            "avg_check_weight": float(row_w.mean()) if row_w.size else 0.0,
            "max_var_degree": float(col_w.max()) if col_w.size else 0.0,
            "avg_var_degree": float(col_w.mean()) if col_w.size else 0.0,
        }


    def _duplicate_add_count(self, cand):
        return float(sum(self.state.has_edge(*e) for e in cand["edges_to_add"]))