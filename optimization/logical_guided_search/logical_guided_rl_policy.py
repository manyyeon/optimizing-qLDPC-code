import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateActorCritic(nn.Module):
    def __init__(self, global_dim: int, candidate_dim: int, hidden: int = 128):
        super().__init__()

        self.cand_encoder = nn.Sequential(
            nn.Linear(global_dim + candidate_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor_head = nn.Linear(hidden, 1)

        self.critic = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs, cand_obs, mask):
        """
        global_obs: [B, G]
        cand_obs:   [B, K, C]
        mask:       [B, K]
        """
        B, K, _ = cand_obs.shape

        g = global_obs.unsqueeze(1).expand(B, K, global_obs.shape[-1])
        x = torch.cat([g, cand_obs], dim=-1)

        h = self.cand_encoder(x)
        logits = self.actor_head(h).squeeze(-1)

        logits = logits.masked_fill(mask <= 0.0, -1e9)

        value = self.critic(global_obs).squeeze(-1)

        return logits, value

    def act(self, obs):
        device = next(self.parameters()).device

        global_obs = torch.tensor(obs["global"], dtype=torch.float32, device=device).unsqueeze(0)
        cand_obs = torch.tensor(obs["candidates"], dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(obs["mask"], dtype=torch.float32, device=device).unsqueeze(0)

        logits, value = self.forward(global_obs, cand_obs, mask)
        dist = torch.distributions.Categorical(logits=logits)

        print(f"Action probabilities: {dist.probs.cpu().detach().numpy()}")

        action = dist.sample()
        logprob = dist.log_prob(action)

        return (
            int(action.item()),
            float(logprob.item()),
            float(value.item()),
        )