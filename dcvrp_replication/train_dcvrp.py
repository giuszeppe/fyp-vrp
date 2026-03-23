from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.utils.ops import gather_by_index
from rl4co.utils.trainer import RL4COTrainer

from dcvrp_env import DCVRPEnv, DCVRPGenerator


class DCVRPInitEmbedding(nn.Module):
    """Paper-style node embedding.

    Undisclosed customers are replaced by depot coordinates with zero demand,
    then projected into the node embedding space.
    """

    def __init__(self, embed_dim: int, linear_bias: bool = True):
        super().__init__()
        self.init_embed = nn.Linear(3, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict):
        depot = td["depot"]                        # (B, 1, 2)
        locs = td["locs"]                          # (B, N, 2)
        demands = td["demands"]                    # (B, N)
        disclosure_times = td["disclosure_times"]  # (B, N)
        current_time = td["current_time"]          # (B, 1)

        disclosed = disclosure_times <= current_time
        depot_xy = depot.expand(-1, locs.size(1), -1)
        dyn_locs = torch.where(disclosed.unsqueeze(-1), locs, depot_xy)
        dyn_demands = torch.where(disclosed, demands, torch.zeros_like(demands))

        depot_feat = torch.cat(
            [depot, torch.zeros(depot.size(0), 1, 1, device=depot.device)], dim=-1
        )
        cust_feat = torch.cat([dyn_locs, dyn_demands.unsqueeze(-1)], dim=-1)
        feats = torch.cat([depot_feat, cust_feat], dim=1)
        return self.init_embed(feats)


class DCVRPContextEmbedding(nn.Module):
    """Decoder context: current node embedding + remaining capacity + current time."""

    def __init__(self, embed_dim: int, linear_bias: bool = True):
        super().__init__()
        self.W_placeholder = nn.Parameter(torch.Tensor(embed_dim + 2).uniform_(-1, 1))
        self.project_context = nn.Linear(embed_dim + 2, embed_dim, bias=linear_bias)

    def forward(self, embeddings: torch.Tensor, td: TensorDict):
        batch_size = embeddings.size(0)
        current_emb = gather_by_index(embeddings, td["current_node"])
        if current_emb.dim() == 3:
            current_emb = current_emb.squeeze(1)
        remaining_capacity = 1.0 - td["used_capacity"].float()
        cur_time = td["current_time"].float()
        context = torch.cat([current_emb, remaining_capacity, cur_time], dim=-1)

        first_step = td["i"][(0,) * td["i"].dim()].item() < 1
        if first_step:
            context = self.W_placeholder[None, :].expand(batch_size, -1)
        return self.project_context(context)


class DCVRPDynamicEmbedding(nn.Module):
    """Update decoder keys/values when disclosure state changes."""

    def __init__(self, embed_dim: int, linear_bias: bool = True):
        super().__init__()
        self.project_node_step = nn.Linear(2, 3 * embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict):
        disclosed = (td["disclosure_times"] <= td["current_time"]).float()
        served = td["visited"].float()
        x = torch.stack([disclosed, served], dim=-1)
        kv = self.project_node_step(x)
        glimpse_key_dyn, glimpse_val_dyn, logit_key_dyn = kv.chunk(3, dim=-1)

        # Prepend a zero row for depot since the stacked customer state excludes depot.
        zeros = torch.zeros(
            td.batch_size[0], 1, glimpse_key_dyn.size(-1),
            device=glimpse_key_dyn.device,
            dtype=glimpse_key_dyn.dtype,
        )
        glimpse_key_dyn = torch.cat([zeros, glimpse_key_dyn], dim=1)
        glimpse_val_dyn = torch.cat([zeros, glimpse_val_dyn], dim=1)
        logit_key_dyn = torch.cat([zeros, logit_key_dyn], dim=1)
        return glimpse_key_dyn, glimpse_val_dyn, logit_key_dyn


def build_model(env: DCVRPEnv, embed_dim: int, num_encoder_layers: int, num_heads: int):
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=embed_dim,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        init_embedding=DCVRPInitEmbedding(embed_dim),
        context_embedding=DCVRPContextEmbedding(embed_dim),
        dynamic_embedding=DCVRPDynamicEmbedding(embed_dim),
    )
    model = REINFORCE(
        env,
        policy,
        baseline="rollout",
        batch_size=512,
        train_data_size=100_000,
        val_data_size=10_000,
        optimizer_kwargs={"lr": 1e-4},
    )
    return model


def get_default_horizon(num_loc: int) -> float:
    # Paper values for {10, 20, 50}; smooth fallback otherwise.
    if num_loc == 10:
        return 3.8
    if num_loc == 20:
        return 6.4
    if num_loc == 50:
        return 10.98
    return 1.6 * math.sqrt(num_loc)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL4CO Attention Model on DCVRP")
    parser.add_argument("--num-loc", type=int, default=20)
    parser.add_argument("--dynamic-ratio", type=float, default=0.5)
    parser.add_argument("--vehicle-capacity", type=int, default=30)
    parser.add_argument("--max-disclosure-time", type=float, default=None)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    max_disclosure_time = args.max_disclosure_time
    if max_disclosure_time is None:
        max_disclosure_time = get_default_horizon(args.num_loc)

    generator = DCVRPGenerator(
        num_loc=args.num_loc,
        dynamic_ratio=args.dynamic_ratio,
        vehicle_capacity=args.vehicle_capacity,
        max_disclosure_time=max_disclosure_time,
    )
    env = DCVRPEnv(generator=generator)
    model = build_model(
        env,
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
    )

    trainer = RL4COTrainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        gradient_clip_val=1.0,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
