"""RL4CO model builder wrappers."""

from __future__ import annotations

from pathlib import Path

from dvrptw_bench.rl.penalized_cvrptw_env import PenalizedCVRPTWEnv
import torch
from rl4co.envs import CVRPTWEnv
from rl4co.models.zoo.am import AttentionModel, AttentionModelPolicy
from rl4co.utils.trainer import RL4COTrainer
from dvrptw_bench.data.rl4co_der_generator import DERSolomonCVRPTWGenerator, FamilySpec
from dvrptw_bench.data.der_solomon_generator import DERTimeWindowGenerator
from tensordict import TensorDict

from dvrptw_bench.common.typing import Route, Solution, VRPTWInstance as Instance


class RLModel:
    def __init__(
        self,
        device=None,
        env=None,
        policy=None,
        model=None,
        max_epochs: int = 100,
        batch_size: int = 512,
        train_data_size: int = 100_000,
        val_data_size: int = 10_000,
        lr: float = 1e-4,
        num_loc: int = 100,
        normalize_coords: bool = True,
    ):
        self.device = device
        self.model = model
        self.policy = policy
        self.env = env
        self.max_epochs = max_epochs
        self.normalize_coords = normalize_coords
        self.batch_size = batch_size
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.lr = lr
        self.num_loc = num_loc

        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        if self.env is None:
            self.env = CVRPTWEnv(generator_params={"num_loc": num_loc})

        if self.policy is None:
            self.policy = AttentionModelPolicy(env_name=self.env.name,embed_dim=128,num_encoder_layers=3, num_heads=8, ).to(self.device)

        if self.model is None:
            self.model = AttentionModel(
                self.env,
                self.policy,
                baseline="rollout",
                batch_size=self.batch_size,
                train_data_size=self.train_data_size,
                val_data_size=self.val_data_size,
                val_batch_size = 64, 
                test_batch_size = 64, 
                optimizer_kwargs={"lr": self.lr},
            )

        accelerator = "cpu"
        devices = 1
        if self.device.type == "cuda":
            accelerator = "gpu"
            devices = 1
        elif self.device.type == "mps":
            accelerator = "mps"
            devices = 1

        self.trainer = RL4COTrainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=None,
        )

    def train(self):
        self.trainer.fit(self.model)

    def save(self, path: str | Path) -> None:
        self.trainer.save_checkpoint(str(path))

    def load(self, path: str | Path) -> None:
        # IMPORTANT: load_from_checkpoint returns a model; assign it back.
        self.model = AttentionModel.load_from_checkpoint(
            str(path),
            env=self.env,
            policy=self.policy,
            weights_only=False,
            load_baseline=False,
            batch_size=self.batch_size,
            train_data_size=self.train_data_size,
            val_data_size=self.val_data_size,
            optimizer_kwargs={"lr": self.lr},
            strict=False
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _normalize_coord(coord: torch.Tensor) -> torch.Tensor:
        x, y = coord[:, 0], coord[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        # Avoid division by zero on degenerate coordinates
        if torch.isclose(x_range, torch.tensor(0.0, device=coord.device)):
            x_scaled = torch.zeros_like(x)
        else:
            x_scaled = (x - x_min) / x_range

        if torch.isclose(y_range, torch.tensor(0.0, device=coord.device)):
            y_scaled = torch.zeros_like(y)
        else:
            y_scaled = (y - y_min) / y_range

        return torch.stack([x_scaled, y_scaled], dim=1)

    def _instance_to_td(self, instance: Instance) -> TensorDict:
        coords = torch.tensor(
            [[instance.depot.x, instance.depot.y], *[[c.x, c.y] for c in instance.customers]],
            dtype=torch.float32,
        )
        if self.normalize_coords:
            coords = self._normalize_coord(coords)

        demand = torch.tensor([c.demand for c in instance.customers], dtype=torch.float32)
        capacity = float(instance.vehicle_capacity)

        durations = torch.tensor(
            [0.0, *[c.service_time for c in instance.customers]],
            dtype=torch.float32,
        )
        time_windows = torch.tensor(
            [
                [instance.depot.ready_time, instance.depot.due_time],
                *[[c.ready_time, c.due_time] for c in instance.customers],
            ],
            dtype=torch.float32,
        )

        td = TensorDict(
            {
                "depot": coords[0],
                "locs": coords[1:],
                # Keep this convention internally consistent
                "demand": demand / capacity,
                "capacity": torch.tensor(1.0, dtype=torch.float32),
                "time_windows": time_windows,
                "durations": durations,
            },
            batch_size=[],
        )
        return td.unsqueeze(0)

    def _decode_actions(self, actions: torch.Tensor, instance: Instance) -> list[Route]:
        action_seq = actions[0].detach().cpu().tolist()
        idx_to_customer_id = {idx + 1: c.id for idx, c in enumerate(instance.customers)}

        routes: list[Route] = []
        current: list[int] = []
        seen: set[int] = set()

        for token in action_seq:
            token = int(token)

            if token == 0:
                if current:
                    routes.append(Route(vehicle_id=len(routes), node_ids=current))
                    current = []
                continue

            customer_id = idx_to_customer_id.get(token)
            if customer_id is None:
                continue

            # Avoid duplicated customer IDs if decoding emits repeats
            if customer_id in seen:
                continue

            seen.add(customer_id)
            current.append(customer_id)

        if current:
            routes.append(Route(vehicle_id=len(routes), node_ids=current))

        if not routes:
            routes = [Route(vehicle_id=0, node_ids=[])]
        return routes

    def solve(
        self,
        instance: Instance,
        decode_type: str = "greedy",
        num_samples: int = 1,
        select_best: bool = False,
    ) -> Solution:
        device = self.device
        env = self.env
        policy = self.model.policy.to(device).eval()

        td = self._instance_to_td(instance).to(device)
        td_reset = env.reset(td)

        with torch.inference_mode():
            if decode_type == "sampling":
                out = policy(
                    td_reset.clone(),
                    env,
                    decode_type="sampling",
                    num_samples=num_samples,
                    select_best=select_best,
                )
            else:
                out = policy(td_reset.clone(), env, decode_type="greedy")

            # Evaluate on the same representation used for inference
            cost = -env.get_reward(td_reset, out["actions"], False).float().item()
            # Compute total distance from reward if using PenalizedCVRPTWEnv, otherwise use cost directly
            total_distance = cost if not isinstance(env, PenalizedCVRPTWEnv) else (cost) - (env.vehicle_penalty * PenalizedCVRPTWEnv.count_routes_from_actions(out["actions"]).float().item())

        routes = self._decode_actions(out["actions"], instance)
        return Solution(strategy="rl4co", routes=routes, total_distance=total_distance)


def build_attention_model(
    *,
    device=None,
    max_epochs: int = 100,
    batch_size: int = 512,
    train_data_size: int = 100_000,
    val_data_size: int = 10_000,
    lr: float = 1e-4,
    normalize_coords: bool = True,
    der_templates: dict | None = None,
    family_specs=None,
    der_seed: int = 123,
    vehicle_penalty: float = 50,
    env = None,
    num_loc: int = 100,
):
    env = None

    if der_templates is not None:
        if not der_templates:
            raise ValueError("der_templates was provided but is empty.")

        if family_specs is None:
            family_specs = [FamilySpec(name) for name in sorted(der_templates.keys())]
        else:
            family_specs = [fs for fs in family_specs if fs.name in der_templates]
            if not family_specs:
                available = ", ".join(sorted(der_templates.keys()))
                raise ValueError(
                    f"No family_specs match the provided der_templates. Available template families: {available}"
                )

        sample_template = next(iter(der_templates.values()))
        num_loc = sample_template.n_customers
        max_loc = max(
            max(float(t.depot.x), float(t.depot.y), *(max(float(c.x), float(c.y)) for c in t.customers))
            for t in der_templates.values()
        )
        max_time = max(float(t.depot.due_time) for t in der_templates.values())

        der_gen = DERTimeWindowGenerator(der_templates, seed=der_seed)
        generator = DERSolomonCVRPTWGenerator(
            der_generator=der_gen,
            family_specs=family_specs,
            seed=der_seed,
            num_loc=num_loc,
            max_loc=max_loc,
            max_time=max_time,
            normalize_coords=normalize_coords,
        )
        env = env if env is not None else PenalizedCVRPTWEnv(generator=generator, vehicle_penalty=vehicle_penalty)

    return RLModel(
        device=device,
        env=env,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        lr=lr,
        num_loc=num_loc,
        normalize_coords=normalize_coords,
    )
