from __future__ import annotations
from pathlib import Path
from typing import Literal

import torch
from rl4co.envs import CVRPTWEnv
from rl4co.models.zoo.am import AttentionModel, AttentionModelPolicy
from rl4co.utils.trainer import RL4COTrainer


Family = Literal["C1", "C2", "R1", "R2", "RC1", "RC2"]

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
        self.batch_size = batch_size
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.lr = lr
        self.num_loc = num_loc
        self.normalize_coords = normalize_coords

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
            self.policy = AttentionModelPolicy(
                env_name=self.env.name,
                embed_dim=128,
                num_encoder_layers=3,
                num_heads=8,
            ).to(self.device)

        if self.model is None:
            self.model = AttentionModel(
                self.env,
                self.policy,
                baseline="rollout",
                batch_size=self.batch_size,
                train_data_size=self.train_data_size,
                val_data_size=self.val_data_size,
                val_batch_size=64,
                test_batch_size=64,
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

    def train(self) -> None:
        self.trainer.fit(self.model)

    def save(self, path: str | Path) -> None:
        self.trainer.save_checkpoint(str(path))


def build_attention_model(
    *,
    device=None,
    max_epochs: int = 100,
    batch_size: int = 512,
    train_data_size: int = 100_000,
    val_data_size: int = 10_000,
    lr: float = 1e-4,
    num_loc: int = 100,
):
    return RLModel(
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        lr=lr,
        num_loc=num_loc,
        normalize_coords=False,
    )


def train_one_configuration(
    output_dir: Path,
    num_customers: int,
    learning_rate: float,
    penalty: float,
    seed: int,
    max_epochs: int,
    batch_size: int,
    train_data_size: int,
    val_data_size: int,
    family: Family,
    penalized: bool,
) -> Path:
    model = build_attention_model(
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        lr=learning_rate,
        num_loc=num_customers,
    )
    model.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "penalized" if penalized else "non_penalized"
    ckpt_path = output_dir / (
        f"attention-model-{suffix}-{family}-customers-{num_customers}-"
        f"penalty-{penalty}-lr-{learning_rate}-seed-{seed}.ckpt"
    )
    model.save(ckpt_path)
    return ckpt_path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (SCRIPT_DIR / "trained_models").resolve()

NUM_CUSTOMERS = [25,50]
LEARNING_RATES = [1e-4, 5e-4]
SEEDS = [6238116, 7960100, 7454245]
MAX_EPOCHS = 100
BATCH_SIZE = 512
TRAIN_DATA_SIZE = 100_000
VAL_DATA_SIZE = 10_000
RUN_MODES = [True]

if __name__ == "__main__":
    for num_customers in NUM_CUSTOMERS:
        for learning_rate in LEARNING_RATES:
            for seed in SEEDS:
                for penalized in RUN_MODES:
                    print(
                        f"Training model: customers={num_customers}, lr={learning_rate}, "
                        f"seed={seed}"
                    )
                    ckpt_path = train_one_configuration(
                        output_dir=OUTPUT_DIR,
                        num_customers=num_customers,
                        learning_rate=learning_rate,
                        penalty=0,
                        seed=seed,
                        max_epochs=MAX_EPOCHS,
                        batch_size=BATCH_SIZE,
                        train_data_size=TRAIN_DATA_SIZE,
                        val_data_size=VAL_DATA_SIZE,
                        family="RC1",
                        penalized=penalized,
                    )
                    print(f"Saved checkpoint to {ckpt_path}")
