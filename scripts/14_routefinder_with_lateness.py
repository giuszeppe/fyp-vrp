import sys
import time
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from rl4co.utils.trainer import RL4COTrainer

sys.path.append(str(Path("..").resolve() / "src"))

from dvrptw_bench.data.instance_filters import find_rc_instances
from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.heuristics.ortools_solver import ORToolsVRPTWSolver
from dvrptw_bench.metrics.objective import total_distance
from dvrptw_bench.rl.env_flexible import MTVRPFlexibleEnv
from dvrptw_bench.rl.routefinder_adapter import (
    instance_to_routefinder_td,
    routefinder_actions_to_solution,
)
from routefinder.envs.mtvrp import MTVRPGenerator
from routefinder.models import RouteFinderBase, RouteFinderPolicy
from routefinder.utils import evaluate as evaluate_routefinder
from routefinder.utils import rollout, greedy_policy, evaluate
from routefinder.baselines.solve import solve


def optional_int(value: str):
    lowered = str(value).strip().lower()
    if lowered in {"none", "all"}:
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RouteFinder on generated MTVRP instances and evaluate on Solomon RC instances."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../dataset/solomon_rc100"),
        help="Root directory for Solomon RC dataset.",
    )
    parser.add_argument(
        "--rc-dataset-root",
        type=Path,
        default=Path("../dataset/solomon_rc100"),
        help="RC dataset root (kept for parity with notebook constants).",
    )
    parser.add_argument(
        "--c-dataset-root",
        type=Path,
        default=Path("../dataset/solomon_c100"),
        help="C dataset root (kept for parity with notebook constants).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../outputs/notebook_routefinder_solomon_generated"),
        help="Directory where CSV/plots/checkpoints are saved.",
    )

    parser.add_argument(
        "--num-customers",
        type=int,
        default=10,
        help="Number of customers per instance.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--train-data-size",
        type=int,
        default=1000,
        help="Generated training dataset size.",
    )
    parser.add_argument(
        "--val-data-size",
        type=int,
        default=100,
        help="Generated validation dataset size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--num-augment",
        type=int,
        default=8,
        help="Number of augmentations during RouteFinder evaluation.",
    )
    parser.add_argument(
        "--ortools-time-limit-s",
        type=float,
        default=0.1,
        help="OR-Tools time limit in seconds.",
    )
    parser.add_argument(
        "--max-eval-instances",
        type=optional_int,
        default=5,
        help="Maximum number of RC instances to evaluate. Use 'none' or 'all' for no limit.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=10.11,
        help="MTVRP generator max_time parameter.",
    )
    parser.add_argument(
        "--num-rollout-test-instances",
        type=int,
        default=50,
        help="Number of generated instances for the rollout/evaluate demo block.",
    )
    parser.add_argument(
        "--pyvrp-max-runtime",
        type=float,
        default=3.0,
        help="Max runtime for PyVRP baseline in demo block.",
    )
    parser.add_argument(
        "--pyvrp-num-procs",
        type=int,
        default=32,
        help="Number of processes for PyVRP baseline in demo block.",
    )

    parser.add_argument(
        "--normalize-coords",
        dest="normalize_coords",
        action="store_true",
        default=True,
        help="Normalize coordinates to [0, 1] when creating RouteFinder tensors.",
    )
    parser.add_argument(
        "--no-normalize-coords",
        dest="normalize_coords",
        action="store_false",
        help="Disable coordinate normalization.",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and assume model is already initialized.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Solomon evaluation block.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip saving/showing matplotlib plots.",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip rollout/evaluate/PyVRP demo block.",
    )

    return parser.parse_args()


def gap(sol, bks):
    return ((sol - bks) / bks).mean() * 100


def summarize_td(td, name):
    tw = td["time_windows"][0].cpu()
    st = td["service_time"][0].cpu()

    starts = tw[:, 0]
    ends = tw[:, 1]
    widths = ends - starts

    print(f"\n{name}")
    print("depot due:", ends[0].item())
    print("start  min/mean/max:", starts.min().item(), starts.mean().item(), starts.max().item())
    print("end    min/mean/max:", ends.min().item(), ends.mean().item(), ends.max().item())
    print("width  min/mean/max:", widths.min().item(), widths.mean().item(), widths.max().item())
    print("service min/mean/max:", st.min().item(), st.mean().item(), st.max().item())


def main():
    args = parse_args()

    dataset_root = args.dataset_root
    rc_dataset_root = args.rc_dataset_root
    c_dataset_root = args.c_dataset_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "gpu"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
    else:
        device = torch.device("cpu")
        accelerator = "cpu"

    print("Device:", device)
    print("Dataset root:", dataset_root.resolve())
    print("RC dataset root:", rc_dataset_root.resolve())
    print("C dataset root:", c_dataset_root.resolve())
    print("Output root:", output_root.resolve())

    generator = MTVRPGenerator(
        num_loc=args.num_customers,
        max_time=args.max_time,
        variant_preset="vrptw",
    )
    env = MTVRPFlexibleEnv(generator, check_solution=False)
    policy = RouteFinderPolicy(env_name=env.name).to(device)
    model = RouteFinderBase(
        env,
        policy,
        batch_size=args.batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        optimizer_kwargs={
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    )

    trainer = RL4COTrainer(
        max_epochs=args.num_epochs,
        accelerator=accelerator,
        devices=1,
        logger=None,
        num_sanity_val_steps=0,
        precision="32-true",
    )

    if not args.skip_training:
        trainer.fit(model)
        # trainer.save_checkpoint(output_root / "end_of_training.pt")
        # model = model.load_from_checkpoint(
        #     output_root / "checkpoints/last.ckpt",
        #     weights_only=False,
        #     map_location=torch.device("cpu"),
        # )

    if not args.skip_eval:
        instance_paths = find_rc_instances(dataset_root)
        assert instance_paths, f"No RC instances found under {dataset_root.resolve()}"

        if args.max_eval_instances is not None:
            instance_paths = instance_paths[: args.max_eval_instances]

        instances = [
            parse_solomon(path, max_customers=args.num_customers)
            for path in instance_paths
        ]
        print("Loaded instances:", [instance.instance_id for instance in instances])

        def solve_with_routefinder(instance, num_augment=args.num_augment):
            t0 = time.perf_counter()
            td = instance_to_routefinder_td(
                instance,
                normalize_coords=args.normalize_coords,
            ).to(device)
            td_reset = env.reset(td)
            model.to(device).eval()

            with torch.inference_mode():
                if num_augment > 1:
                    out = evaluate_routefinder(
                        model,
                        td_reset.clone(),
                        num_augment=num_augment,
                    )
                    actions = out.get(
                        "best_aug_actions",
                        out.get("best_multistart_actions", out.get("actions")),
                    )
                else:
                    out = model.policy(
                        td_reset.clone(),
                        env,
                        phase="test",
                        decode_type="greedy",
                        return_actions=True,
                    )
                    actions = out["actions"]

            solution = routefinder_actions_to_solution(
                actions,
                instance,
                strategy="routefinder",
            )
            solution.total_distance = total_distance(instance, solution)
            solution.solve_time_s = time.perf_counter() - t0
            solution.details.update({"num_augment": num_augment})
            return solution

        ortools = ORToolsVRPTWSolver()
        routefinder_solutions = {}
        ortools_solutions = {}
        rows = []

        for instance in instances:
            rf_solution = solve_with_routefinder(instance)
            or_solution = ortools.solve(
                instance,
                time_limit_s=args.ortools_time_limit_s,
            )

            routefinder_solutions[instance.instance_id] = rf_solution
            ortools_solutions[instance.instance_id] = or_solution

            gap_pct = (
                100.0
                * (rf_solution.total_distance - or_solution.total_distance)
                / or_solution.total_distance
            )
            rows.append(
                {
                    "instance_id": instance.instance_id,
                    "n_customers": instance.n_customers,
                    "routefinder_distance": rf_solution.total_distance,
                    "ortools_distance": or_solution.total_distance,
                    "gap_to_ortools_pct": gap_pct,
                    "routefinder_routes": len(
                        [r for r in rf_solution.routes if r.node_ids]
                    ),
                    "ortools_routes": len(
                        [r for r in or_solution.routes if r.node_ids]
                    ),
                    "routefinder_time_s": rf_solution.solve_time_s,
                    "ortools_time_s": or_solution.solve_time_s,
                }
            )

        results_df = (
            pd.DataFrame(rows)
            .sort_values("gap_to_ortools_pct")
            .reset_index(drop=True)
        )
        results_df[
            [
                "routefinder_distance",
                "ortools_distance",
                "gap_to_ortools_pct",
                "routefinder_time_s",
                "ortools_time_s",
            ]
        ] = results_df[
            [
                "routefinder_distance",
                "ortools_distance",
                "gap_to_ortools_pct",
                "routefinder_time_s",
                "ortools_time_s",
            ]
        ].round(3)

        print(results_df.to_string(index=False))
        results_df.to_csv(output_root / "routefinder_vs_ortools.csv", index=False)
        print("Saved table to", output_root / "routefinder_vs_ortools.csv")

        if not args.skip_plots:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            plot_df = results_df.set_index("instance_id")
            plot_df[["routefinder_distance", "ortools_distance"]].plot(
                kind="bar",
                ax=axes[0],
            )
            axes[0].set_title("RouteFinder vs OR-Tools Distance")
            axes[0].set_ylabel("Total distance")
            axes[0].grid(axis="y", alpha=0.3)

            plot_df["gap_to_ortools_pct"].plot(kind="bar", ax=axes[1], color="tab:orange")
            axes[1].axhline(0.0, color="black", linewidth=1)
            axes[1].set_title("RouteFinder Gap vs OR-Tools")
            axes[1].set_ylabel("Gap (%)")
            axes[1].grid(axis="y", alpha=0.3)

            fig.tight_layout()
            fig.savefig(output_root / "routefinder_vs_ortools_summary.png", dpi=180)
            plt.show()
            print(
                "Saved summary plot to",
                output_root / "routefinder_vs_ortools_summary.png",
            )

        if not args.skip_plots:
            from dvrptw_bench.viz.route_plot import plot_routes

            for instance in instances[: min(3, len(instances))]:
                print(f"\n=== {instance.instance_id} ===")
                print(
                    "RouteFinder distance:",
                    routefinder_solutions[instance.instance_id].total_distance,
                )
                print(
                    "OR-Tools distance:",
                    ortools_solutions[instance.instance_id].total_distance,
                )
                plot_routes(instance, routefinder_solutions[instance.instance_id])
                plot_routes(instance, ortools_solutions[instance.instance_id])

        print(env.generator(1)["time_windows"][0][:10])
        print(
            instance_to_routefinder_td(
                instances[0],
                normalize_coords=args.normalize_coords,
            )["time_windows"][0][:10]
        )

        summarize_td(env.generator(1), "train generator")
        summarize_td(
            instance_to_routefinder_td(
                instances[0],
                normalize_coords=args.normalize_coords,
            ),
            "solomon inference",
        )

    if not args.skip_demo:
        td_data = env.generator(args.num_rollout_test_instances)
        variant_names = env.get_variant_names(td_data)

        td_test = env.reset(td_data)
        actions = rollout(env, td_test.clone(), greedy_policy)
        rewards_nearest_neighbor = env.get_reward(td_test, actions)

        print(f"Averaged cost: {-rewards_nearest_neighbor.mean():.3f}")

        for idx in [0, 1, 2]:
            env.render(td_test[idx], actions[idx])
            print("Cost: ", -rewards_nearest_neighbor[idx].item())
            print("Problem: ", variant_names[idx])

        td_test = td_test.to(device)
        model = model.to(device)
        out = evaluate(model, td_test.clone())

        actions = out["best_aug_actions"]
        rewards = env.get_reward(td_test, actions)

        print(f"Averaged cost: {-rewards.mean():.3f}")

        for i in range(3):
            print(f"Problem {i + 1} | Cost: {-rewards[i]:.3f}")
            print("Variant", env.get_variant_names(td_test[i]))
            env.render(td_test[i].cpu(), actions[i].cpu())

        td_test = td_test.cpu()
        actions_pyvrp, costs_pyvrp = solve(
            td_test,
            max_runtime=args.pyvrp_max_runtime,
            num_procs=args.pyvrp_num_procs,
            solver="pyvrp",
        )
        rewards_pyvrp = env.get_reward(td_test.clone().cpu(), actions_pyvrp)

        print(f"Averaged cost PyVRP: {-rewards_pyvrp.mean():.3f}")
        print(
            f"Nearest Neighbor gap to HGS-PyVRP: "
            f"{gap(rewards_nearest_neighbor.cpu(), rewards_pyvrp.cpu()):.3f}%"
        )
        print(
            f"RouteFinder gap to HGS-PyVRP: "
            f"{gap(rewards.cpu(), rewards_pyvrp.cpu()):.3f}%"
        )


if __name__ == "__main__":
    main()