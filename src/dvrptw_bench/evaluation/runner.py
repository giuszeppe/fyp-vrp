"""Scenario generation, resumable ledgers, and modality execution."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Any, Iterable

from dvrptw_bench.data.solomon_parser import parse_solomon
from dvrptw_bench.dynamic.arrivals import build_dynamic_scenario
from dvrptw_bench.dynamic.simulator import DynamicSimulator
from dvrptw_bench.evaluation.compat import AI_MODEL_SPECS, create_model, result_from_solution
from dvrptw_bench.evaluation.models import Ledger, Manifest, ModelSpec, ScenarioArtifact, WorkState, WorkUnit
from dvrptw_bench.evaluation.storage import atomic_write_json, ensure_dir, read_json


def project_root_from(start: Path) -> Path:
    for current in [start, *start.parents]:
        if (current / "pyproject.toml").exists():
            return current
    return start


@dataclass(frozen=True)
class EvaluationPaths:
    root: Path
    manifest: Path
    models: Path
    instances_root: Path
    scenarios_root: Path
    results_root: Path
    state_root: Path


def build_paths(data_root: Path) -> EvaluationPaths:
    config_root = data_root / "config"
    instances_root = data_root / "instances"
    return EvaluationPaths(
        root=data_root,
        manifest=config_root / "manifest.json",
        models=config_root / "models.json",
        instances_root=instances_root / "base",
        scenarios_root=instances_root / "generated",
        results_root=data_root / "results",
        state_root=data_root / "state",
    )


def _iter_solomon_instance_files(dataset_dir: Path) -> list[Path]:
    instance_files: list[Path] = []
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".txt":
            stem = path.stem.upper()
            if stem.startswith("C") or stem.startswith("RC") or stem.startswith("R"):
                instance_files.append(path)
    return sorted(instance_files)


def _slug_float(value: float) -> str:
    return str(value).replace(".", "p")


def _scenario_id(instance_name: str, size: int, seed: int, dod: float, cutoff: float) -> str:
    return f"{Path(instance_name).stem}_n{size}_seed{seed}_dod{_slug_float(dod)}_cut{_slug_float(cutoff)}"


def _result_file_name(work: WorkUnit) -> str:
    parts = [work.instance_name.replace(".txt", ""), f"n{work.evaluation_size}", work.model_name.replace(":", "_")]
    if work.seed is not None:
        parts.append(f"seed{work.seed}")
    if work.degree_of_dynamicity is not None:
        parts.append(f"dod{_slug_float(work.degree_of_dynamicity)}")
    if work.cutoff_time is not None:
        parts.append(f"cut{_slug_float(work.cutoff_time)}")
    return "__".join(parts) + ".json"


def init_workspace(
    *,
    data_root: Path,
    dataset_root: Path,
    seeds: list[int],
    evaluation_sizes: list[int],
    degrees_of_dynamicity: list[float],
    cutoff_times: list[float],
) -> Manifest:
    paths = build_paths(data_root)
    ensure_dir(paths.instances_root)
    ensure_dir(paths.scenarios_root)
    ensure_dir(paths.results_root)
    ensure_dir(paths.state_root)
    ensure_dir(paths.manifest.parent)

    instances = [path.name for path in _iter_solomon_instance_files(dataset_root)]
    manifest = Manifest(
        dataset_root=str(dataset_root.resolve()),
        seeds=sorted(dict.fromkeys(seeds)),
        evaluation_sizes=evaluation_sizes,
        degrees_of_dynamicity=degrees_of_dynamicity,
        cutoff_times=cutoff_times,
        instances=instances,
        updated_at=datetime.utcnow(),
    )
    atomic_write_json(paths.manifest, manifest.model_dump(mode="json"))

    if not paths.models.exists():
        default_models = [
            ModelSpec(model_id=model_id, **spec).model_dump(mode="json")
            for model_id, spec in AI_MODEL_SPECS.items()
        ]
        atomic_write_json(paths.models, default_models)

    for instance_name in instances:
        source = dataset_root / instance_name
        target = paths.instances_root / instance_name
        if not target.exists():
            target.write_text(source.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    return manifest


def load_manifest(data_root: Path) -> Manifest:
    paths = build_paths(data_root)
    data = read_json(paths.manifest, None)
    if data is None:
        raise FileNotFoundError(f"Manifest not found at {paths.manifest}")
    return Manifest.model_validate(data)


def load_model_registry(data_root: Path) -> dict[str, dict[str, Any]]:
    paths = build_paths(data_root)
    data = read_json(paths.models, [])
    models = [ModelSpec.model_validate(item) for item in data]
    return {model.model_id: model.model_dump(mode="json") for model in models if model.enabled}


def generate_scenarios(data_root: Path, seeds: list[int] | None = None) -> list[ScenarioArtifact]:
    manifest = load_manifest(data_root)
    paths = build_paths(data_root)
    scenario_records: list[ScenarioArtifact] = []
    active_seeds = seeds or manifest.seeds
    dataset_root = Path(manifest.dataset_root)
    for instance_name in manifest.instances:
        source_path = dataset_root / instance_name
        for size in manifest.evaluation_sizes:
            base_instance = parse_solomon(source_path, max_customers=size)
            for seed in active_seeds:
                for dod in manifest.degrees_of_dynamicity:
                    for cutoff in manifest.cutoff_times:
                        scenario = build_dynamic_scenario(
                            base_instance,
                            epsilon=dod,
                            seed=seed,
                            cutoff_ratio=cutoff,
                        )
                        record = ScenarioArtifact(
                            scenario_id=_scenario_id(instance_name, size, seed, dod, cutoff),
                            instance_name=instance_name,
                            evaluation_size=size,
                            seed=seed,
                            degree_of_dynamicity=dod,
                            cutoff_time=cutoff,
                            source_path=str(source_path.resolve()),
                            dynamic_customer_ids=sorted(scenario.dynamic_customer_ids),
                            reveal_times=scenario.reveal_times,
                            feasible=scenario.feasible,
                            dropped_reason=scenario.dropped_reason,
                            instance=scenario.instance.model_dump(mode="json"),
                        )
                        scenario_path = paths.scenarios_root / f"{record.scenario_id}.json"
                        atomic_write_json(scenario_path, record.model_dump(mode="json"))
                        scenario_records.append(record)
    return scenario_records


def _scenario_files(paths: EvaluationPaths) -> list[Path]:
    return sorted(paths.scenarios_root.glob("*.json"))


def _load_scenarios(paths: EvaluationPaths) -> list[ScenarioArtifact]:
    return [ScenarioArtifact.model_validate(read_json(path, {})) for path in _scenario_files(paths)]


def _ledger_path(paths: EvaluationPaths, modality: str) -> Path:
    return paths.state_root / f"{modality}_ledger.json"


def load_ledger(paths: EvaluationPaths, modality: str) -> Ledger:
    data = read_json(_ledger_path(paths, modality), None)
    if data is None:
        return Ledger(modality=modality)
    return Ledger.model_validate(data)


def save_ledger(paths: EvaluationPaths, ledger: Ledger) -> None:
    ledger.updated_at = datetime.utcnow().isoformat()
    atomic_write_json(_ledger_path(paths, ledger.modality), ledger.model_dump(mode="json"))


def _lock_path(paths: EvaluationPaths, modality: str) -> Path:
    return paths.state_root / f"{modality}.lock"


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def modality_lock(paths: EvaluationPaths, modality: str):
    lock_path = _lock_path(paths, modality)
    payload = {
        "pid": os.getpid(),
        "modality": modality,
        "acquired_at": datetime.utcnow().isoformat(),
    }
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = read_json(lock_path, {})
            existing_pid = int(existing.get("pid", -1)) if isinstance(existing, dict) else -1
            if _is_pid_running(existing_pid):
                raise RuntimeError(
                    f"Another '{modality}' run is already active for this data root "
                    f"(pid {existing_pid})."
                )
            lock_path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            import json

            handle.write(json.dumps(payload, indent=2, sort_keys=True))
        break
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _oracle_work_units(paths: EvaluationPaths, manifest: Manifest) -> list[WorkUnit]:
    units: list[WorkUnit] = []
    for instance_name in manifest.instances:
        for size in manifest.evaluation_sizes:
            unit = WorkUnit(
                work_id=f"oracle::{Path(instance_name).stem}::n{size}",
                modality="oracle",
                instance_name=instance_name,
                evaluation_size=size,
                model_name="oracle",
                result_path=str(paths.results_root / "oracle" / _result_file_name(WorkUnit(
                    work_id="tmp",
                    modality="oracle",
                    instance_name=instance_name,
                    evaluation_size=size,
                    model_name="oracle",
                    result_path="",
                ))),
            )
            units.append(unit)
    return units


def _scenario_work_units(paths: EvaluationPaths, modality: str, model_names: Iterable[str]) -> list[WorkUnit]:
    units: list[WorkUnit] = []
    for scenario in _load_scenarios(paths):
        if not scenario.feasible:
            continue
        for model_name in model_names:
            unit = WorkUnit(
                work_id=f"{modality}::{scenario.scenario_id}::{model_name}",
                modality=modality,
                instance_name=scenario.instance_name,
                evaluation_size=scenario.evaluation_size,
                seed=scenario.seed,
                degree_of_dynamicity=scenario.degree_of_dynamicity,
                cutoff_time=scenario.cutoff_time,
                model_name=model_name,
                scenario_id=scenario.scenario_id,
                result_path=str(paths.results_root / modality / _result_file_name(WorkUnit(
                    work_id="tmp",
                    modality=modality,
                    instance_name=scenario.instance_name,
                    evaluation_size=scenario.evaluation_size,
                    seed=scenario.seed,
                    degree_of_dynamicity=scenario.degree_of_dynamicity,
                    cutoff_time=scenario.cutoff_time,
                    model_name=model_name,
                    scenario_id=scenario.scenario_id,
                    result_path="",
                ))),
            )
            units.append(unit)
    return units


def enumerate_work_units(data_root: Path, modality: str) -> list[WorkUnit]:
    paths = build_paths(data_root)
    manifest = load_manifest(data_root)
    models = load_model_registry(data_root)
    if modality == "oracle":
        return _oracle_work_units(paths, manifest)
    if modality == "heuristic":
        return _scenario_work_units(paths, modality, ["ortools"])
    if modality == "ai":
        return _scenario_work_units(paths, modality, sorted(models.keys()))
    if modality == "hybrid":
        return _scenario_work_units(paths, modality, [f"hybrid:{name}" for name in sorted(models.keys())])
    raise ValueError(f"Unknown modality: {modality}")


def sync_ledger(data_root: Path, modality: str) -> Ledger:
    paths = build_paths(data_root)
    ledger = load_ledger(paths, modality)
    for unit in enumerate_work_units(data_root, modality):
        ledger.items.setdefault(unit.work_id, WorkState(work_id=unit.work_id))
    save_ledger(paths, ledger)
    return ledger


def _load_scenario_instance(paths: EvaluationPaths, scenario_id: str):
    record = ScenarioArtifact.model_validate(read_json(paths.scenarios_root / f"{scenario_id}.json", {}))
    from dvrptw_bench.common.typing import VRPTWInstance

    return VRPTWInstance.model_validate(record.instance)


def _lookup_oracle_cost(paths: EvaluationPaths, unit: WorkUnit) -> float | None:
    oracle_file = paths.results_root / "oracle" / f"{Path(unit.instance_name).stem}__n{unit.evaluation_size}__oracle.json"
    if not oracle_file.exists():
        return None
    payload = read_json(oracle_file, {})
    return payload.get("total_cost")


def _run_oracle_unit(paths: EvaluationPaths, project_root: Path, unit: WorkUnit, models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    oracle_budget_s = 10.0
    instance = parse_solomon(Path(load_manifest(paths.root).dataset_root) / unit.instance_name, max_customers=unit.evaluation_size)
    model_type, solver = create_model("oracle", project_root, models)
    solution = solver.solve(instance, time_limit_s=oracle_budget_s, warm_start=None)
    result = result_from_solution(
        instance=instance,
        model_name="oracle",
        model_type=model_type,
        evaluation_size=unit.evaluation_size,
        degree_of_dynamicity=0.0,
        cutoff_time=1.0,
        run_id=0,
        solution=solution,
        oracle_cost=None,
    ).to_dict()
    return result


def _run_dynamic_unit(paths: EvaluationPaths, project_root: Path, unit: WorkUnit, models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    instance = _load_scenario_instance(paths, unit.scenario_id or "")
    model_type, solver = create_model(unit.model_name, project_root, models)
    simulator = DynamicSimulator(instance)
    if unit.modality == "heuristic":
        call_index = {"value": 0}

        def solver_fn(instance, time_limit_s, warm_start=None):
            _ = (time_limit_s, warm_start)
            budget = 30.0 if call_index["value"] == 0 else 5.0
            call_index["value"] += 1
            return solver.solve(instance, time_limit_s=budget, warm_start=warm_start)

        budget = 5.0
    elif unit.modality == "hybrid":
        def solver_fn(instance, time_limit_s, warm_start=None):
            _ = warm_start
            return solver.solve(instance, time_limit_s=0.5, warm_start=None)

        budget = 0.5
    else:
        def solver_fn(instance, time_limit_s, warm_start=None):
            return solver.solve(instance, time_limit_s=time_limit_s, warm_start=warm_start)

        budget = 5.0
    final_solution, _events, _scenario = simulator.run(
        solver_fn=solver_fn,
        epsilon=unit.degree_of_dynamicity or 0.0,
        budget_s=budget,
        seed=unit.seed or 0,
        cutoff_ratio=unit.cutoff_time or 0.5,
    )
    result = result_from_solution(
        instance=instance,
        model_name=unit.model_name,
        model_type=model_type,
        evaluation_size=unit.evaluation_size,
        degree_of_dynamicity=unit.degree_of_dynamicity or 0.0,
        cutoff_time=unit.cutoff_time or 0.5,
        run_id=0,
        solution=final_solution,
        oracle_cost=_lookup_oracle_cost(paths, unit),
    ).to_dict()
    return result


def _recover_in_progress(ledger: Ledger) -> None:
    for state in ledger.items.values():
        if state.status == "in_progress":
            state.status = "pending"
            state.error_message = None


def _claim_pending_units(
    paths: EvaluationPaths,
    ledger: Ledger,
    units: list[WorkUnit],
    limit: int | None = None,
) -> list[WorkUnit]:
    claimed: list[WorkUnit] = []
    for unit in units:
        state = ledger.items[unit.work_id]
        if state.status != "pending":
            continue
        if limit is not None and len(claimed) >= limit:
            break
        state.status = "in_progress"
        state.attempts += 1
        state.started_at = datetime.utcnow().isoformat()
        state.completed_at = None
        state.output_path = None
        state.error_message = None
        claimed.append(unit)
    save_ledger(paths, ledger)
    return claimed


def _claim_next_pending_unit(
    paths: EvaluationPaths,
    ledger: Ledger,
    units: deque[WorkUnit],
) -> WorkUnit | None:
    while units:
        unit = units.popleft()
        state = ledger.items[unit.work_id]
        if state.status != "pending":
            continue
        state.status = "in_progress"
        state.attempts += 1
        state.started_at = datetime.utcnow().isoformat()
        state.completed_at = None
        state.output_path = None
        state.error_message = None
        save_ledger(paths, ledger)
        return unit
    return None


def _finalize_success(paths: EvaluationPaths, ledger: Ledger, unit: WorkUnit, result: dict[str, Any]) -> None:
    result_path = Path(unit.result_path)
    atomic_write_json(result_path, result)
    state = ledger.items[unit.work_id]
    state.status = "completed"
    state.completed_at = datetime.utcnow().isoformat()
    state.output_path = str(result_path)
    state.error_message = None
    save_ledger(paths, ledger)
    print(
        f"[{unit.modality}] completed "
        f"instance={unit.instance_name} "
        f"size={unit.evaluation_size} "
        f"output={result_path}"
        ,
        flush=True,
    )


def _finalize_failure(paths: EvaluationPaths, ledger: Ledger, unit: WorkUnit, error_message: str) -> None:
    state = ledger.items[unit.work_id]
    state.status = "failed"
    state.error_message = error_message
    save_ledger(paths, ledger)
    print(
        f"[{unit.modality}] failed "
        f"instance={unit.instance_name} "
        f"size={unit.evaluation_size} "
        f"error={error_message}",
        flush=True,
    )


def _reset_units_to_pending(paths: EvaluationPaths, ledger: Ledger, units: list[WorkUnit]) -> None:
    changed = False
    for unit in units:
        state = ledger.items[unit.work_id]
        if state.status != "in_progress":
            continue
        state.status = "pending"
        state.completed_at = None
        state.output_path = None
        state.error_message = None
        changed = True
    if changed:
        save_ledger(paths, ledger)


def _default_workers(modality: str, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if modality in {"oracle", "heuristic"}:
        return max(1, min(os.cpu_count() or 1, task_count))
    return 1


def _budget_for_unit(unit: WorkUnit) -> float:
    if unit.modality == "oracle":
        return 10.0
    if unit.modality == "heuristic":
        return 5.0
    if unit.modality == "hybrid":
        return 0.5
    return 5.0


def _log_unit_started(unit: WorkUnit) -> None:
    print(
        f"[{unit.modality}] started "
        f"instance={unit.instance_name} "
        f"size={unit.evaluation_size} "
        f"budget_s={_budget_for_unit(unit):.1f} "
        f"work_id={unit.work_id}",
        flush=True,
    )


def _execute_unit_payload(
    data_root_str: str,
    project_root_str: str,
    modality: str,
    unit_payload: dict[str, Any],
) -> dict[str, Any]:
    data_root = Path(data_root_str)
    project_root = Path(project_root_str)
    paths = build_paths(data_root)
    models = load_model_registry(data_root)
    unit = WorkUnit.model_validate(unit_payload)
    if modality == "oracle":
        return _run_oracle_unit(paths, project_root, unit, models)
    return _run_dynamic_unit(paths, project_root, unit, models)


def _run_units_sequential(
    paths: EvaluationPaths,
    project_root: Path,
    modality: str,
    units: list[WorkUnit],
    models: dict[str, dict[str, Any]],
) -> list[tuple[WorkUnit, dict[str, Any] | None, str | None]]:
    outcomes: list[tuple[WorkUnit, dict[str, Any] | None, str | None]] = []
    for unit in units:
        try:
            if modality == "oracle":
                result = _run_oracle_unit(paths, project_root, unit, models)
            else:
                result = _run_dynamic_unit(paths, project_root, unit, models)
            outcomes.append((unit, result, None))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            outcomes.append((unit, None, str(exc)))
    return outcomes


def _submit_unit(
    executor: ProcessPoolExecutor,
    data_root: Path,
    project_root: Path,
    modality: str,
    unit: WorkUnit,
) -> Future:
    _log_unit_started(unit)
    return executor.submit(
        _execute_unit_payload,
        str(data_root),
        str(project_root),
        modality,
        unit.model_dump(mode="json"),
    )


def _stop_executor(executor: ProcessPoolExecutor) -> None:
    executor.shutdown(wait=False, cancel_futures=True)
    for process in getattr(executor, "_processes", {}).values():
        try:
            process.terminate()
        except Exception:
            pass


def _run_parallel_streaming(
    paths: EvaluationPaths,
    ledger: Ledger,
    data_root: Path,
    project_root: Path,
    modality: str,
    queue: deque[WorkUnit],
    worker_count: int,
) -> int:
    completed = 0
    in_flight: dict[Future, WorkUnit] = {}
    executor = ProcessPoolExecutor(max_workers=worker_count)
    try:
        for _ in range(min(worker_count, len(queue))):
            unit = _claim_next_pending_unit(paths, ledger, queue)
            if unit is None:
                break
            future = _submit_unit(executor, data_root, project_root, modality, unit)
            in_flight[future] = unit

        while in_flight:
            done, _pending = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                unit = in_flight.pop(future)
                try:
                    result = future.result()
                    _finalize_success(paths, ledger, unit, result)
                    completed += 1
                except Exception as exc:
                    _finalize_failure(paths, ledger, unit, str(exc))

                next_unit = _claim_next_pending_unit(paths, ledger, queue)
                if next_unit is not None:
                    next_future = _submit_unit(executor, data_root, project_root, modality, next_unit)
                    in_flight[next_future] = next_unit
        executor.shutdown(wait=True, cancel_futures=False)
        return completed
    except KeyboardInterrupt:
        _reset_units_to_pending(paths, ledger, list(in_flight.values()))
        _stop_executor(executor)
        raise
    except BaseException:
        _reset_units_to_pending(paths, ledger, list(in_flight.values()))
        _stop_executor(executor)
        raise


def run_modality(
    data_root: Path,
    modality: str,
    limit: int | None = None,
    workers: int | None = None,
) -> tuple[int, int]:
    paths = build_paths(data_root)
    project_root = project_root_from(data_root.resolve())
    models = load_model_registry(data_root)
    if modality != "oracle" and not _scenario_files(paths):
        generate_scenarios(data_root)
    if modality not in {"oracle", "heuristic", "ai", "hybrid"}:
        raise ValueError(f"Unknown modality: {modality}")
    if workers is not None and workers < 1:
        raise ValueError("--workers must be >= 1")
    if modality not in {"oracle", "heuristic"} and workers not in {None, 1}:
        raise ValueError(f"Multiprocessing is only supported for oracle and heuristic, not '{modality}'")
    with modality_lock(paths, modality):
        ledger = sync_ledger(data_root, modality)
        _recover_in_progress(ledger)
        save_ledger(paths, ledger)
        all_units = enumerate_work_units(data_root, modality)
        pending_units = [unit for unit in all_units if ledger.items[unit.work_id].status == "pending"]
        if limit is not None:
            pending_units = pending_units[:limit]
        attempted = len(pending_units)
        if attempted == 0:
            return 0, 0

        if workers is None:
            worker_count = _default_workers(modality, attempted)
        else:
            worker_count = max(1, workers)

        if worker_count == 1:
            claimed = _claim_pending_units(paths, ledger, pending_units, limit=attempted)
            for unit in claimed:
                _log_unit_started(unit)
            outcomes = _run_units_sequential(paths, project_root, modality, claimed, models)
            completed = 0
            for unit, result, error_message in outcomes:
                if result is not None:
                    _finalize_success(paths, ledger, unit, result)
                    completed += 1
                else:
                    _finalize_failure(paths, ledger, unit, error_message or "unknown worker error")
            return completed, attempted
        else:
            queue = deque(pending_units)
            completed = _run_parallel_streaming(
                paths=paths,
                ledger=ledger,
                data_root=data_root,
                project_root=project_root,
                modality=modality,
                queue=queue,
                worker_count=worker_count,
            )
            return completed, attempted


def status_summary(data_root: Path) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    paths = build_paths(data_root)
    for modality in ("oracle", "heuristic", "ai", "hybrid"):
        ledger = sync_ledger(data_root, modality)
        counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        for item in ledger.items.values():
            counts[item.status] += 1
        summary[modality] = counts
    return summary
