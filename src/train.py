"""Training, checkpointing, and pilot-search orchestration."""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .agop import compute_input_agop, summarize_agop
from .data import make_mod_add_dataset, validate_mod_add_dataset
from .fourier import (
    alignment_from_agop,
    make_fourier_basis,
    make_random_basis,
    subspace_alignment_from_eigenvectors,
    valid_fourier_K,
    validate_fourier,
)
from .metrics import evaluate_loss_acc, first_crossing, make_checkpoint_steps, parameter_l2_norm
from .models import make_model
from .ntk import compute_correct_logit_ntk, ntk_relative_drift
from .utils import ensure_dir, get_git_hash, save_json, save_yaml, select_device, set_seed


CORE_RESULT_FILES = ("metrics.csv", "agop_spectra.csv")


def _config_value(cfg: dict[str, Any], key: str, default: Any) -> Any:
    return cfg[key] if key in cfg else default


def _make_run_id(cfg: dict[str, Any], seed: int, prefix: str = "run") -> str:
    return (
        f"{prefix}_seed{seed}_p{cfg['p']}_tf{cfg['train_fraction']}_"
        f"{cfg['model_type']}_w{cfg['hidden_width']}_lr{cfg['lr']}_"
        f"wd{cfg['weight_decay']}_init{cfg['init_scale']}"
    ).replace(".", "p")


def _sample_probe_indices(n: int, size: int | str, seed: int) -> np.ndarray:
    if size == "full" or size is None or int(size) >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(size), replace=False)).astype(np.int64)


def _make_ntk_probe(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = max(1, size // 2)
    n_train = min(len(train_idx), half)
    n_test = min(len(test_idx), size - n_train)
    chosen = []
    if n_train:
        chosen.append(rng.choice(train_idx, size=n_train, replace=False))
    if n_test:
        chosen.append(rng.choice(test_idx, size=n_test, replace=False))
    if not chosen:
        return np.array([], dtype=np.int64)
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out.astype(np.int64)


def run_correctness_checks(run_dir: Path, p: int, X_full: torch.Tensor, y_full: torch.Tensor, meta: dict[str, Any]) -> None:
    messages = []
    messages.extend(validate_mod_add_dataset(X_full, y_full, meta))
    valid_K = valid_fourier_K(p, [1, 2, 4, 8, 16])
    if valid_K:
        messages.extend(validate_fourier(p, valid_K[0]))
    with open(run_dir / "checks.log", "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(msg + "\n")


def compute_checkpoint_metrics(
    *,
    cfg: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    run_id: str,
    seed: int,
    step: int,
    elapsed_seconds: float,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    X_agop_probe: torch.Tensor,
    X_ntk_probe: torch.Tensor,
    y_ntk_probe: torch.Tensor,
    fourier_Q: dict[int, torch.Tensor],
    random_P: dict[int, torch.Tensor],
    K0_ntk: torch.Tensor | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor | None]:
    train_loss, train_acc = evaluate_loss_acc(model, X_train, y_train, device)
    test_loss, test_acc = evaluate_loss_acc(model, X_test, y_test, device)
    weight_norm = parameter_l2_norm(model)
    row: dict[str, Any] = {
        "run_id": run_id,
        "seed": int(seed),
        "p": int(cfg["p"]),
        "train_fraction": float(cfg["train_fraction"]),
        "model_type": cfg["model_type"],
        "hidden_width": int(cfg["hidden_width"]),
        "lr": float(cfg["lr"]),
        "weight_decay": float(cfg["weight_decay"]),
        "init_scale": float(cfg["init_scale"]),
        "step": int(step),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "weight_norm": weight_norm,
        "log_weight_norm": float(math.log(weight_norm + 1e-12)),
        "elapsed_seconds": float(elapsed_seconds),
    }
    spectra_rows: list[dict[str, Any]] = []

    if bool(cfg.get("compute_agop", True)):
        A = compute_input_agop(
            model,
            X_agop_probe,
            batch_size=int(cfg.get("agop_batch_size", 128)),
            device=device,
        )
        fourier_dims = {K: Q.shape[1] for K, Q in fourier_Q.items()}
        agop_summary, evals = summarize_agop(A, fourier_dims=fourier_dims)
        row.update(agop_summary)
        eig_out = None
        for K, Q in fourier_Q.items():
            align = alignment_from_agop(A, Q)
            row[f"align_K{K}"] = float(align["alignment"])
            row[f"raw_overlap_K{K}"] = float(align["raw_overlap"])
            eig_out = align
            if bool(cfg.get("random_alignment", True)) and K in random_P:
                s = Q.shape[1]
                U_r = eig_out["eigenvectors"][:, :s]
                rand_align, rand_raw = subspace_alignment_from_eigenvectors(U_r, random_P[K])
                row[f"random_align_K{K}"] = rand_align
                row[f"random_raw_overlap_K{K}"] = rand_raw
        for idx, val in enumerate(evals[: min(2 * int(cfg["p"]), 50)].tolist(), start=1):
            spectra_rows.append(
                {
                    "run_id": run_id,
                    "seed": int(seed),
                    "step": int(step),
                    "eig_idx": idx,
                    "eigenvalue": float(val),
                }
            )

    if bool(cfg.get("compute_ntk", True)) and len(X_ntk_probe) > 0:
        Kt = compute_correct_logit_ntk(model, X_ntk_probe, y_ntk_probe, device=device)
        if K0_ntk is None:
            K0_ntk = Kt
        row["ntk_drift_correct_logit"] = ntk_relative_drift(Kt, K0_ntk)
    else:
        row["ntk_drift_correct_logit"] = float("nan")

    return row, spectra_rows, K0_ntk


def run_single_experiment(
    cfg: dict[str, Any],
    seed: int,
    results_dir: str | Path,
    device_arg: str | None = None,
    run_prefix: str = "run",
    run_subdir: str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train one seed/config and write per-run artifacts."""
    cfg = dict(cfg)
    cfg["seed"] = int(seed)
    device = select_device(device_arg or cfg.get("device", "auto"))
    set_seed(int(seed), deterministic=bool(cfg.get("deterministic", False)))
    results_dir = ensure_dir(results_dir)
    run_id = _make_run_id(cfg, seed, prefix=run_prefix)
    run_dir = ensure_dir(results_dir / (run_subdir or f"run_seed{seed}"))

    X_train, y_train, X_test, y_test, X_full, y_full, meta = make_mod_add_dataset(
        int(cfg["p"]),
        float(cfg["train_fraction"]),
        int(seed),
    )
    np.savez(
        run_dir / "split_indices.npz",
        train_indices=meta["train_indices"],
        test_indices=meta["test_indices"],
        pairs=meta["pairs"],
    )
    if bool(cfg.get("run_checks", True)):
        run_correctness_checks(run_dir, int(cfg["p"]), X_full, y_full, meta)

    input_dim = X_full.shape[1]
    output_dim = int(cfg["p"])
    model = make_model(
        cfg.get("model_type", "quadratic"),
        input_dim,
        int(cfg["hidden_width"]),
        output_dim,
        float(cfg["init_scale"]),
        int(seed),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    checkpoints = make_checkpoint_steps(int(cfg["max_steps"]), cfg.get("checkpoint_preset", "main"))
    checkpoint_set = set(checkpoints)

    p = int(cfg["p"])
    requested_K = cfg.get("fourier_K", [1, 2, 4, 8, 16])
    valid_K = valid_fourier_K(p, requested_K)
    fourier_Q = {K: make_fourier_basis(p, K)[0] for K in valid_K}
    random_P = {
        K: make_random_basis(2 * p, fourier_Q[K].shape[1], seed=seed + 1000 + K)[1]
        for K in valid_K
    }

    if cfg.get("agop_probe_size", "full") == "full" or p <= int(cfg.get("agop_exact_p_max", 47)):
        agop_size: int | str = "full"
    else:
        agop_size = int(cfg.get("agop_probe_size", 1024))
    agop_probe_idx = _sample_probe_indices(len(X_full), agop_size, seed + 999)
    X_agop_probe = X_full[torch.from_numpy(agop_probe_idx).long()]

    ntk_probe_idx = _make_ntk_probe(
        np.asarray(meta["train_indices"]),
        np.asarray(meta["test_indices"]),
        int(cfg.get("ntk_probe_size", 64)),
        seed + 1234,
    )
    X_ntk_probe = X_full[torch.from_numpy(ntk_probe_idx).long()] if len(ntk_probe_idx) else X_full[:0]
    y_ntk_probe = y_full[torch.from_numpy(ntk_probe_idx).long()] if len(ntk_probe_idx) else y_full[:0]

    resolved = dict(cfg)
    resolved.update(
        {
            "device": str(device),
            "git_hash": get_git_hash(),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "checkpoint_steps": checkpoints,
            "valid_fourier_K": valid_K,
            "agop_probe_size_resolved": int(len(X_agop_probe)),
            "ntk_probe_size_resolved": int(len(X_ntk_probe)),
        }
    )
    save_yaml(resolved, run_dir / "config_resolved.yaml")
    save_json(resolved, run_dir / "config_resolved.json")

    metrics_rows: list[dict[str, Any]] = []
    spectra_rows: list[dict[str, Any]] = []
    K0_ntk = None
    start_time = time.time()
    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    iterator = range(int(cfg["max_steps"]) + 1)
    if progress:
        iterator = tqdm(iterator, desc=run_id, leave=False)

    initial_train_loss = None
    for step in iterator:
        if step in checkpoint_set:
            row, spectra, K0_ntk = compute_checkpoint_metrics(
                cfg=cfg,
                model=model,
                device=device,
                run_id=run_id,
                seed=seed,
                step=step,
                elapsed_seconds=time.time() - start_time,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                X_agop_probe=X_agop_probe,
                X_ntk_probe=X_ntk_probe,
                y_ntk_probe=y_ntk_probe,
                fourier_Q=fourier_Q,
                random_P=random_P,
                K0_ntk=K0_ntk,
            )
            if step == 0:
                initial_train_loss = row["train_loss"]
            metrics_rows.append(row)
            spectra_rows.extend(spectra)

        if step == int(cfg["max_steps"]):
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_train_d)
        loss = F.cross_entropy(logits, y_train_d)
        loss.backward()
        optimizer.step()

    metrics_df = pd.DataFrame(metrics_rows)
    spectra_df = pd.DataFrame(spectra_rows)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    if not spectra_df.empty:
        spectra_df.to_csv(run_dir / "agop_spectra.csv", index=False)
    else:
        pd.DataFrame(columns=["run_id", "seed", "step", "eig_idx", "eigenvalue"]).to_csv(
            run_dir / "agop_spectra.csv",
            index=False,
        )
    if bool(cfg.get("save_final_model", False)):
        torch.save({"model_state_dict": model.state_dict(), "config": resolved}, run_dir / "final_model.pt")

    final = metrics_df.iloc[-1].to_dict()
    summary = summarize_run_metrics(metrics_df)
    summary.update(
        {
            "run_id": run_id,
            "seed": int(seed),
            "run_dir": str(run_dir),
            "initial_train_loss": float(initial_train_loss) if initial_train_loss is not None else float("nan"),
            "final_train_loss": float(final["train_loss"]),
            "final_test_acc": float(final["test_acc"]),
        }
    )
    save_json(summary, run_dir / "run_summary.json")
    return summary


def summarize_run_metrics(metrics_df: pd.DataFrame) -> dict[str, Any]:
    steps = metrics_df["step"].tolist()
    train_acc = metrics_df["train_acc"].tolist()
    test_acc = metrics_df["test_acc"].tolist()
    t_train_fit = first_crossing(steps, train_acc, 0.99, "above")
    both_test = metrics_df.apply(
        lambda r: float(r["test_acc"]) >= 0.95 and float(r["train_acc"]) >= 0.99,
        axis=1,
    ).astype(float)
    t_grok = first_crossing(steps, both_test.tolist(), 0.5, "above", consecutive=2)
    t_test_90 = first_crossing(steps, test_acc, 0.90, "above")
    t_test_95 = first_crossing(steps, test_acc, 0.95, "above")
    if t_train_fit is None or t_grok is None:
        grokking_like = False
    else:
        delay = t_grok - t_train_fit
        grokking_like = delay > 0 and delay >= max(1, int(0.05 * max(t_grok, 1)))
    return {
        "max_train_acc": float(metrics_df["train_acc"].max()),
        "max_test_acc": float(metrics_df["test_acc"].max()),
        "final_train_acc": float(metrics_df["train_acc"].iloc[-1]),
        "final_test_acc": float(metrics_df["test_acc"].iloc[-1]),
        "t_train_fit": int(t_train_fit) if t_train_fit is not None else None,
        "t_test_90": int(t_test_90) if t_test_90 is not None else None,
        "t_test_95": int(t_test_95) if t_test_95 is not None else None,
        "t_grok": int(t_grok) if t_grok is not None else None,
        "grokking_like": bool(grokking_like),
    }


def combine_results(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = Path(results_dir)
    metric_paths = sorted(results_dir.glob("**/metrics.csv"))
    spectrum_paths = sorted(results_dir.glob("**/agop_spectra.csv"))
    metrics = [pd.read_csv(path) for path in metric_paths if path.is_file()]
    spectra = [pd.read_csv(path) for path in spectrum_paths if path.is_file()]
    metrics_all = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    spectra_all = pd.concat(spectra, ignore_index=True) if spectra else pd.DataFrame()
    if not metrics_all.empty:
        metrics_all.to_csv(results_dir / "metrics_all.csv", index=False)
    if not spectra_all.empty:
        spectra_all.to_csv(results_dir / "agop_spectra_all.csv", index=False)
    return metrics_all, spectra_all


def run_standard_experiments(
    cfg: dict[str, Any],
    results_dir: str | Path | None = None,
    seeds: list[int] | None = None,
    device: str | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    results_dir = ensure_dir(results_dir or cfg.get("results_dir", "results/main"))
    seeds = seeds if seeds is not None else list(cfg.get("seeds", [0]))
    save_yaml(cfg, results_dir / "config_resolved.yaml")
    summaries = []
    for seed in seeds:
        summaries.append(run_single_experiment(cfg, int(seed), results_dir, device_arg=device, progress=progress))
    pd.DataFrame(summaries).to_csv(results_dir / "run_summaries.csv", index=False)
    combine_results(results_dir)
    return summaries


def _grid_dicts(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def _candidate_score(row: dict[str, Any]) -> float:
    score = 0.0
    score += 4.0 * float(row.get("max_train_acc", 0.0))
    score += 6.0 * float(row.get("max_test_acc", 0.0))
    if row.get("grokking_like"):
        score += 100.0
    if row.get("t_train_fit") is not None and row.get("t_test_95") is not None:
        score += min(max(row["t_test_95"] - row["t_train_fit"], 0), 50000) / 1000.0
    return score


def _run_candidate(
    base: dict[str, Any],
    candidate: dict[str, Any],
    stage_cfg: dict[str, Any],
    results_dir: Path,
    config_id: str,
    device: str | None,
    progress: bool,
) -> dict[str, Any]:
    cfg = dict(base)
    cfg.update(candidate)
    cfg.update(stage_cfg)
    seed = int(cfg.get("seeds", [0])[0] if isinstance(cfg.get("seeds", [0]), list) else cfg.get("seeds", 0))
    run_dir_name = f"{stage_cfg.get('stage_name', 'stage')}_{config_id}"
    summary = run_single_experiment(
        cfg,
        seed,
        results_dir,
        device_arg=device,
        run_prefix=config_id,
        run_subdir=run_dir_name,
        progress=progress,
    )
    summary.update({k: cfg[k] for k in ["p", "train_fraction", "model_type", "hidden_width", "lr", "weight_decay", "init_scale", "max_steps"]})
    summary["config_id"] = config_id
    return summary


def run_pilot_search(cfg: dict[str, Any], device: str | None = None, progress: bool = True) -> pd.DataFrame:
    """Adaptive pilot search: scout broadly, extend top configs, measure top candidates."""
    results_dir = ensure_dir(cfg.get("results_dir", "results/pilot"))
    base = dict(cfg.get("base", {}))
    grid = _grid_dicts(cfg.get("grid", {}))
    summaries: list[dict[str, Any]] = []

    stage1_cfg = dict(cfg.get("stage1", {}))
    stage1_cfg["stage_name"] = "stage1"
    for i, cand in enumerate(grid):
        config_id = f"cfg{i:04d}"
        summaries.append(_run_candidate(base, cand, stage1_cfg, results_dir, config_id, device, progress))
        pd.DataFrame(summaries).to_csv(results_dir / "pilot_summary.csv", index=False)

    stage1_df = pd.DataFrame(summaries)
    keep_top = int(stage1_cfg.get("keep_top", 24))
    top_stage1 = stage1_df.assign(score=stage1_df.apply(lambda r: _candidate_score(r.to_dict()), axis=1))
    top_stage1 = top_stage1.sort_values("score", ascending=False).head(keep_top)

    stage2_cfg = dict(cfg.get("stage2", {}))
    stage2_cfg["stage_name"] = "stage2"
    stage2_rows = []
    for _, row in top_stage1.iterrows():
        cand = {k: row[k] for k in ["p", "train_fraction", "model_type", "hidden_width", "lr", "weight_decay", "init_scale"]}
        config_id = str(row["config_id"]) + "_ext"
        stage2_rows.append(_run_candidate(base, cand, stage2_cfg, results_dir, config_id, device, progress))
        pd.concat([pd.DataFrame(summaries), pd.DataFrame(stage2_rows)], ignore_index=True).to_csv(
            results_dir / "pilot_summary.csv", index=False
        )

    stage2_df = pd.DataFrame(stage2_rows)
    top_stage2 = stage2_df.assign(score=stage2_df.apply(lambda r: _candidate_score(r.to_dict()), axis=1))
    top_stage2 = top_stage2.sort_values("score", ascending=False).head(int(stage2_cfg.get("keep_top", 8)))

    stage3_cfg = dict(cfg.get("stage3", {}))
    stage3_cfg["stage_name"] = "stage3"
    stage3_rows = []
    for _, row in top_stage2.iterrows():
        cand = {k: row[k] for k in ["p", "train_fraction", "model_type", "hidden_width", "lr", "weight_decay", "init_scale", "max_steps"]}
        config_id = str(row["config_id"]) + "_metrics"
        stage3_rows.append(_run_candidate(base, cand, stage3_cfg, results_dir, config_id, device, progress))

    all_rows = summaries + stage2_rows + stage3_rows
    pilot_df = pd.DataFrame(all_rows)

    metric_candidates = pd.DataFrame(stage3_rows)
    if cfg.get("fallback", {}).get("enabled", False) and not bool(metric_candidates.get("grokking_like", pd.Series(dtype=bool)).any()):
        fallback = dict(cfg.get("fallback", {}))
        fallback_grid = {k: v for k, v in fallback.items() if isinstance(v, list)}
        fallback_stage = {k: v for k, v in fallback.items() if not isinstance(v, list)}
        fallback_stage["stage_name"] = "fallback"
        fallback_rows = []
        for i, cand in enumerate(_grid_dicts(fallback_grid)):
            config_id = f"fallback{i:03d}"
            fallback_rows.append(_run_candidate(base, cand, fallback_stage, results_dir, config_id, device, progress))
            pd.concat([pilot_df, pd.DataFrame(fallback_rows)], ignore_index=True).to_csv(
                results_dir / "pilot_summary.csv", index=False
            )
        all_rows.extend(fallback_rows)
        metric_candidates = pd.concat([metric_candidates, pd.DataFrame(fallback_rows)], ignore_index=True)

    pilot_df = pd.DataFrame(all_rows)
    if not pilot_df.empty:
        pilot_df["score"] = pilot_df.apply(lambda r: _candidate_score(r.to_dict()), axis=1)
        pilot_df = pilot_df.sort_values("score", ascending=False)
    pilot_df.to_csv(results_dir / "pilot_summary.csv", index=False)

    if not metric_candidates.empty:
        best = metric_candidates.assign(score=metric_candidates.apply(lambda r: _candidate_score(r.to_dict()), axis=1))
        best = best.sort_values("score", ascending=False).iloc[0].to_dict()
    elif not pilot_df.empty:
        best = pilot_df.iloc[0].to_dict()
    else:
        raise RuntimeError("Pilot search produced no candidates")

    best_cfg = dict(base)
    for key in ["p", "train_fraction", "model_type", "hidden_width", "lr", "weight_decay", "init_scale"]:
        best_cfg[key] = best[key]
    pilot_t_grok = best.get("t_grok") or best.get("t_test_95") or best.get("max_steps") or base.get("max_steps", 50000)
    main_steps = int(min(100000, max(30000, math.ceil(1.5 * float(pilot_t_grok) / 5000.0) * 5000)))
    if not best.get("grokking_like", False):
        main_steps = max(main_steps, min(100000, int(best.get("max_steps", 50000))))
    best_cfg.update(
        {
            "mode": "standard",
            "results_dir": "results/main",
            "seeds": [0, 1, 2, 3, 4],
            "max_steps": main_steps,
            "checkpoint_preset": "main",
            "compute_agop": True,
            "compute_ntk": True,
            "save_final_model": True,
        }
    )
    save_yaml(best_cfg, results_dir / "best_config.yaml")
    combine_results(results_dir)
    return pilot_df
