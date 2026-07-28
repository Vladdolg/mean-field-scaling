from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from statistics import NormalDist
from scipy.stats import halfnorm

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
import pandas as pd

# sys.path.insert(0, ".")
# from SSA.numba_1d import make_ssa_state_1d, SSAState1D

from numba_1d_plus import make_ssa_state_1d, SSAState1D


# =============================================================================
# Конфиг
# =============================================================================
@dataclass
class ExperimentConfig:
    # Модель
    b: float = 1.0
    sigma: float = 1.0
    d_prime: float = 1.0

    # Геометрия
    L: float = 1000.0

    # Sweep по (sigma, d_prime)
    # sigma_values: tuple[float, ...] = (0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75, 2.1, 2.5)
    # d_prime_values: tuple[float, ...] = (0.10, 0.13, 0.17, 0.22, 0.28, 0.36, 0.46, 0.6, 0.78, 1.0,)

    sigma_values: tuple[float, ...] = (0.50, 0.75, 1.15, 1.75, 2.6, 4.00,)
    d_prime_values: tuple[float, ...] = (0.10, 0.15, 0.25, 0.40, 0.65, 1.00,)

    gaussian_cutoff_sigmas: float = 5.0

    # Sweep по d внутри каждой пары (sigma, d_prime)
    d_test_range: tuple[float, float, int] = (0.01, 0.01, 1)

    # Warmup
    initial_density_frac: float = 0.1
    warmup_threshold_frac: float = 0.50        # ждём 50% от pop_exp перед оценкой τ
    warmup_stability_tol: float = 0.05        # 5% — критерий стабильности двух окон
    warmup_min_stable_windows: int = 10        # минимум 10 стабильных окна подряд
    warmup_event_chunk: int = 5000            # чанк до достижения порога 70%
    max_warmup_wall_seconds: float = 3000.0

    # Измерения по физическому времени
    event_frac: float = 0.10
    min_events_per_sample: int = 10

    # Pilot / batch means
    pilot_samples_multiple: int = 10
    pilot_max_lag: int = 80
    batch_tau_multiple: float = 10.0
    min_batch_size: int = 50

    # Адаптивная остановка
    min_batches: int = 20
    max_batches: int = 100000
    ci_confidence: float = 0.95

    # Критерий преподавателя: +-5% на разницу с mean field
    #   CI_half_width(n̂) ≤ rel_ci_target * |n̂ - n_MF|
    # При очень малой разнице (d≈0) — floor, чтобы не мерить вечно.
    rel_ci_target: float = 0.05
    delta_floor_frac: float = 0.0002  # floor = rel_ci_target * delta_floor_frac * |n_MF|

    # Параллельность
    n_jobs: int = 4

    # Сохранение
    output_dir: str = "."
    points_filename: str = "kappa_big_d.csv"

    def get_d_values(self) -> NDArray[np.float64]:
        return np.linspace(*self.d_test_range)

    def n_expected(self, d: float) -> float:
        return (self.b - d) / self.d_prime if self.d_prime > 0 else np.nan

    def pop_expected(self, d: float) -> int:
        n_exp = self.n_expected(d)
        if not np.isfinite(n_exp):
            return 0
        return max(0, int(round(n_exp * self.L)))


# =============================================================================
# Построение gaussian таблиц ядер
# =============================================================================
def make_gaussian_tables_1d(
    sigma: float,
    cutoff_sigmas: float = 5.0,
    n_birth: int = 256,
    n_death: int = 1024,
):
    """
    Таблицы для SSA.numba_1d под 1D normal kernels.

    Birth:
      X_child = X_parent ± R,
      где R ~ HalfNormal(sigma), но усечён на cutoff = cutoff_sigmas * sigma.

    Death:
      w(r) = 0.5 * pdf_halfnormal_truncated(r),  r in [0, cutoff]
      Так нужно, потому что симулятор работает с расстоянием r >= 0,
      а вклад слева/справа в 1D учитывается через частицы, а не через знак.
    """
    cutoff = cutoff_sigmas * sigma
    dist = halfnorm(scale=sigma)
    mass = dist.cdf(cutoff)
    
    birth_q = (np.arange(n_birth, dtype=np.float64) + 0.5) / n_birth
    ppf_args = birth_q * mass
    birth_r = dist.ppf(ppf_args)
    birth_r = np.clip(birth_r, 0.0, cutoff)
    
    death_r = np.linspace(0.0, cutoff, n_death, dtype=np.float64)
    death_w = 0.5 * dist.pdf(death_r)

    birth_x = birth_q[np.newaxis, :]
    birth_y = birth_r[np.newaxis, :]
    death_x = death_r[np.newaxis, np.newaxis, :]
    death_y = death_w[np.newaxis, np.newaxis, :]
    cutoffs = np.array([[cutoff]], dtype=np.float64)

    return birth_x, birth_y, death_x, death_y, cutoffs


# =============================================================================
# Вспомогательная статистика
# =============================================================================
def estimate_autocorrelation_time(samples: NDArray[np.float64], max_lag: int = 100) -> float:
    samples = np.asarray(samples)
    n = len(samples)
    max_lag = min(max_lag, n // 2)
    if max_lag < 2:
        return 1.0

    mean = samples.mean()
    var = samples.var()
    if var < 1e-12:
        return 1.0

    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        c = np.mean((samples[: n - lag] - mean) * (samples[lag:] - mean))
        autocorr[lag] = c / var

    tau = 1.0
    for k in range(1, max_lag):
        if autocorr[k] < 0.05:
            break
        tau += 2.0 * autocorr[k]

    return max(1.0, tau)


def compute_mean_and_ci_from_batch_means(
    batch_means: list[float], confidence: float
) -> tuple[float, float, float, float, float]:
    arr = np.asarray(batch_means)
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, math.inf, math.inf, -math.inf, math.inf
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    alpha = 1.0 - confidence
    half_width = float(NormalDist().inv_cdf(1.0 - alpha / 2.0)) * se
    return mean, se, half_width, mean - half_width, mean + half_width


def compute_ci_target(
    cfg: ExperimentConfig, density_mean: float, n_mf: float
) -> tuple[float, bool]:
    """
    Критерий преподавателя: ±5% на разницу с mean field.

    CI_half_width ≤ 0.05 * max(|delta|, floor)
      где floor = 0.002 * |n_MF|   (чтобы не мерить вечно при delta → 0)

    Возвращает (target, is_strict):
      target    — порог для CI_half_width
      is_strict — True если |delta| > floor (т.е. сработал именно 5%-критерий,
                  а не floor). Помечает точки, реально достигшие 5% на разницу.
    """
    delta = abs(density_mean - n_mf)
    floor = cfg.delta_floor_frac * abs(n_mf)
    is_strict = delta >= floor
    target = cfg.rel_ci_target * max(delta, floor)
    return target, is_strict


# =============================================================================
# Спавн случайных частиц (замена spawn_random)
# =============================================================================
def spawn_uniform(sim: SSAState1D, species: int, count: int, L: float) -> int:
    spawned = 0
    for _ in range(count):
        pos = np.random.uniform(0.0, L)
        if sim.spawn_particle(species, pos):
            spawned += 1
    return spawned


# =============================================================================
# Warmup / sampling
# =============================================================================
def run_warmup(sim: SSAState1D, cfg: ExperimentConfig, pop_exp: int) -> dict:
    warmup_start = time.perf_counter()
    initial_pop = max(10, int(cfg.initial_density_frac * pop_exp))
    spawn_uniform(sim, 0, initial_pop, cfg.L)
    threshold_pop = cfg.warmup_threshold_frac * pop_exp

    # --- Фаза 1: ждём 70% от pop_exp ---
    phase1_chunks = 0
    while sim.current_population() < threshold_pop:
        if time.perf_counter() - warmup_start > cfg.max_warmup_wall_seconds:
            return {
                "warmup_time_wall": time.perf_counter() - warmup_start,
                "warmup_reached_target": False,
                "warmup_phase1_chunks": phase1_chunks,
                "warmup_tau_int": np.nan,
                "warmup_window_size": 0,
                "warmup_stability_windows": 0,
                "warmup_stable_count": 0,
            }
        sim.run_events(cfg.warmup_event_chunk)
        phase1_chunks += 1
        if sim.current_population() <= 0:
            spawn_uniform(sim, 0, initial_pop, cfg.L)

    # --- Фаза 2: грубая оценка τ_int по пилотным образцам ---
    pilot_dt = cfg.event_frac / (2.0 * cfg.b)
    pilot_samples = collect_density_samples_time(sim, cfg, int(cfg.pilot_samples_multiple * pop_exp), pilot_dt)
    tau_int = estimate_autocorrelation_time(pilot_samples, max_lag=cfg.pilot_max_lag)
    window_size = max(cfg.min_batch_size, int(math.ceil(cfg.batch_tau_multiple * tau_int)))

    # --- Фаза 3: проверяем стабильность двух последовательных окон ---
    stable_count = 0
    stability_windows = 0
    prev_mean = float(pilot_samples.mean())

    while True:
        if time.perf_counter() - warmup_start > cfg.max_warmup_wall_seconds:
            return {
                "warmup_time_wall": time.perf_counter() - warmup_start,
                "warmup_reached_target": False,
                "warmup_phase1_chunks": phase1_chunks,
                "warmup_tau_int": tau_int,
                "warmup_window_size": window_size,
                "warmup_stability_windows": stability_windows,
                "warmup_stable_count": stable_count,
            }

        window_samples = collect_density_samples_time(sim, cfg, window_size, pilot_dt)
        curr_mean = float(window_samples.mean())
        stability_windows += 1

        if prev_mean > 0 and abs(curr_mean - prev_mean) / prev_mean < cfg.warmup_stability_tol:
            stable_count += 1
        else:
            stable_count = 0

        prev_mean = curr_mean

        if stable_count >= cfg.warmup_min_stable_windows:
            return {
                "warmup_time_wall": time.perf_counter() - warmup_start,
                "warmup_reached_target": True,
                "warmup_phase1_chunks": phase1_chunks,
                "warmup_tau_int": tau_int,
                "warmup_window_size": window_size,
                "warmup_stability_windows": stability_windows,
                "warmup_stable_count": stable_count,
            }


def collect_density_samples_time(
    sim: SSAState1D,
    cfg: ExperimentConfig,
    n_samples: int,
    dt_sample: float,
) -> NDArray[np.float64]:
    samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        sim.run_until_time(dt_sample)
        samples[i] = sim.current_population() / cfg.L
    return samples


# =============================================================================
# Один d
# =============================================================================
def run_single_d(d_val: float, cfg: ExperimentConfig, seed: int) -> dict:
    n_exp = cfg.n_expected(d_val)
    pop_exp = cfg.pop_expected(d_val)
    
    res = {
        "d": d_val,
        "mf_density": n_exp,
        "mf_population": pop_exp,
        "density_mean": np.nan,
        "density_mean_se": np.nan,
        "density_half_width": np.nan,
        "density_ci_lower": np.nan,
        "density_ci_upper": np.nan,
        "tau_int": np.nan,
        "batch_size": 0,
        "batches_used": 0,
        "sample_dt": np.nan,
        "measurement_sim_time": 0.0,
        "measurement_time_wall": 0.0,
        "ci_target": np.nan,
        "is_strict": False,          # True = реально ±5% на разницу, False = сработал floor
        "converged": False,
        "warmup_reached_target": False,
        "warmup_time_wall": 0.0,
        "warmup_phase1_chunks": 0,
        "warmup_tau_int": np.nan,
        "warmup_window_size": 0,
        "warmup_stability_windows": 0,
        "warmup_stable_count": 0,
    }

    if not np.isfinite(n_exp) or pop_exp < 10:
        return res

    birth_x, birth_y, death_x, death_y, cutoffs = make_gaussian_tables_1d(
        sigma=cfg.sigma,
        cutoff_sigmas=cfg.gaussian_cutoff_sigmas,
    )
    
    cutoff = cfg.gaussian_cutoff_sigmas * cfg.sigma
    cell_count = max(20, int(math.ceil(cfg.L / (cutoff / 2.0))))
    avg_per_cell = max(1, pop_exp / cell_count)
    cell_capacity = max(64, int(avg_per_cell * 6))

    sim = make_ssa_state_1d(
        M=1,
        area_len=cfg.L,
        birth_rates=np.array([cfg.b]),
        death_rates=np.array([d_val]),
        dd_matrix=np.array([[cfg.d_prime]]),
        birth_x=birth_x,
        birth_y=birth_y,
        death_x=death_x,
        death_y=death_y,
        cutoffs=cutoffs,
        cell_count=cell_count,
        cell_capacity=cell_capacity,
        is_periodic=True,
        resync_interval=10000,
        seed=seed,
    )

    warmup_res = run_warmup(sim, cfg, pop_exp)
    res.update(
        {
            "warmup_reached_target": warmup_res["warmup_reached_target"],
            "warmup_time_wall": warmup_res["warmup_time_wall"],
            "warmup_phase1_chunks": warmup_res["warmup_phase1_chunks"],
            "warmup_tau_int": warmup_res["warmup_tau_int"],
            "warmup_window_size": warmup_res["warmup_window_size"],
            "warmup_stability_windows": warmup_res["warmup_stability_windows"],
            "warmup_stable_count": warmup_res["warmup_stable_count"],
        }
    )
    if not warmup_res["warmup_reached_target"]:
        return res

    measurement_start_wall = time.perf_counter()

    sample_dt = cfg.event_frac / (2.0 * cfg.b) # стационарное состояние: d + d'n ~ b -> с частицей происходит что-то раз в 2b времени

    pilot_samples = collect_density_samples_time(sim, cfg, int(cfg.pilot_samples_multiple * pop_exp), sample_dt)
    tau_int = estimate_autocorrelation_time(pilot_samples, max_lag=cfg.pilot_max_lag)

    batch_size = max(cfg.min_batch_size, int(math.ceil(cfg.batch_tau_multiple * tau_int)))

    batch_means: list[float] = []
    measurement_sim_time = cfg.pilot_samples_multiple * sample_dt
    converged = False
    ci_target = np.inf
    is_strict = False

    for _ in range(cfg.max_batches):
        batch_samples = collect_density_samples_time(sim, cfg, batch_size, sample_dt)
        batch_means.append(float(batch_samples.mean()))
        measurement_sim_time += batch_size * sample_dt

        if len(batch_means) < cfg.min_batches:
            continue

        density_mean, _, density_half_width, _, _ = (
            compute_mean_and_ci_from_batch_means(batch_means, cfg.ci_confidence)
        )

        ci_target, is_strict = compute_ci_target(cfg, density_mean, n_exp)
        converged = density_half_width <= ci_target
        if converged:
            break

    measurement_time_wall = time.perf_counter() - measurement_start_wall

    density_mean, density_mean_se, density_half_width, density_lo, density_hi = (
        compute_mean_and_ci_from_batch_means(batch_means, cfg.ci_confidence)
    )

    res.update(
        {
            "density_mean": density_mean,
            "density_mean_se": density_mean_se,
            "density_half_width": density_half_width,
            "density_ci_lower": density_lo,
            "density_ci_upper": density_hi,
            "tau_int": tau_int,
            "batch_size": batch_size,
            "batches_used": len(batch_means),
            "sample_dt": sample_dt,
            "measurement_sim_time": measurement_sim_time,
            "measurement_time_wall": measurement_time_wall,
            "ci_target": ci_target,
            "is_strict": is_strict,
            "converged": converged,
        }
    )
    return res



# =============================================================================
# Внешний sweep по sigma x d_prime
# =============================================================================
def run_one_task(pair_cfg: ExperimentConfig, d_val: float, seed: int) -> dict:
    r = run_single_d(d_val, pair_cfg, seed)
    r["sigma"] = pair_cfg.sigma
    r["d_prime"] = pair_cfg.d_prime
    return r


def run_sigma_d_prime_grid(cfg: ExperimentConfig) -> list[dict]:
    points_rows: list[dict] = []

    pair_cfgs = [
        replace(cfg, sigma=sigma_val, d_prime=d_prime, n_jobs=1)
        for sigma_val, d_prime in product(cfg.sigma_values, cfg.d_prime_values)
    ]

    tasks = []
    seed0 = 42
    for pair_idx, pair_cfg in enumerate(pair_cfgs):
        for j, d_val in enumerate(pair_cfg.get_d_values()):
            tasks.append((pair_cfg, d_val, seed0 + 1000 * pair_idx + j))

    flat_results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(run_one_task)(pair_cfg, d_val, seed)
        for pair_cfg, d_val, seed in tasks
    )

    for pair_cfg in pair_cfgs:
        pair_results = [
            r for r in flat_results
            if r["sigma"] == pair_cfg.sigma and r["d_prime"] == pair_cfg.d_prime
        ]
        pair_results.sort(key=lambda r: r["d"])

        for r in pair_results:
            points_rows.append(
                {
                    "sigma": pair_cfg.sigma,
                    "d_prime": pair_cfg.d_prime,
                    "b": pair_cfg.b,
                    "L": pair_cfg.L,
                    "d": r["d"],
                    "mf_density": r["mf_density"],
                    "mf_population": r["mf_population"],
                    "density_mean": r["density_mean"],
                    "density_mean_se": r["density_mean_se"],
                    "density_half_width": r["density_half_width"],
                    "density_ci_lower": r["density_ci_lower"],
                    "density_ci_upper": r["density_ci_upper"],
                    "tau_int": r["tau_int"],
                    "batch_size": r["batch_size"],
                    "batches_used": r["batches_used"],
                    "sample_dt": r["sample_dt"],
                    "measurement_sim_time": r["measurement_sim_time"],
                    "measurement_time_wall": r["measurement_time_wall"],
                    "ci_target": r["ci_target"],
                    "is_strict": r["is_strict"],
                    "converged": r["converged"],
                    "warmup_reached_target": r["warmup_reached_target"],
                    "warmup_time_wall": r["warmup_time_wall"],
                    "warmup_phase1_chunks": r["warmup_phase1_chunks"],
                    "warmup_tau_int": r["warmup_tau_int"],
                    "warmup_window_size": r["warmup_window_size"],
                    "warmup_stability_windows": r["warmup_stability_windows"],
                    "warmup_stable_count": r["warmup_stable_count"],
                }
            )

    return points_rows


# =============================================================================
# Entry point
# =============================================================================
def main(cfg: ExperimentConfig) -> None:
    points_rows = run_sigma_d_prime_grid(cfg)
    
    out_dir = Path(cfg.output_dir)
    pd.DataFrame(points_rows).to_csv(out_dir / cfg.points_filename, index=False)