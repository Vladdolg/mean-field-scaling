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
    warmup_threshold_frac: float = 0.50        # ждём 50% от pop_exp перед Z-тестом
    warmup_min_stable_windows: int = 10        # 10 пройденных проверок подряд
    warmup_event_chunk: int = 5000
    max_warmup_wall_seconds: float = 3000.0

    # Warmup: критерий стационарности через тест ЭКВИВАЛЕНТНОСТИ E[Z] ≈ 0
    warmup_z_min_windows: int = 10          # минимум накопленных окон перед проверкой
    warmup_z_tcrit: float = 2.0         # множитель для полуширины ДИ E[Z]
    warmup_z_resolution: float = 1e-2   # δ = res * b * n̂  — граница допуска на дрейф

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

def compute_cv_estimate(
    n_means: NDArray[np.float64],
    z_means: NDArray[np.float64],
    confidence: float,
) -> dict:
    """
    n̂_CV = (1/m) * Σ_j ( n̄_j - ĉ^(-j) * z̄_j ),

        ĉ^(-j) = Σ_{i≠j} (n̄_i - n̄̄^(-j))(z̄_i - z̄̄^(-j)) / Σ_{i≠j} (z̄_i - z̄̄^(-j))^2

    ĉ^(-j) не использует батч j, поэтому E[ĉ^(-j) * z̄_j] = ĉ^(-j) * E[z̄_j] = 0
    (батчи длиной >> τ_int практически независимы) — смещения нет.

    CI строится по g_j = n̄_j - ĉ^(-j) * z̄_j как по iid-выборке.
    Реализация O(m) через полные суммы, а не O(m^2).
    """
    n = np.asarray(n_means, dtype=np.float64)
    z = np.asarray(z_means, dtype=np.float64)
    m = n.size

    out = {
        "cv_mean": np.nan, "cv_se": np.nan, "cv_half_width": np.inf,
        "cv_lo": -np.inf, "cv_hi": np.inf,
        "c_hat_full": np.nan, "corr_batch": np.nan, "var_reduction": np.nan,
        "z_mean": np.nan, "z_mean_se": np.nan, "z_tstat": np.nan,
    }
    if m < 3:
        return out

    Sn = n.sum()
    Sz = z.sum()
    Szz = float(z @ z)
    Snz = float(n @ z)

    m1 = m - 1
    sn = Sn - n            # суммы без батча j
    sz = Sz - z
    nbar = sn / m1
    zbar = sz / m1
    Sxy = (Snz - n * z) - m1 * nbar * zbar
    Sxx = (Szz - z * z) - m1 * zbar * zbar

    c_loo = np.where(Sxx > 0.0, Sxy / np.where(Sxx > 0.0, Sxx, 1.0), 0.0)
    g = n - c_loo * z

    cv_mean = float(g.mean())
    cv_se = float(g.std(ddof=1) / math.sqrt(m))
    alpha = 1.0 - confidence
    zq = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
    cv_hw = zq * cv_se

    # --- диагностика ---
    var_z = float(z.var(ddof=1))
    var_n = float(n.var(ddof=1))
    c_full = float(np.cov(n, z, ddof=1)[0, 1] / var_z) if var_z > 0.0 else np.nan
    corr_b = float(np.corrcoef(n, z)[0, 1]) if var_z > 0.0 and var_n > 0.0 else np.nan
    var_red = float(g.var(ddof=1) / var_n) if var_n > 0.0 else np.nan

    z_mean = float(z.mean())
    z_se = float(z.std(ddof=1) / math.sqrt(m))
    z_t = z_mean / z_se if z_se > 0.0 else np.nan

    out.update({
        "cv_mean": cv_mean, "cv_se": cv_se, "cv_half_width": cv_hw,
        "cv_lo": cv_mean - cv_hw, "cv_hi": cv_mean + cv_hw,
        "c_hat_full": c_full, "corr_batch": corr_b, "var_reduction": var_red,
        "z_mean": z_mean, "z_mean_se": z_se, "z_tstat": z_t,
    })
    return out


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

    def _return(phase1_chunks, tau_int, window_size, stability_windows,
              z_mean=np.nan, z_se=np.nan, z_t=np.nan, flag=False):
        return {
            "warmup_time_wall": time.perf_counter() - warmup_start,
            "warmup_reached_target": flag,
            "warmup_phase1_chunks": phase1_chunks,
            "warmup_tau_int": tau_int,
            "warmup_window_size": window_size,
            "warmup_stability_windows": stability_windows,
            "warmup_z_mean_final": z_mean,
            "warmup_z_se_final": z_se,
            "warmup_z_tstat_final": z_t,
        }

    # --- Фаза 1: рост до threshold_frac * pop_exp ---
    # Гейт обязателен: при N = 0 имеем Z ≡ 0 тождественно, и Z-тест прошёл бы
    # тривиально на вымершей конфигурации.
    phase1_chunks = 0
    while sim.current_population() < threshold_pop:
        if time.perf_counter() - warmup_start > cfg.max_warmup_wall_seconds:
            return _return(phase1_chunks, np.nan, 0, 0, 0)
        sim.run_events(cfg.warmup_event_chunk)
        phase1_chunks += 1
        if sim.current_population() <= 0:
            spawn_uniform(sim, 0, initial_pop, cfg.L)

    # --- Фаза 2: грубая оценка τ_int ---
    pilot_dt = cfg.event_frac / (2.0 * cfg.b)
    pilot_n, _ = collect_samples_time(
        sim, cfg, int(cfg.pilot_samples_multiple * pop_exp), pilot_dt
    )
    tau_int = estimate_autocorrelation_time(pilot_n, max_lag=cfg.pilot_max_lag)
    window_size = max(cfg.min_batch_size, int(math.ceil(cfg.batch_tau_multiple * tau_int)))

    # --- Фаза 3: тест эквивалентности E[Z] ≈ 0 с НАКОПЛЕНИЕМ ---
    # Окна длиной >> τ_int => оконные средние z̄_w практически независимы.
    # Накапливаем ВСЕ окна: SE(z̄) ∝ 1/√k убывает со временем (сходимость,
    # а не подбрасывание монеты). Останавливаемся, когда ДИ для E[Z] целиком
    # лежит в допуске [-δ, +δ]:
    #       |z̄| + t_crit * SE(z̄)  <=  δ = res * b * n̂
    # Пока система в переходе, |z̄| велико => тест не проходит. В равновесии
    # z̄ -> 0 и SE -> 0, поэтому левая часть -> 0 < δ, и тест проходит
    # детерминированно, как только накопится достаточно окон.
    # SE входит в критерий, поэтому пройти при большом SE нельзя — гарантия
    # разрешения встроена в саму остановку.
    w_min = max(2, cfg.warmup_z_min_windows)
    tcrit = cfg.warmup_z_tcrit
    res = cfg.warmup_z_resolution

    k = 0
    Sz = 0.0
    Szz = 0.0
    Sn = 0.0
    stability_windows = 0
    z_mean = z_se = z_t = np.nan

    while True:
        if time.perf_counter() - warmup_start > cfg.max_warmup_wall_seconds:
            return _return(phase1_chunks, tau_int, window_size,
                         stability_windows, z_mean, z_se, z_t)

        win_n, win_z = collect_samples_time(sim, cfg, window_size, pilot_dt)
        zw = float(win_z.mean())
        nw = float(win_n.mean())
        k += 1
        Sz += zw
        Szz += zw * zw
        Sn += nw
        stability_windows = k

        if k < w_min:
            continue

        z_mean = Sz / k
        var = (Szz - k * z_mean * z_mean) / (k - 1)
        if var < 0.0:
            var = 0.0
        z_se = math.sqrt(var / k)
        z_t = z_mean / z_se if z_se > 0.0 else np.inf

        n_hat = Sn / k
        delta = res * cfg.b * n_hat
        margin = abs(z_mean) + tcrit * z_se

        if delta > 0.0 and margin <= delta:
            return _return(phase1_chunks, tau_int, window_size,
                         stability_windows, z_mean, z_se, z_t, flag=True)


def collect_samples_time(
    sim: SSAState1D,
    cfg: ExperimentConfig,
    n_samples: int,
    dt_sample: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Возвращает (n_samples_arr, z_samples_arr), оба на единицу длины:
        n = N / L
        z = (b*N - D) / L,   D = sim.total_death_rate

    E[z] = 0 точно в стационаре (генератор, применённый к f = N).
    b*N считаем как cfg.b * N (не через sim.total_birth_rate), чтобы не тащить
    накопленный FP-дрейф инкрементальных агрегатов.
    """
    n_arr = np.empty(n_samples, dtype=np.float64)
    z_arr = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        sim.run_until_time(dt_sample)
        pop = sim.current_population()
        n_arr[i] = pop / cfg.L
        z_arr[i] = (cfg.b * pop - sim.total_death_rate) / cfg.L
    return n_arr, z_arr


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
        "cv_density_mean": np.nan,
        "cv_density_mean_se": np.nan,
        "cv_density_half_width": np.nan,
        "cv_density_ci_lower": np.nan,
        "cv_density_ci_upper": np.nan,
        "c_hat_full": np.nan,
        "corr_batch": np.nan,
        "corr_sample": np.nan,
        "var_reduction": np.nan,
        "z_mean": np.nan,
        "z_mean_se": np.nan,
        "z_tstat": np.nan,
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
        "warmup_z_mean_final": np.nan,
        "warmup_z_se_final": np.nan,
        "warmup_z_tstat_final": np.nan,
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
            "warmup_z_mean_final": warmup_res["warmup_z_mean_final"],
            "warmup_z_se_final": warmup_res["warmup_z_se_final"],
            "warmup_z_tstat_final": warmup_res["warmup_z_tstat_final"],
        }
    )
    if not warmup_res["warmup_reached_target"]:
        return res

    measurement_start_wall = time.perf_counter()

    sample_dt = cfg.event_frac / (2.0 * cfg.b)

    pilot_n, pilot_z = collect_samples_time(
        sim, cfg, int(cfg.pilot_samples_multiple * pop_exp), sample_dt
    )
    tau_int = estimate_autocorrelation_time(pilot_n, max_lag=cfg.pilot_max_lag)

    # диагностика: корреляция n и Z на ПОСЭМПЛОВОМ уровне (не влияет на оценку)
    if pilot_n.var() > 0.0 and pilot_z.var() > 0.0:
        corr_sample = float(np.corrcoef(pilot_n, pilot_z)[0, 1])
    else:
        corr_sample = np.nan

    batch_size = max(cfg.min_batch_size, int(math.ceil(cfg.batch_tau_multiple * tau_int)))

    n_batch_means: list[float] = []
    z_batch_means: list[float] = []
    measurement_sim_time = cfg.pilot_samples_multiple * sample_dt
    converged = False
    ci_target = np.inf
    is_strict = False

    for _ in range(cfg.max_batches):
        bn, bz = collect_samples_time(sim, cfg, batch_size, sample_dt)
        n_batch_means.append(float(bn.mean()))
        z_batch_means.append(float(bz.mean()))
        measurement_sim_time += batch_size * sample_dt

        if len(n_batch_means) < cfg.min_batches:
            continue

        cv = compute_cv_estimate(
            np.asarray(n_batch_means), np.asarray(z_batch_means), cfg.ci_confidence
        )
        # остановка теперь по CV-оценке
        ci_target, is_strict = compute_ci_target(cfg, cv["cv_mean"], n_exp)
        converged = cv["cv_half_width"] <= ci_target
        if converged:
            break

    measurement_time_wall = time.perf_counter() - measurement_start_wall

    # наивная оценка — baseline для сравнения (в остановке не участвует)
    density_mean, density_mean_se, density_half_width, density_lo, density_hi = (
        compute_mean_and_ci_from_batch_means(n_batch_means, cfg.ci_confidence)
    )
    cv = compute_cv_estimate(
        np.asarray(n_batch_means), np.asarray(z_batch_means), cfg.ci_confidence
    )

    res.update(
        {
            "density_mean": density_mean,
            "density_mean_se": density_mean_se,
            "density_half_width": density_half_width,
            "density_ci_lower": density_lo,
            "density_ci_upper": density_hi,
            "cv_density_mean": cv["cv_mean"],
            "cv_density_mean_se": cv["cv_se"],
            "cv_density_half_width": cv["cv_half_width"],
            "cv_density_ci_lower": cv["cv_lo"],
            "cv_density_ci_upper": cv["cv_hi"],
            "c_hat_full": cv["c_hat_full"],
            "corr_batch": cv["corr_batch"],
            "corr_sample": corr_sample,
            "var_reduction": cv["var_reduction"],
            "z_mean": cv["z_mean"],
            "z_mean_se": cv["z_mean_se"],
            "z_tstat": cv["z_tstat"],
            "tau_int": tau_int,
            "batch_size": batch_size,
            "batches_used": len(n_batch_means),
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
                    "cv_density_mean": r["cv_density_mean"],
                    "cv_density_mean_se": r["cv_density_mean_se"],
                    "cv_density_half_width": r["cv_density_half_width"],
                    "cv_density_ci_lower": r["cv_density_ci_lower"],
                    "cv_density_ci_upper": r["cv_density_ci_upper"],
                    "c_hat_full": r["c_hat_full"],
                    "corr_batch": r["corr_batch"],
                    "corr_sample": r["corr_sample"],
                    "var_reduction": r["var_reduction"],
                    "z_mean": r["z_mean"],
                    "z_mean_se": r["z_mean_se"],
                    "z_tstat": r["z_tstat"],
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
                    "warmup_z_mean_final": r["warmup_z_mean_final"],
                    "warmup_z_se_final": r["warmup_z_se_final"],
                    "warmup_z_tstat_final": r["warmup_z_tstat_final"],
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