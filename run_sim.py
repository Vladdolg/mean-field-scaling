from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from scipy.stats import halfnorm, t as student_t

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
import pandas as pd

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
    warmup_event_chunk: int = 5000
    max_warmup_batches: int = 10000

    # Warmup: критерий стационарности через тест ЭКВИВАЛЕНТНОСТИ E[Z] ≈ 0
    warmup_z_min_windows: int = 20          # минимум накопленных окон перед проверкой
    warmup_duration_safety: float = 3.0     # запас к k_min
    warmup_z_confidence: float = 0.95
    warmup_z_resolution: float = 1e-3   # δ = res * b * n̂  — граница допуска на дрейф

    # Измерения по физическому времени
    event_frac: float = 0.10

    # Pilot / batch means
    sokal_multiple: float = 20.0
    pilot_target_ratio: float = 50.0     # требуемое N/window (бывшее tau_n_over_w)
    pilot_chunk_samples: int = 20000     # стартовый чанк
    pilot_max_doublings: int = 6         # потолок числа итераций
    pilot_size_safety: float = 1.3       # запас при прыжке к целевому размеру

    batch_tau_multiple: float = 10.0
    min_batch_size: int = 50

    # Измерение c
    calibration_batches: int = 30   

    # Адаптивная остановка
    min_batches: int = 50
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
def estimate_autocorrelation_time(samples, c: float = 20.0) -> tuple[float, int]:
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    if n < 8:
        return 1.0, 1

    x = x - x.mean()
    nfft = 1 << (int(np.ceil(np.log2(n))) + 1)
    spec = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(spec * np.conj(spec), nfft)[:n].real / n
    if not np.isfinite(acov[0]) or acov[0] <= 0.0:
        return 1.0, 1
    rho = acov / acov[0]

    tau = 1.0
    window = n // 2                 # окно не найдено -> n/W = 2, флаг сам покажет
    for k in range(1, n // 2):
        tau += 2.0 * rho[k]
        if k >= c * tau:
            window = k
            break
    return max(1.0, float(tau)), window

def estimate_tau_adaptive(
    sim: SSAState1D,
    cfg: ExperimentConfig,
    dt_sample: float,
) -> dict:
    """
    Адаптивная оценка τ_int вместо фиксированных pilot_samples_multiple * pop_exp.

    Требуемая длина ряда пропорциональна САМОМУ τ_int, а не размеру популяции:
    окно Сокола W ≈ sokal_multiple * τ_int, а качество оценки задаётся отношением
    N/W. Привязка к pop_exp с этим не связана: в тестах она дала N/W = 66 при
    d = 0.01 и 35 при d = 0.1 (ниже стандарта), а при d' = 0.1 заложила бы
    ~10^6 сэмплов при потребности на порядок меньшей.

    Схема: набрать чанк, оценить τ и окно, проверить N/W >= pilot_target_ratio.
    Если мало — набрать НОВЫЙ чанк размера max(2*текущий, safety*ratio*W) и
    оценить заново. Прыжок сразу к нужному размеру экономит итерации против
    слепого удвоения; множитель 2 остаётся нижней границей на случай, если τ
    был занижен.

    Оценка КАЖДЫЙ РАЗ идёт только по последнему чанку:
      * в фазе 2 ранние сэмплы сняты с ещё неравновесной системы и завышают τ
        (в тестах 155 против 74 и 258 против 128);
      * при коротком ряде окно Сокола не находится, estimate_autocorrelation_time
        возвращает window = n//2, то есть ratio = 2, критерий не проходит и
        происходит добор. Ошибка «мало данных» самокорректируется.

    Цена tail-only — ранние чанки выбрасываются; при прыжке к цели это обычно
    один стартовый чанк, то есть накладные ~ pilot_chunk_samples.
    """
    n_target = int(cfg.pilot_chunk_samples)
    total_samples = 0
    iterations = 0
    tau_int = np.nan
    window = 0
    ratio = np.nan
    ok = False
    chunk_n = np.empty(0, dtype=np.float64)
    chunk_z = np.empty(0, dtype=np.float64)

    for _ in range(cfg.pilot_max_doublings + 1):
        chunk_n, chunk_z = collect_samples_time(sim, cfg, n_target, dt_sample)
        total_samples += n_target
        iterations += 1

        tau_int, window = estimate_autocorrelation_time(chunk_n, c=cfg.sokal_multiple)
        window = max(1, int(window))
        ratio = chunk_n.size / window

        if ratio >= cfg.pilot_target_ratio:
            ok = True
            break

        need = int(math.ceil(cfg.pilot_size_safety * cfg.pilot_target_ratio * window))
        n_target = max(2 * n_target, need)

    return {
        "tau_int": float(tau_int),
        "window": window,
        "ratio": float(ratio),
        "n_chunk": int(chunk_n.size),
        "n_total": int(total_samples),
        "iterations": iterations,
        "converged": ok,
        "pilot_n": chunk_n,
        "pilot_z": chunk_z,
    }

def batch_means_lag1(batch_means):
    x = np.asarray(batch_means, dtype=np.float64)
    m = x.size
    if m < 8:
        return float("nan")
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    rho1 = float(np.dot(x[:-1], x[1:]) / denom)
    return rho1

def t_quantile(confidence: float, df: int) -> float:
    """
    Двусторонний квантиль Стьюдента t_{(1+confidence)/2, df}.
    Используется вместо нормального, потому что дисперсия оценивается
    по той же выборке. df < 1 -> интервал не определён.
    """
    if df < 1:
        return float("inf")
    return float(student_t.ppf(0.5 * (1.0 + confidence), df))

def estimate_cv_coefficient(
    n_means: NDArray[np.float64], z_means: NDArray[np.float64]
) -> float:
    """
    Наклон регрессии n̄ на z̄ по КАЛИБРОВОЧНЫМ батчам: c = Cov(n̄, z̄) / Var(z̄).
 
    Оценивается по данным, независимым от измерительных батчей, и далее фиксируется.
    Несмещённость n̂_CV верна при ЛЮБОМ константном c, поскольку E[z̄_j] = 0
    в стационаре; точность c влияет только на величину снижения дисперсии
    (раздутие ~ 1 + 1/calib_batches). При Var(z̄) = 0 возвращается 0.0 —
    CV-оценка вырождается в наивную, что безопасно.
    """
    n = np.asarray(n_means, dtype=np.float64)
    z = np.asarray(z_means, dtype=np.float64)
    if n.size < 3:
        return 0.0
    var_z = float(z.var(ddof=1))
    if not np.isfinite(var_z) or var_z <= 0.0:
        return 0.0
    c = float(np.cov(n, z, ddof=1)[0, 1] / var_z)
    return c if np.isfinite(c) else 0.0

def compute_mean_and_ci_from_batch_means(
    batch_means: list[float], confidence: float
) -> dict:
    arr = np.asarray(batch_means)

    out = {
        "density_mean": np.nan, 
        "density_mean_se": np.nan,
        "density_half_width": np.inf,
        "density_ci_lower": -np.inf, 
        "density_ci_upper": np.inf,
        }
    
    mean = float(arr.mean())
    out.update({"density_mean": mean})
    if arr.size < 2:
        return out
    
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    half_width = t_quantile(confidence, arr.size - 1) * se
    
    out.update({ 
        "density_mean_se": se,
        "density_half_width": half_width,
        "density_ci_lower": mean - half_width, 
        "density_ci_upper": mean + half_width,
        })
    return out

def compute_cv_estimate(
    n_means: NDArray[np.float64],
    z_means: NDArray[np.float64],
    confidence: float,
    c_fixed: float,
) -> dict:
    """
    n̂_CV = (1/m) * Σ_j g_j,   g_j = n̄_j - c * z̄_j,   где c — КОНСТАНТА.
 
    c оценён заранее по независимому калибровочному блоку (estimate_cv_coefficient)
    и здесь не пересчитывается. Отсюда два свойства:
 
      * смещения нет при любом c: E[g_j] = E[n̄_j] - c * E[z̄_j] = N,
        поскольку в стационаре E[z̄_j] = 0;
      * g_j независимы между собой (батчи независимы, а c не является функцией
        данных измерения), поэтому SE = s_g / √m законна, а df = m - 1.
 
    Прежняя версия оценивала c leave-one-out по самим измерительным батчам.
    Это тоже убирало смещение, но делало g_j взаимно зависимыми через общий c,
    из-за чего SE была занижена на O(1/m) и требовала df = m - 2.
    """
    n = np.asarray(n_means, dtype=np.float64)
    z = np.asarray(z_means, dtype=np.float64)
    m = n.size
 
    out = {
        "cv_mean": np.nan, "cv_se": np.nan, "cv_half_width": np.inf,
        "cv_lower": -np.inf, "cv_upper": np.inf,
        "c_used": float(c_fixed),
        "c_full": np.nan, "corr_batch": np.nan, "var_reduction": np.nan,
        "z_mean": np.nan, "z_mean_se": np.nan, "z_tstat": np.nan,
    }
    if m < 2:
        return out
 
    g = n - c_fixed * z
 
    cv_mean = float(g.mean())
    cv_se = float(g.std(ddof=1) / math.sqrt(m))
    cv_hw = t_quantile(confidence, m - 1) * cv_se
 
    # --- диагностика (в саму оценку НЕ входит) ---
    var_z = float(z.var(ddof=1))
    var_n = float(n.var(ddof=1))
    # c_full — апостериорно оптимальный наклон по измерительным батчам.
    # Служит проверкой качества калибровки: c_used должен быть близок к c_full.
    c_full = float(np.cov(n, z, ddof=1)[0, 1] / var_z) if var_z > 0.0 else np.nan
    corr_b = float(np.corrcoef(n, z)[0, 1]) if var_z > 0.0 and var_n > 0.0 else np.nan
    var_red = float(g.var(ddof=1) / var_n) if var_n > 0.0 else np.nan
 
    z_mean = float(z.mean())
    z_se = float(z.std(ddof=1) / math.sqrt(m))
    z_t = z_mean / z_se if z_se > 0.0 else np.nan
 
    out.update({
        "cv_mean": cv_mean, "cv_se": cv_se, "cv_half_width": cv_hw,
        "cv_lower": cv_mean - cv_hw, "cv_upper": cv_mean + cv_hw,
        "c_full": c_full, "corr_batch": corr_b, "var_reduction": var_red,
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
                tau_n_over_w=np.nan, pilot_samples=np.nan, pilot_iterations=np.nan, pilot_converged=np.nan,
                z_mean=np.nan, z_se=np.nan, z_t=np.nan,
                A_est=np.nan, k_min=np.nan, flag=False):
        return {
            "warmup_time_wall": time.perf_counter() - warmup_start,
            "warmup_reached_target": flag,
            "warmup_phase1_chunks": phase1_chunks,
            "warmup_tau_int": tau_int,
            "warmup_tau_n_over_w": tau_n_over_w,
            "warmup_pilot_samples": pilot_samples, 
            "warmup_pilot_iterations": pilot_iterations, 
            "warmup_pilot_converged": pilot_converged,
            "warmup_window_size": window_size,
            "warmup_stability_windows": stability_windows,
            "warmup_z_mean_final": z_mean,
            "warmup_z_se_final": z_se,
            "warmup_z_tstat_final": z_t,
            "warmup_A_est": A_est,
            "warmup_k_min": k_min,
        }

    # --- Фаза 1: рост до threshold_frac * pop_exp ---
    # Гейт обязателен: при N = 0 имеем Z ≡ 0 тождественно, и Z-тест прошёл бы
    # тривиально на вымершей конфигурации.
    phase1_chunks = 0
    while sim.current_population() < threshold_pop:
        if phase1_chunks > cfg.max_warmup_batches:
            return _return(phase1_chunks, np.nan, 0, 0)
        sim.run_events(cfg.warmup_event_chunk)
        phase1_chunks += 1
        if sim.current_population() <= 0:
            spawn_uniform(sim, 0, initial_pop, cfg.L)

    # --- Фаза 2: грубая оценка τ_int ---
    pilot_dt = cfg.event_frac / (2.0 * cfg.b)
    wp = estimate_tau_adaptive(sim, cfg, pilot_dt)
    tau_int = wp["tau_int"]
    window_size = max(cfg.min_batch_size, int(math.ceil(cfg.batch_tau_multiple * tau_int)))

    
    # --- Фаза 3: контроль ОСТАТОЧНОГО ДРЕЙФА ---
    # E[Z] = dE[n]/dt точно (генератор, применённый к N). Поэтому накопленное
    # z̄ за фазу 3 равно полному изменению плотности, делённому на длительность:
    #       z̄ ≈ A / T,   A = |n_eq - n(начало фазы 3)|
    # Отсюда условие |z̄| <= δ = res*b*n̂ есть в точности условие на ДЛИТЕЛЬНОСТЬ:
    #       T >= A/δ,  т.е.  k >= k_min = A / (δ * T_window)
    #
    # SE-член из критерия УБРАН. Он требовал статистического разрешения, а не
    # равновесия, и связывал всё время работы (233 и 384 окна на тестовых точках
    # против 9 и 5, которых требует содержательная часть). Хуже того, он устроен
    # против цели: сигнал в z̄ падает как 1/T, а шум SE — как 1/√T, поэтому чем
    # дольше идёт фаза 3, тем ХУЖЕ тест различает остаточное смещение. z_se и
    # z_tstat считаются по-прежнему, но только как диагностика в CSV.
    #
    # Пол w_min нужен, чтобы k_min не был посчитан по вырожденной выборке:
    # пока окон мало, A_est занижена, и условие прошло бы тривиально.
    w_min = max(2, cfg.warmup_z_min_windows)
    T_window = window_size * pilot_dt

    Sz = 0.0
    Szz = 0.0
    n_hist: list[float] = []
    stability_windows = 0
    z_mean = z_se = z_t = np.nan
    A_est = k_min = np.nan

    while True:
        if stability_windows + phase1_chunks > cfg.max_warmup_batches:
            return _return(phase1_chunks, tau_int, window_size, stability_windows,
                           wp["ratio"], wp["n_total"], wp["iterations"], wp["converged"],
                           z_mean, z_se, z_t, A_est, k_min)

        win_n, win_z = collect_samples_time(sim, cfg, window_size, pilot_dt)
        zw = float(win_z.mean())
        Sz += zw
        Szz += zw * zw
        n_hist.append(float(win_n.mean()))
        stability_windows += 1

        if stability_windows < w_min:
            continue

        z_mean = Sz / stability_windows
        var = (Szz - stability_windows * z_mean * z_mean) / (stability_windows - 1)
        if var < 0.0:
            var = 0.0
        z_se = math.sqrt(var / stability_windows)
        z_t = z_mean / z_se if z_se > 0.0 else np.inf

        n_arr = np.asarray(n_hist)
        n_hat = float(n_arr.mean())
        delta = cfg.warmup_z_resolution * cfg.b * n_hat
        if delta <= 0.0:
            continue

        # A: полное изменение плотности за фазу 3. Равновесие оцениваем по второй
        # половине окон (она свободна от переходного процесса), старт — по первому окну.
        mid = stability_windows // 2
        n_eq_est = float(n_arr[mid:].mean())
        A_est = abs(n_eq_est - n_arr[0])
        k_min = A_est / (delta * T_window)

        if (abs(z_mean) <= delta and stability_windows >= cfg.warmup_duration_safety * k_min):
            return _return(phase1_chunks, tau_int, window_size, stability_windows,
                           wp["ratio"], wp["n_total"], wp["iterations"], wp["converged"],
                           z_mean, z_se, z_t, A_est, k_min, flag=True)
     


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
        "c_used": np.nan,
        "calib_batches": 0,
        "c_full": np.nan,
        "corr_batch": np.nan,
        "corr_sample": np.nan,
        "var_reduction": np.nan,
        "z_mean": np.nan,
        "z_mean_se": np.nan,
        "z_tstat": np.nan,
        "tau_int": np.nan,
        "tau_n_over_w": np.nan,
        "pilot_samples": np.nan, 
        "pilot_iterations": np.nan, 
        "pilot_converged": False,
        "batch_lag1_rho": np.nan,
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
        "warmup_tau_n_over_w": np.nan,
        "warmup_pilot_samples": np.nan, 
        "warmup_pilot_iterations": np.nan, 
        "warmup_pilot_converged": False,
        "warmup_window_size": 0,
        "warmup_stability_windows": 0,
        "warmup_z_mean_final": np.nan,
        "warmup_z_se_final": np.nan,
        "warmup_z_tstat_final": np.nan,
        "warmup_A_est": np.nan,
        "warmup_k_min": np.nan,
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
            "warmup_tau_n_over_w": warmup_res["warmup_tau_n_over_w"],
            "warmup_pilot_samples": warmup_res["warmup_pilot_samples"], 
            "warmup_pilot_iterations": warmup_res["warmup_pilot_iterations"], 
            "warmup_pilot_converged": warmup_res["warmup_pilot_converged"],
            "warmup_window_size": warmup_res["warmup_window_size"],
            "warmup_stability_windows": warmup_res["warmup_stability_windows"],
            "warmup_z_mean_final": warmup_res["warmup_z_mean_final"],
            "warmup_z_se_final": warmup_res["warmup_z_se_final"],
            "warmup_z_tstat_final": warmup_res["warmup_z_tstat_final"],
            "warmup_A_est": warmup_res["warmup_A_est"],
            "warmup_k_min": warmup_res["warmup_k_min"],
        }
    )
    if not warmup_res["warmup_reached_target"]:
        return res

    measurement_start_wall = time.perf_counter()

    sample_dt = cfg.event_frac / (2.0 * cfg.b)

    mp = estimate_tau_adaptive(sim, cfg, sample_dt)
    pilot_n, pilot_z = mp["pilot_n"], mp["pilot_z"]
    tau_int = mp["tau_int"]

    # диагностика: корреляция n и Z на ПОСЭМПЛОВОМ уровне (не влияет на оценку)
    if pilot_n.var() > 0.0 and pilot_z.var() > 0.0:
        corr_sample = float(np.corrcoef(pilot_n, pilot_z)[0, 1])
    else:
        corr_sample = np.nan

    batch_size = max(cfg.min_batch_size, int(math.ceil(cfg.batch_tau_multiple * tau_int)))

    # --- Калибровочный блок: оценка c по НЕЗАВИСИМЫМ батчам ---------------------
    # Эти батчи идут ТОЛЬКО на оценку c и выбрасываются из измерения. Благодаря
    # этому в измерении c — константа, а не функция измерительных данных, поэтому
    # g_j = n̄_j - c*z̄_j строго iid и формула s_g/√m применима без оговорок.
    calib_n: list[float] = []
    calib_z: list[float] = []
    for _ in range(cfg.calibration_batches):
        cn, cz = collect_samples_time(sim, cfg, batch_size, sample_dt)
        calib_n.append(float(cn.mean()))
        calib_z.append(float(cz.mean()))
    c_used = estimate_cv_coefficient(np.asarray(calib_n), np.asarray(calib_z))
    calibration_sim_time = cfg.calibration_batches * batch_size * sample_dt

    n_batch_means: list[float] = []
    z_batch_means: list[float] = []
    measurement_sim_time = mp["n_total"] * sample_dt + calibration_sim_time
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

        cv = compute_cv_estimate(np.asarray(n_batch_means), np.asarray(z_batch_means), cfg.ci_confidence, c_used)
        # остановка теперь по CV-оценке
        ci_target, is_strict = compute_ci_target(cfg, cv["cv_mean"], n_exp)
        converged = cv["cv_half_width"] <= ci_target
        if converged:
            break

    measurement_time_wall = time.perf_counter() - measurement_start_wall

    # наивная оценка — baseline для сравнения (в остановке не участвует)
    n = compute_mean_and_ci_from_batch_means(n_batch_means, cfg.ci_confidence)
    cv = compute_cv_estimate(np.asarray(n_batch_means), np.asarray(z_batch_means), cfg.ci_confidence, c_used)

    rho1 = batch_means_lag1(np.asarray(n_batch_means) - c_used * np.asarray(z_batch_means))

    res.update(
        {
            "density_mean": n["density_mean"],
            "density_mean_se": n["density_mean_se"],
            "density_half_width": n["density_half_width"],
            "density_ci_lower": n["density_ci_lower"],
            "density_ci_upper": n["density_ci_upper"],
            "cv_density_mean": cv["cv_mean"],
            "cv_density_mean_se": cv["cv_se"],
            "cv_density_half_width": cv["cv_half_width"],
            "cv_density_ci_lower": cv["cv_lower"],
            "cv_density_ci_upper": cv["cv_upper"],
            "c_used": cv["c_used"],
            "calib_batches": cfg.calibration_batches,
            "c_full": cv["c_full"],
            "corr_batch": cv["corr_batch"],
            "corr_sample": corr_sample,
            "var_reduction": cv["var_reduction"],
            "z_mean": cv["z_mean"],
            "z_mean_se": cv["z_mean_se"],
            "z_tstat": cv["z_tstat"],
            "tau_int": tau_int,
            "tau_n_over_w": mp["ratio"],
            "pilot_samples": mp["n_total"], 
            "pilot_iterations": mp["iterations"], 
            "pilot_converged": mp["converged"],
            "batch_lag1_rho": rho1,
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
                    "c_used": r["c_used"],
                    "calib_batches": r["calib_batches"],
                    "c_full": r["c_full"],
                    "corr_batch": r["corr_batch"],
                    "corr_sample": r["corr_sample"],
                    "var_reduction": r["var_reduction"],
                    "z_mean": r["z_mean"],
                    "z_mean_se": r["z_mean_se"],
                    "z_tstat": r["z_tstat"],
                    "tau_int": r["tau_int"],
                    "tau_n_over_w": r["tau_n_over_w"],
                    "pilot_samples": r["pilot_samples"], 
                    "pilot_iterations": r["pilot_iterations"], 
                    "pilot_converged": r["pilot_converged"],
                    "batch_lag1_rho": r["batch_lag1_rho"],
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
                    "warmup_tau_n_over_w": r["warmup_tau_n_over_w"],
                    "warmup_pilot_samples": r["warmup_pilot_samples"], 
                    "warmup_pilot_iterations": r["warmup_pilot_iterations"], 
                    "warmup_pilot_converged": r["warmup_pilot_converged"],
                    "warmup_window_size": r["warmup_window_size"],
                    "warmup_stability_windows": r["warmup_stability_windows"],
                    "warmup_z_mean_final": r["warmup_z_mean_final"],
                    "warmup_z_se_final": r["warmup_z_se_final"],
                    "warmup_z_tstat_final": r["warmup_z_tstat_final"],
                    "warmup_A_est": r["warmup_A_est"],
                    "warmup_k_min": r["warmup_k_min"],
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