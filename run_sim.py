from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from scipy.stats import halfnorm, t as student_t, chi2 as chi2_dist

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
    sigma_values: tuple[float, ...] = (0.50, 0.75, 1.15, 1.75, 2.6, 4.00,)
    d_prime_values: tuple[float, ...] = (0.10, 0.15, 0.25, 0.40, 0.65, 1.00,)

    gaussian_cutoff_sigmas: float = 5.0

    # --- Адаптивное окно по d внутри каждой пары (sigma, d_prime) -------------
    # ОДНА фаза. Лестница спускается мелким шагом, и рабочим окном становятся
    # window_points ПОСЛЕДНИХ измеренных ступеней — тех же самых, по которым
    # принято решение об остановке. Отдельной фазы производства нет: раньше
    # критерий считался по ступеням НАД d_max, а рабочие точки ставились ПОД
    # ним, то есть кривизна экстраполировалась в область, где её не мерили.
    #
    # Это ничего не стоит. При window_points=7 и probe_step=1.2 полная лестница
    # стоит 1/(1 - 1.2^-2) = 3.27 от нижней точки, а само окно (7 нижних
    # ступеней) — 3.02. То есть на спуск сверх окна уходит 8% бюджета пары, и
    # держать для него отдельную ослабленную точность смысла нет: все ступени
    # меряются с полным rel_ci_target.

    # Априор для выбора СТАРТА лестницы: kappa ~ kappa_prior_coef * d'/sigma
    # (курсовая, 0.297 на диапазоне d'/sigma в [0.025; 0.385]). Используется
    # ТОЛЬКО чтобы выбрать, где мерить, — в результат не входит.
    kappa_prior_coef: float = 0.297
    probe_target_deficit: float = 0.05   # целевой Δ/n_mf на верхней ступени
    probe_step: float = 1.2              # делитель d на каждой ступени
    probe_max_rungs: int = 40
    window_points: int = 7               # длина окна в ступенях
    # Размах окна = probe_step**(window_points - 1) = 1.2**6 = 2.99. Сужать его
    # нельзя: SE(s) растёт как (1+W)/(W-1), и одновременно растёт остаточное
    # смещение s от отброшенного члена (~1/W при фиксированном верхнем крае).
    # Поэтому probe_step и window_points меняются только вместе, с сохранением
    # W ≈ 3.

    # Сколько ступеней считается параллельно за один заход. Лестница
    # последовательна по смыслу, но при n_jobs>1 простаивать невыгодно: считаем
    # пачку, потом проверяем гейт на всех новых окнах сразу. Перелёт в среднем
    # (probe_batch-1)/2 ступеней — они остаются в таблице с in_window=False.
    probe_batch: int = 4

    # Единственная граница на d — сверху, и она физическая. При d -> b система
    # подходит к границе вымирания, и портятся сразу две вещи, обе
    # пропорциональные (b - d):
    #   * равновесная популяция pop_exp = (b-d)*L/d' -> 0, и warmup-гейт
    #     "50% от pop_exp" начинает ловить вымирающую конфигурацию;
    #   * скорость релаксации к равновесию тоже равна (b - d), то есть warmup
    #     замедляется как 1/(b-d).
    # Отсюда два условия, дающие ОДНО число d_cap = min(...):
    #   pop_exp(d) >= probe_pop_min       <=>  d <= b - probe_pop_min * d' / L
    #   (b - d) / b >= probe_rate_frac    <=>  d <= (1 - probe_rate_frac) * b
    # Снизу границы нет: спуск останавливается сам, а вырожденно дорогая точка
    # помечается converged=False.
    probe_pop_min: int = 500
    probe_rate_frac: float = 0.30

    # --- Гейт остановки -------------------------------------------------------
    # Рабочая модель — Δ = s·d + c·d², в переменной g = Δ/d это g = s + c·d.
    # Квадратика здесь не «второй член разложения в нуле», а способ снять
    # остаточное искривление на окне; чтобы верить полученной κ, нужно, чтобы
    # после её снятия данные были описаны. Отсюда два условия:
    #
    #   1) chi2_lin <= квантиль хи-квадрат(gof_confidence, ndf)
    #      — двухчленная модель ОПИСЫВАЕТ данные. Это общий тест против любого
    #        отклонения, а не против одного конкретного следующего монома.
    #
    #   2) nonlin_ratio + SE <= nonlin_tol,  nonlin_ratio = |c|·d_top/s
    #      — режим пертурбативный, поправка не сравнима с самим сигналом.
    #        Смещение κ это НЕ ограничивает (член c в модели), поэтому и имя
    #        nonlin_tol, а не bias_tol.
    #
    # Отдельного теста на «третий член» нет и оценки возможного смещения тоже.
    # Пробовалось, убрано: порог вида |nu| + 2·SE(nu) <= 0.6 заваливал ЗАВЕДОМО
    # двухчленные данные в 47% случаев, а при выравнивании уровня ложных
    # срабатываний до 5% не давал ничего сверх chi2 и на умеренных амплитудах
    # был слабее него (Монте-Карло, 7 точек, W≈3, rel_ci_target=0.05).
    #
    # Границу применимости стоит держать в голове, но не в таблице: chi2
    # проверяет модель на той точности, которая есть. Отклонение, почти
    # неотличимое от прямой на окне шириной 3 (скажем, sqrt(d)), не видно ни
    # этим тестом, ни каким-либо другим в пределах одного окна — это свойство
    # экстраполяции, а не дефект критерия.
    nonlin_tol: float = 0.15          # допустимое nonlin_ratio + SE
    gof_confidence: float = 0.95      # уровень для хи-квадрат-квантиля

    base_seed: int = 67

    # Warmup
    initial_density_frac: float = 0.1
    warmup_event_chunk: int = 5000
    warmup_chunks: int = 10
    max_warmup_batches: int = 10000

    # Warmup: критерий стационарности через тест ЭКВИВАЛЕНТНОСТИ E[Z] ≈ 0
    warmup_z_min_windows: int = 20          # минимум накопленных окон перед проверкой
    warmup_duration_safety: float = 3.0     # запас к k_min
    warmup_z_resolution: float = 1e-3   # δ = res * b * n̂  — граница допуска на дрейф

    # Измерения по физическому времени
    event_frac: float = 0.10

    # Pilot / batch means
    sokal_multiple: float = 20.0
    pilot_target_ratio: float = 50.0     # требуемое N/window
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
    # Единая точность для ВСЕХ ступеней: окно состоит из тех же точек, по
    # которым принимается решение, поэтому мерить их слабее нечем оправдать.
    #
    # Оба условия гейта упираются в неё напрямую и от абсолютного d не зависят:
    # в g-переменной SE(g) = eps·s постоянна, поэтому и разрешение chi2, и
    # SE(nonlin_ratio) фиксированы точностью, а не глубиной спуска. При W≈3,
    # n=7 это SE(s)/s ≈ 2.9%. Уменьшать rel_ci_target вдвое — вчетверо по
    # времени, и на порог chi2 это не влияет: сходимость даёт спуск, а не
    # точность точки.
    rel_ci_target: float = 0.05
    delta_floor_frac: float = 0.00001  # floor = rel_ci_target * delta_floor_frac * |n_MF|

    # Параллельность
    n_jobs: int = 4

    # Сохранение
    output_dir: str = "."
    points_filename: str = "points.csv"
    pairs_filename: str = "pairs.csv"

    def n_expected(self, d: float) -> float:
        return (self.b - d) / self.d_prime if self.d_prime > 0 else np.nan

    def pop_expected(self, d: float) -> int:
        n_exp = self.n_expected(d)
        if not np.isfinite(n_exp):
            return 0
        return max(0, int(round(n_exp * self.L)))

    def d_cap(self, d_prime: float) -> float:
        """
        Физический потолок на d: минимум из условия на популяцию и условия на
        скорость релаксации. Оба масштабируются как (b - d), но первое зависит
        от d' и L, а второе — нет, поэтому связывает то одно, то другое.
        """
        if d_prime <= 0.0 or self.L <= 0.0:
            return np.nan
        by_pop = self.b - self.probe_pop_min * d_prime / self.L
        by_rate = (1.0 - self.probe_rate_frac) * self.b
        return min(by_pop, by_rate)


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
    Адаптивная оценка τ_int.

    Требуемая длина ряда пропорциональна САМОМУ τ_int, а не размеру популяции:
    окно Сокола W ≈ sokal_multiple * τ_int, а качество оценки задаётся отношением
    N/W.

    Схема: набрать чанк, оценить τ и окно, проверить N/W >= pilot_target_ratio.
    Если мало — набрать НОВЫЙ чанк размера max(2*текущий, safety*ratio*W) и
    оценить заново. Прыжок сразу к нужному размеру экономит итерации против
    слепого удвоения; множитель 2 остаётся нижней границей на случай, если τ
    был занижен.

    Оценка КАЖДЫЙ РАЗ идёт только по последнему чанку:
      * в фазе 2 ранние сэмплы сняты с ещё неравновесной системы и завышают τ;
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
    cfg: ExperimentConfig, density_mean: float, n_mf: float, rel_target: float
) -> tuple[float, bool]:
    """
    Критерий преподавателя: ±rel_target на разницу с mean field.

    CI_half_width ≤ rel_target * max(|delta|, floor)
      где floor = delta_floor_frac * |n_MF|   (чтобы не мерить вечно при delta → 0)

    rel_target приходит аргументом, а не берётся из cfg: точность — свойство
    вызова, а не процедуры. Сейчас все ступени лестницы меряются с одним
    cfg.rel_ci_target, но параметр оставлен явным, чтобы двухуровневую схему
    можно было вернуть, не трогая run_single_d.

    Возвращает (target, is_strict):
      target    — порог для CI_half_width
      is_strict — True если |delta| > floor (т.е. сработал именно относительный
                  критерий, а не floor).
    """
    delta = abs(density_mean - n_mf)
    floor = cfg.delta_floor_frac * abs(n_mf)
    is_strict = delta >= floor
    target = rel_target * max(delta, floor)
    return target, is_strict


# =============================================================================
# Спавн случайных частиц
# =============================================================================
def spawn_uniform(
    sim: SSAState1D, species: int, count: int, L: float, rng: np.random.Generator
) -> int:
    spawned = 0
    for _ in range(count):
        pos = float(rng.uniform(0.0, L))
        if sim.spawn_particle(species, pos):
            spawned += 1
    return spawned


# =============================================================================
# Warmup / sampling
# =============================================================================
def run_warmup(
    sim: SSAState1D, cfg: ExperimentConfig, pop_exp: int, rng: np.random.Generator
) -> dict:
    warmup_start = time.perf_counter()
    initial_pop = max(10, int(cfg.initial_density_frac * pop_exp))
    spawn_uniform(sim, 0, initial_pop, cfg.L, rng)

    def _return(phase1_chunks, tau_int, window_size, stability_windows,
                tau_n_over_w=np.nan, pilot_samples=np.nan, pilot_iterations=np.nan, pilot_converged=False,
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

    # --- Фаза 1: первоначальный рост системы ---
    # Длительность фиксирована и не привязана к pop_exp: порог вида
    # "доля от mean-field популяции" отбирал бы конфигурацию по величине N,
    # а именно отклонение N от mean field и измеряется. Равновесие обеспечивают
    # фаза 2 (пилот длиной ~1000 единиц времени при 1/(b-d) ~ 1) и критерий
    # дрейфа фазы 3; при вымирании n̂ = 0 даёт δ = 0, и warmup честно не сходится.
    phase1_chunks = 0
    while phase1_chunks < cfg.warmup_chunks:
        sim.run_events(cfg.warmup_event_chunk)
        phase1_chunks += 1
        if sim.current_population() <= 0:
            spawn_uniform(sim, 0, initial_pop, cfg.L, rng)

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
    # SE-члена в критерии нет: он требовал бы статистического разрешения, а не
    # равновесия, и работает против цели — сигнал в z̄ падает как 1/T, а шум SE —
    # как 1/√T, поэтому чем дольше идёт фаза 3, тем хуже тест различает
    # остаточное смещение. z_se и z_tstat считаются, но только как диагностика
    # в CSV.
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
def run_single_d(
    d_val: float, cfg: ExperimentConfig, seed: int, rel_ci_target: float
) -> dict:
    """
    rel_ci_target передаётся явно: разведка и производство используют одну и ту
    же процедуру измерения, но с разными требованиями к точности.
    """
    n_exp = cfg.n_expected(d_val)
    pop_exp = cfg.pop_expected(d_val)

    res = {
        "d": d_val,
        "mf_density": n_exp,
        "mf_population": pop_exp,
        "rel_ci_target_used": rel_ci_target,
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
        "is_strict": False,          # True = сработал относительный критерий, False = floor
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

    rng = np.random.default_rng(seed)
    warmup_res = run_warmup(sim, cfg, pop_exp, rng)
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
        # остановка по CV-оценке
        ci_target, is_strict = compute_ci_target(cfg, cv["cv_mean"], n_exp, rel_ci_target)
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
# Адаптивное окно по d: лестница -> rho -> рабочая сетка
# =============================================================================
def derive_seed(cfg: ExperimentConfig, sigma: float, d_prime: float,
                phase_id: int, d_val: float) -> int:
    """
    Seed как чистая функция ФИЗИЧЕСКИХ параметров точки: ключ — (b, sigma,
    d_prime, d, фаза). Точка воспроизводится по своим параметрам, как бы и в
    каком порядке ни запускался расчёт.

    Float'ы кладутся в ключ битовым представлением: точно, без потерь на
    округление и без зависимости от форматирования. Питоновский hash() здесь
    непригоден — он солится на каждый процесс.
    """
    key = [int(cfg.base_seed), int(phase_id)]
    for x in (cfg.b, sigma, d_prime, d_val):
        key.append(int(struct.unpack("<Q", struct.pack("<d", float(x)))[0]))
    return int(np.random.SeedSequence(key).generate_state(1, dtype=np.uint32)[0])


def probe_start_d(cfg: ExperimentConfig, sigma: float, d_prime: float) -> float:
    """
    Верхняя ступень лестницы — НА ПАРУ, а не единая константа.

    Относительный дефицит на ступени d равен

        Δ/n_mf = κ·d / (b - d),   κ ≈ kappa_prior_coef · d'/σ,

    поэтому старт выбирается из условия на ОЖИДАЕМЫЙ дефицит:
        κ·d/(b-d) = probe_target_deficit  =>  d = tgt·b / (κ + tgt).

    Формула сама по себе всегда даёт d < b, но при малых κ подходит к b вплотную
    (для σ=4, d'=0.1 это 0.93·b), а там система стоит у границы вымирания.
    Единственное ограничение сверху — физическое, cfg.d_cap: минимум из условий
    на равновесную популяцию и на скорость релаксации.

    Априор κ используется ТОЛЬКО для выбора места первого измерения; на оценки он
    не влияет: где бы ни стояла верхняя ступень, rho считается по фактически
    измеренным точкам. Промах априора не страшен в обе стороны: слишком высокий
    старт стоит нескольких дешёвых верхних ступеней, слишком низкий — сразу
    удовлетворяет критерию и просто даёт окно шире.
    """
    kappa_pred = cfg.kappa_prior_coef * d_prime / sigma if sigma > 0.0 else np.inf
    tgt = cfg.probe_target_deficit
    denom = kappa_pred + tgt
    if not np.isfinite(denom) or denom <= 0.0:
        return np.nan

    d = tgt * cfg.b / denom
    cap = cfg.d_cap(d_prime)
    if not np.isfinite(cap) or cap <= 0.0:
        return np.nan
    return float(min(d, cap))


def _wls(x: NDArray[np.float64], y: NDArray[np.float64],
         sy: NDArray[np.float64], degree: int) -> tuple | None:
    """
    Взвешенный МНК y = sum_j p_j * x^j, j = 0..degree.

    Веса берутся как ИЗВЕСТНЫЕ дисперсии: ковариация (X^T W X)^{-1} остаточным
    разбросом не домножается. Та же конвенция, что в fit_kappa_per_pair. Иначе
    chi2 потерял бы смысл теста: масштабирование ковариации остатками ровно
    прячет тот промах модели, который этот тест и должен ловить.

    Возвращает (p, cov, chi2, ndf) или None, если система вырождена.
    """
    if x.size < degree + 1:
        return None
    X = np.vander(x, degree + 1, increasing=True)
    w = 1.0 / sy**2
    A = X.T @ (X * w[:, None])
    if not np.all(np.isfinite(A)):
        return None
    try:
        cov = np.linalg.inv(A)
        p = cov @ (X.T @ (w * y))
    except np.linalg.LinAlgError:
        return None
    if not (np.all(np.isfinite(cov)) and np.all(np.isfinite(p))):
        return None
    resid = (y - X @ p) / sy
    return p, cov, float(resid @ resid), int(x.size - degree - 1)


def _ratio_se(num: float, den: float, var_num: float, var_den: float,
              cov: float, scale: float) -> float:
    """
    SE(scale * num / den) по полной дельта-формуле.

    Нужна для nonlin_ratio = |c|·d_top/s — отношения, у которого шумят и
    числитель, и знаменатель.

    Члены с var_den и cov оставлены, хотя решают они мало: у окна вблизи гейта
    поправка к SE составляет единицы процентов (измерено на семи точках, от
    -0.3% до +1.5%), до 25-40% она дорастает только на окнах, которые гейт
    заваливает с большим запасом, то есть там, где это уже ничего не меняет.
    Держатся они не ради величины эффекта, а потому что формула для отношения
    без них просто неверна, а стоят ноль.
    """
    if den == 0.0 or not np.isfinite(den):
        return float("nan")
    var = (var_num / den**2
           + num**2 * var_den / den**4
           - 2.0 * num * cov / den**3)
    return float(scale * math.sqrt(var)) if var > 0.0 else float("nan")


def window_metrics(ds: list[float], deltas: list[float],
                   ses: list[float]) -> dict:
    """
    Все метрики окна по его собственным точкам.

    Дефицит Δ(d) = s·d + c·d² + e·d³, делим на d:
        g(d) = Δ(d)/d = s + c·d + e·d².
    Регрессоры (1, d, d²) в такой форме обусловлены лучше, чем фит Δ через ноль,
    веса SE(g) = SE(Δ)/d. Свободный член в Δ не вводится: Δ(0) = 0 доказано
    точно (при d = 0 процесс пуассоновский и N(0) = b/d'), поэтому лишний
    параметр только съел бы изгиб и замаскировал неадекватность модели.

    Рабочая модель одна — двухчленная (s, c). Метрика режима:
        nonlin_ratio = |c|·d_top / s   — размер поправки на верхнем крае окна.
    Адекватность модели проверяется chi2_lin против квантиля, см. gate_passed.
    """
    out = {
        "win_s": np.nan, "win_s_se": np.nan,
        "win_c": np.nan, "win_c_se": np.nan,
        "rho": np.nan, "rho_se": np.nan,
        "nonlin_ratio": np.nan, "nonlin_ratio_se": np.nan,
        "chi2_lin": np.nan, "ndf_lin": 0,
        "win_points": 0,
        "win_d_top": np.nan, "win_d_bot": np.nan, "win_ratio": np.nan,
    }

    d = np.asarray(ds, dtype=np.float64)
    delta = np.asarray(deltas, dtype=np.float64)
    se = np.asarray(ses, dtype=np.float64)
    if d.size == 0:
        return out

    ok = (np.isfinite(d) & np.isfinite(delta) & np.isfinite(se)
          & (se > 0.0) & (d > 0.0))
    out["win_points"] = int(ok.sum())
    if ok.sum() < 3:
        return out

    x = d[ok]
    g = delta[ok] / x
    sg = se[ok] / x
    d_top = float(x.max())
    out["win_d_top"] = d_top
    out["win_d_bot"] = float(x.min())
    out["win_ratio"] = d_top / float(x.min())

    lin = _wls(x, g, sg, 1)
    if lin is not None:
        p, cov, chi2, ndf = lin
        s_c, c_c = float(p[0]), float(p[1])
        out.update({
            "win_s": s_c, "win_s_se": float(math.sqrt(cov[0, 0])),
            "win_c": c_c, "win_c_se": float(math.sqrt(cov[1, 1])),
            "chi2_lin": chi2, "ndf_lin": ndf,
        })
        if np.isfinite(s_c) and s_c != 0.0 and np.isfinite(c_c):
            out["rho"] = float(abs(c_c / s_c))
            out["rho_se"] = _ratio_se(c_c, s_c, cov[1, 1], cov[0, 0],
                                      cov[0, 1], 1.0)
            out["nonlin_ratio"] = out["rho"] * d_top
            out["nonlin_ratio_se"] = out["rho_se"] * d_top

    return out


def rungs_remaining(m: dict, cfg: ExperimentConfig) -> int:
    """
    Прогноз: сколько ещё ступеней до срабатывания гейта.

    nonlin_ratio = |c|·d_top/s пропорционален d_top, то есть падает ровно в
    probe_step раз за ступень, поэтому число ступеней до порога считается в лоб:
        n = ceil( log((nonlin+SE)/nonlin_tol) / log(probe_step) ).

    Нужно это только чтобы сжимать пачку на подходе к остановке. Прогноз
    систематически ЗАВЫШЕН (второе условие гейта, chi2, он не учитывает вовсе,
    а SE в числителе почти не падает) — измерено: завышает в 9 окнах из 13 на
    коротком спуске и в 17 из 22 на длинном. Для своей задачи это правильная
    сторона ошибки: пачка сжимается позже, чем можно, но не раньше, чем нужно,
    а преждевременное сжатие стоит всего параллелизма на самых дорогих
    ступенях.

    Возвращает большое число, если окна ещё нет или метрика не посчиталась —
    тогда пачка берётся полной.
    """
    nl, nl_se = m.get("nonlin_ratio"), m.get("nonlin_ratio_se")
    if nl is None or nl_se is None or not (np.isfinite(nl) and np.isfinite(nl_se)):
        return 1 << 30
    top = nl + nl_se
    if top <= cfg.nonlin_tol:
        return 1
    step = math.log(cfg.probe_step)
    if step <= 0.0:
        return 1 << 30
    return max(1, int(math.ceil(math.log(top / cfg.nonlin_tol) / step)))


def gate_passed(m: dict, cfg: ExperimentConfig) -> bool:
    """
    Окно принимается, если выполнено И то, и другое:

      1) двухчленная модель описывает данные:
             chi2_lin <= chi2_ppf(gof_confidence, ndf_lin)
      2) режим пертурбативный:
             nonlin_ratio + SE(nonlin_ratio) <= nonlin_tol

    Условие (1) — то самое «описывает ли модель данные», и порог у него не
    произвольный допуск, а квантиль. «Не отвергли» означает «промах модели
    меньше разрешения при этой точности» — не больше и не меньше того.

    Условие (2) берётся с верхней границей, а не с точечной оценкой:
    неразрешённая кривизна даёт большой SE и заставляет спуститься ещё на
    ступень, вместо того чтобы пройти гейт по недостатку мощности.

    Оба условия сходятся при спуске: в g-переменной SE постоянна (SE(g) ≈ eps·s
    не зависит от d), а отклонение от прямой на окне падает как d_top².
    """
    vals = (m.get("nonlin_ratio"), m.get("nonlin_ratio_se"), m.get("chi2_lin"))
    if not all(v is not None and np.isfinite(v) for v in vals):
        return False
    if int(m.get("ndf_lin", 0)) < 1:
        return False
    if m["chi2_lin"] > chi2_dist.ppf(cfg.gof_confidence, int(m["ndf_lin"])):
        return False
    if m["nonlin_ratio"] + m["nonlin_ratio_se"] > cfg.nonlin_tol:
        return False
    return True


# =============================================================================
# Внешний sweep по sigma x d_prime
# =============================================================================
def run_one_task(pair_cfg: ExperimentConfig, d_val: float, seed: int,
                 rel_ci_target: float, phase: str) -> dict:
    r = run_single_d(d_val, pair_cfg, seed, rel_ci_target)
    r["sigma"] = pair_cfg.sigma
    r["d_prime"] = pair_cfg.d_prime
    r["b"] = pair_cfg.b
    r["L"] = pair_cfg.L
    r["seed"] = seed
    r["phase"] = phase
    return r


def blank_pair_row(cfg: ExperimentConfig, sigma_val: float, d_prime_val: float) -> dict:
    """
    Строка таблицы пар. Существует у КАЖДОЙ пары, включая те, где окно не
    найдено: иначе из таблицы точек не понять, почему пара исчезла.

    Ключи блока метрик берутся из самой window_metrics, чтобы список не
    разъезжался при её правке.
    """
    row = {
        "sigma": sigma_val,
        "d_prime": d_prime_val,
        "b": cfg.b,
        "L": cfg.L,
        "window_ok": False,
        "stop_reason": "not_started",
        "rungs": 0,
        "ladder_d_start": np.nan,
        "ladder_d_top": np.nan,
        "ladder_d_bottom": np.nan,
        "ladder_d_cap": cfg.d_cap(d_prime_val),
    }
    row.update(window_metrics([], [], []))
    row.update({
        "kappa": np.nan,
        "kappa_se": np.nan,
        "pair_time_wall": 0.0,
    })
    return row


def _parallel_generator(cfg: ExperimentConfig, jobs: list):
    """
    Результаты по мере готовности, а не одним пакетом в конце.

    generator_unordered отдаёт точку сразу, как только она досчиталась, что и
    нужно для сохранения после каждой точки. На joblib < 1.4 его нет — там
    откатываемся на упорядоченный генератор: он тоже инкрементальный, но
    придерживает готовые результаты, пока не завершится более ранняя задача.
    """
    for mode in ("generator_unordered", "generator"):
        try:
            return Parallel(n_jobs=cfg.n_jobs, return_as=mode)(jobs)
        except (TypeError, ValueError):
            continue
    return iter(Parallel(n_jobs=cfg.n_jobs)(jobs))


def run_one_pair(cfg: ExperimentConfig, sigma_val: float, d_prime_val: float,
                 points_rows: list[dict], pairs_rows: list[dict],
                 flush=None) -> dict:
    """
    Одна фаза на пару (sigma, d_prime): лестница сверху вниз шагом probe_step.

    Рабочим окном становятся window_points ПОСЛЕДНИХ измеренных ступеней — те
    же самые точки, по которым проверен гейт. Никакой экстраполяции: и решение,
    и результат живут на одном интервале по d.

    Гейт (см. gate_passed) проверяется на каждом новом окне по мере спуска.
    Ступени считаются пачками по probe_batch штук, чтобы не простаивали ядра;
    внутри пачки проверок нет, поэтому после неё окна перебираются в порядке
    убывания d_bot и берётся ПЕРВОЕ прошедшее — то есть перелёт пачки на выбор
    окна не влияет, он стоит только времени. Лишние нижние ступени остаются в
    таблице точек с in_window=False.

    Исход бинарный. Пара останавливается без окна, если очередная ступень не
    измерилась (не сошёлся warmup) или кончились ступени; измеренные точки в
    таблице остаются как есть.

    Строка пары обновляется после каждой точки, точки пишутся сразу.
    """
    pair_start = time.perf_counter()
    pair_cfg = replace(cfg, sigma=sigma_val, d_prime=d_prime_val)

    pair_row = blank_pair_row(cfg, sigma_val, d_prime_val)
    pairs_rows.append(pair_row)

    def touch() -> None:
        pair_row["pair_time_wall"] = time.perf_counter() - pair_start
        flush()

    d_start = probe_start_d(cfg, sigma_val, d_prime_val)
    pair_row["ladder_d_start"] = d_start
    if not np.isfinite(d_start) or d_start <= 0.0:
        pair_row["stop_reason"] = "bad_start"
        touch()
        return pair_row

    k = max(3, int(cfg.window_points))
    batch = max(1, int(cfg.probe_batch))

    # d ступени — ЧИСТАЯ функция её номера: d_j = d_start / probe_step**j.
    # Накапливать делением нельзя: d/q/q и d/q**2 различаются в последнем бите,
    # а derive_seed берёт биты float, поэтому от размера пачки менялся бы весь
    # поток случайных чисел и вместе с ним выбранное окно.
    def rung_d(j: int) -> float:
        return float(d_start / cfg.probe_step**j)

    rows: list[dict] = []      # измеренные строки, по убыванию d
    ds: list[float] = []
    deltas: list[float] = []
    ses: list[float] = []
    scanned = 0                # сколько окон уже проверено
    metrics = window_metrics([], [], [])
    win_lo = -1                # индекс нижней точки принятого окна
    stop_reason = "max_rungs"

    next_rung = 0
    while next_rung < cfg.probe_max_rungs and win_lo < 0:
        # Размер пачки сжимается на подходе к остановке: лишние ступени лежат
        # НИЖЕ дна окна, а стоимость точки ~1/d², поэтому перелёт на три
        # ступени при шаге 1.2 стоит втрое дороже всего остального спуска.
        # Измерено на сквозной симуляции (40 сидов, выбранное окно то же):
        # -21% CPU и -1% стенного на коротком спуске, -44% и -32% на длинном.
        n_new = min(batch, rungs_remaining(metrics, cfg),
                    cfg.probe_max_rungs - next_rung)
        batch_ds = [rung_d(next_rung + i) for i in range(n_new)]
        next_rung += n_new
        jobs = [
            delayed(run_one_task)(pair_cfg, d_val,
                                  derive_seed(cfg, sigma_val, d_prime_val, 0, d_val),
                                  cfg.rel_ci_target, "ladder")
            for d_val in batch_ds
        ]

        done: list[dict] = []
        for r in _parallel_generator(cfg, jobs):
            r["in_window"] = False
            done.append(r)
            points_rows.append(r)
            pair_row["rungs"] = len(rows) + len(done)
            touch()

        done.sort(key=lambda r: -float(r["d"]))

        failed = False
        for r in done:
            # В метрики идёт CV-оценка: она и есть рабочий оценщик плотности,
            # наивная остаётся в таблице только как диагностика.
            measured = (r["warmup_reached_target"]
                        and r["converged"]
                        and r["is_strict"]
                        and np.isfinite(r["cv_density_mean"])
                        and np.isfinite(r["cv_density_mean_se"])
                        and r["cv_density_mean_se"] > 0.0)
            if not measured:
                failed = True
                break
            rows.append(r)
            ds.append(float(r["d"]))
            deltas.append(float(r["mf_density"] - r["cv_density_mean"]))
            ses.append(float(r["cv_density_mean_se"]))

        # Окна перебираются снизу вверх по d_bot: первое прошедшее и есть ответ.
        while scanned + k <= len(ds):
            lo = scanned
            m = window_metrics(ds[lo:lo + k], deltas[lo:lo + k], ses[lo:lo + k])
            metrics = m
            scanned += 1
            if gate_passed(m, cfg):
                win_lo = lo
                stop_reason = "gate"
                break

        if failed:
            if win_lo < 0:
                stop_reason = "point_failed"
            break

    pair_row.update(metrics)
    pair_row.update({
        "window_ok": win_lo >= 0,
        "stop_reason": stop_reason,
        "rungs": len(rows),
        "ladder_d_top": float(max(ds)) if ds else np.nan,
        "ladder_d_bottom": float(min(ds)) if ds else np.nan,
    })

    if win_lo >= 0:
        for r in rows[win_lo:win_lo + k]:
            r["in_window"] = True
        # kappa = s * d'. Больше в строку пары ничего не добавляется: окно
        # принято потому, что модель его описывает, и приписывать к результату
        # оценку «а вдруг всё-таки смещено» значило бы спорить с собственным
        # критерием приёмки.
        pair_row["kappa"] = float(metrics["win_s"] * d_prime_val)
        pair_row["kappa_se"] = float(metrics["win_s_se"] * d_prime_val)

    touch()
    return pair_row


def save_table(rows: list[dict], path: Path, sort_by: list[str]) -> None:
    """
    Атомарная запись таблицы: пишем во временный файл и подменяем им целевой.
    Так частично записанный CSV никогда не окажется на месте готового, даже
    если процесс убьют в момент сохранения.

    Таблица переписывается целиком, а не дописывается. При сотнях строк это
    ничего не стоит, зато набор и порядок колонок гарантированно одинаковы во
    всех строках, а сортировка остаётся корректной после каждой точки.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = [c for c in sort_by if c in df.columns]
    if cols:
        df = df.sort_values(cols, kind="stable")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def run_sigma_d_prime_grid(cfg: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    """
    Пары считаются ПОСЛЕДОВАТЕЛЬНО, ядра забирает внутренний параллелизм пары:
    при паре в масштабе суток раздача по паре на ядро означала бы, что прогон из
    одной пары занимает одно ядро из cfg.n_jobs, а остальные простаивают.

    Обе таблицы сбрасываются на диск после КАЖДОЙ ТОЧКИ: при часах на точку
    падение на середине не должно стоить ничего из уже посчитанного.
    """
    out_dir = Path(cfg.output_dir)
    points_path = out_dir / cfg.points_filename
    pairs_path = out_dir / cfg.pairs_filename

    points_rows: list[dict] = []
    pairs_rows: list[dict] = []

    def flush() -> None:
        save_table(points_rows, points_path, ["sigma", "d_prime", "d"])
        save_table(pairs_rows, pairs_path, ["sigma", "d_prime"])

    for sigma_val, d_prime_val in product(cfg.sigma_values, cfg.d_prime_values):
        run_one_pair(cfg, sigma_val, d_prime_val, points_rows, pairs_rows, flush)

    flush()
    return points_rows, pairs_rows


# =============================================================================
# Entry point
# =============================================================================
def main(cfg: ExperimentConfig) -> None:
    # Таблицы сохраняются внутри run_sigma_d_prime_grid после каждой точки,
    # поэтому отдельная запись в конце не нужна.
    run_sigma_d_prime_grid(cfg)