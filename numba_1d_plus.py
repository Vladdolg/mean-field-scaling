from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from numba import njit, types
from numba.experimental import jitclass


def _interp_uniform_impl(
    xdat: NDArray[np.float64],
    ydat: NDArray[np.float64],
    x: float,
    inv_dx: float,
) -> float:
    length = xdat.shape[0]
    if length == 1:
        return ydat[0]
    rel = x * inv_dx
    if rel <= 0.0:
        return ydat[0]
    upper = length - 1
    if rel >= upper:
        return ydat[upper]
    idx = int(rel)
    frac = rel - idx
    return ydat[idx] + (ydat[idx + 1] - ydat[idx]) * frac


def _distance_1d_impl(
    pos_a: float,
    pos_b: float,
    area_len: float,
    periodic_flag: np.uint8,
) -> float:
    diff = pos_a - pos_b
    if diff < 0.0:
        diff = -diff
    if periodic_flag == 1 and area_len > 0.0:
        wrap = area_len - diff
        if wrap < 0.0:
            wrap = -wrap
        if wrap < diff:
            diff = wrap
    return diff


def _sample_weighted_impl(values: NDArray[np.float64]) -> int:
    n = values.shape[0]
    total = 0.0
    for idx in range(n):
        total += values[idx]
    if total <= 0.0:
        return -1
    r = np.random.random() * total
    cumulative = 0.0
    for idx in range(n):
        cumulative += values[idx]
        if r <= cumulative:
            return idx
    return n - 1


_interp_uniform = njit(cache=True)(_interp_uniform_impl)
_distance_1d = njit(cache=True)(_distance_1d_impl)
_sample_weighted = njit(cache=True)(_sample_weighted_impl)


ssa_state_spec = [
    ("species_count", types.int32),
    ("area_length", types.float64),
    ("cell_count", types.int32),
    ("cell_length", types.float64),
    ("periodic_flag", types.uint8),
    ("cell_capacity", types.int32),
    ("max_slots", types.int64),
    ("seed", types.int64),
    ("b", types.Array(types.float64, 1, "C")),
    ("d", types.Array(types.float64, 1, "C")),
    ("dd", types.Array(types.float64, 2, "C")),
    ("cutoff", types.Array(types.float64, 2, "C")),
    ("cull", types.Array(types.int32, 2, "C")),
    ("birth_x", types.Array(types.float64, 2, "C")),
    ("birth_y", types.Array(types.float64, 2, "C")),
    ("birth_inv_dx", types.Array(types.float64, 1, "C")),
    ("death_x", types.Array(types.float64, 3, "C")),
    ("death_y", types.Array(types.float64, 3, "C")),
    ("death_inv_dx", types.Array(types.float64, 2, "C")),
    ("positions", types.Array(types.float64, 1, "C")),
    ("death_rates", types.Array(types.float64, 1, "C")),
    ("species", types.Array(types.int32, 1, "C")),
    ("cell_index", types.Array(types.int32, 1, "C")),
    ("slot_index", types.Array(types.int32, 1, "C")),
    ("cell_particles", types.Array(types.int64, 2, "C")),
    ("cell_counts", types.Array(types.int32, 1, "C")),
    ("cell_species_counts", types.Array(types.int32, 2, "C")),
    ("cell_birth_rate_by_species", types.Array(types.float64, 2, "C")),
    ("cell_death_rate_by_species", types.Array(types.float64, 2, "C")),
    ("cell_birth_rate", types.Array(types.float64, 1, "C")),
    ("cell_death_rate", types.Array(types.float64, 1, "C")),
    ("total_birth_rate", types.float64),
    ("total_death_rate", types.float64),
    ("population_total", types.int64),
    ("time", types.float64),
    ("event_count", types.int64),
    ("capacity_flag", types.uint8),
]


@jitclass(ssa_state_spec)
class SSAState1D:
    def __init__(
        self,
        species_count: np.int32,
        area_length: float,
        cell_count: np.int32,
        periodic_flag: np.uint8,
        cell_capacity: np.int32,
        seed_value: np.int64,
        b: NDArray[np.float64],
        d: NDArray[np.float64],
        dd: NDArray[np.float64],
        cutoff: NDArray[np.float64],
        cull: NDArray[np.int32],
        birth_x: NDArray[np.float64],
        birth_y: NDArray[np.float64],
        death_x: NDArray[np.float64],
        death_y: NDArray[np.float64],
    ):
        self.species_count = species_count
        self.area_length = area_length
        self.cell_count = cell_count
        self.cell_length = area_length / float(cell_count)
        self.periodic_flag = periodic_flag
        self.cell_capacity = cell_capacity
        self.seed = seed_value

        self.max_slots = np.int64(self.cell_count) * np.int64(self.cell_capacity)

        species_total = int(self.species_count)

        self.b = b
        self.d = d
        self.dd = dd
        self.cutoff = cutoff
        self.cull = cull
        self.birth_x = birth_x
        self.birth_y = birth_y
        self.death_x = death_x
        self.death_y = death_y

        self.birth_inv_dx = np.empty(species_total, dtype=np.float64)
        birth_cols = self.birth_x.shape[1]
        for s in range(species_total):
            if birth_cols > 1:
                dx = self.birth_x[s, 1] - self.birth_x[s, 0]
                if dx != 0.0:
                    self.birth_inv_dx[s] = 1.0 / dx
                else:
                    self.birth_inv_dx[s] = 0.0
            else:
                self.birth_inv_dx[s] = 0.0

        self.death_inv_dx = np.empty((species_total, species_total), dtype=np.float64)
        death_cols = self.death_x.shape[2]
        for s1 in range(species_total):
            for s2 in range(species_total):
                if death_cols > 1:
                    dx = self.death_x[s1, s2, 1] - self.death_x[s1, s2, 0]
                    if dx != 0.0:
                        self.death_inv_dx[s1, s2] = 1.0 / dx
                    else:
                        self.death_inv_dx[s1, s2] = 0.0
                else:
                    self.death_inv_dx[s1, s2] = 0.0

        slots = int(self.max_slots)
        capacity_i = int(self.cell_capacity)

        self.positions = np.zeros(slots, dtype=np.float64)
        self.death_rates = np.zeros(slots, dtype=np.float64)
        self.species = np.full(slots, -1, dtype=np.int32)
        self.cell_index = np.full(slots, -1, dtype=np.int32)
        self.slot_index = np.full(slots, -1, dtype=np.int32)

        self.cell_particles = np.full((self.cell_count, capacity_i), -1, dtype=np.int64)
        self.cell_counts = np.zeros(self.cell_count, dtype=np.int32)
        self.cell_species_counts = np.zeros((self.cell_count, species_total), dtype=np.int32)

        self.cell_birth_rate_by_species = np.zeros((self.cell_count, species_total), dtype=np.float64)
        self.cell_death_rate_by_species = np.zeros((self.cell_count, species_total), dtype=np.float64)
        self.cell_birth_rate = np.zeros(self.cell_count, dtype=np.float64)
        self.cell_death_rate = np.zeros(self.cell_count, dtype=np.float64)

        self.total_birth_rate = 0.0
        self.total_death_rate = 0.0
        self.population_total = np.int64(0)

        self.time = 0.0
        self.event_count = np.int64(0)
        self.capacity_flag = np.uint8(0)

        if seed_value >= 0:
            np.random.seed(int(seed_value))

    def _sample_weighted_local(self, values: NDArray[np.float64]) -> int:
        return _sample_weighted(values)

    def _distance(self, pos_a: float, pos_b: float) -> float:
        return _distance_1d(pos_a, pos_b, self.area_length, self.periodic_flag)

    def _interp(
        self,
        xdat: NDArray[np.float64],
        ydat: NDArray[np.float64],
        x: float,
        inv_dx: float,
    ) -> float:
        return _interp_uniform(xdat, ydat, x, inv_dx)

    def _neighbor_window(self, cell_idx: int, max_cull: int) -> tuple[int, int]:
        if max_cull < 0:
            max_cull = 0
        if self.periodic_flag == 1:
            span = 2 * max_cull + 1
            if span > self.cell_count:
                span = self.cell_count
            start = cell_idx - max_cull
            return start, span
        left = cell_idx - max_cull
        if left < 0:
            left = 0
        right = cell_idx + max_cull
        if right >= self.cell_count:
            right = self.cell_count - 1
        span = right - left + 1
        return left, span

    def _sample_parent(self, cell_idx: int, species_id: int) -> int:
        total = self.cell_species_counts[cell_idx, species_id]
        if total <= 0:
            return -1
        target = int(np.random.random() * total)
        count = self.cell_counts[cell_idx]
        for slot in range(count):
            pid = self.cell_particles[cell_idx, slot]
            if pid == -1:
                continue
            if self.species[pid] == species_id:
                if target == 0:
                    return pid
                target -= 1
        return -1

    def _sample_victim(self, cell_idx: int, species_id: int) -> int:
        total = 0.0
        count = self.cell_counts[cell_idx]
        for slot in range(count):
            pid = self.cell_particles[cell_idx, slot]
            if pid == -1:
                continue
            if self.species[pid] == species_id:
                total += self.death_rates[pid]
        if total <= 0.0:
            return -1
        r = np.random.random() * total
        for slot in range(count):
            pid = self.cell_particles[cell_idx, slot]
            if pid == -1:
                continue
            if self.species[pid] == species_id:
                r -= self.death_rates[pid]
                if r <= 0.0:
                    return pid
        return -1

    def spawn_particle(self, species_id: int, position: float) -> bool:
        if self.population_total >= self.max_slots:
            self.capacity_flag = np.uint8(1)
            return False

        pos = position
        if pos < 0.0 or pos > self.area_length:
            if self.periodic_flag == 1 and self.area_length > 0.0:
                pos = pos - self.area_length * math.floor(pos / self.area_length)
                if pos < 0.0:
                    pos += self.area_length
            else:
                return False

        cell_idx = int(math.floor(pos * self.cell_count / self.area_length))
        if cell_idx < 0:
            cell_idx = 0
        elif cell_idx >= self.cell_count:
            cell_idx = self.cell_count - 1

        cell_slot = self.cell_counts[cell_idx]
        capacity_i = int(self.cell_capacity)
        if cell_slot >= capacity_i:
            self.capacity_flag = np.uint8(1)
            return False

        idx = int(self.population_total)
        self.population_total = np.int64(idx + 1)

        self.positions[idx] = pos
        self.species[idx] = species_id
        self.cell_index[idx] = cell_idx
        self.slot_index[idx] = cell_slot
        self.cell_particles[cell_idx, cell_slot] = idx
        self.cell_counts[cell_idx] = cell_slot + 1
        self.cell_species_counts[cell_idx, species_id] += 1

        base_birth = self.b[species_id]
        base_death = self.d[species_id]

        self.total_birth_rate += base_birth
        self.total_death_rate += base_death
        self.cell_birth_rate_by_species[cell_idx, species_id] += base_birth
        self.cell_birth_rate[cell_idx] += base_birth
        self.cell_death_rate_by_species[cell_idx, species_id] += base_death
        self.cell_death_rate[cell_idx] += base_death
        self.death_rates[idx] = base_death

        added_death = 0.0
        spec_total = int(self.species_count)
        periodic = self.periodic_flag == 1
        for other_species in range(spec_total):
            cutoff_ij = self.cutoff[species_id, other_species]
            strength_ij = self.dd[species_id, other_species]
            cutoff_ji = self.cutoff[other_species, species_id]
            strength_ji = self.dd[other_species, species_id]

            interacts_ij = strength_ij != 0.0 and cutoff_ij > 0.0
            interacts_ji = strength_ji != 0.0 and cutoff_ji > 0.0
            if not interacts_ij and not interacts_ji:
                continue

            cull_ij = int(self.cull[species_id, other_species])
            cull_ji = int(self.cull[other_species, species_id])
            if cull_ij >= cull_ji:
                max_cull = cull_ij
            else:
                max_cull = cull_ji

            start, span = self._neighbor_window(cell_idx, max_cull)
            for offset in range(span):
                neighbor_cell = start + offset
                if periodic:
                    neighbor_cell = neighbor_cell % self.cell_count
                count = self.cell_counts[neighbor_cell]
                for slot_n in range(count):
                    other_idx = self.cell_particles[neighbor_cell, slot_n]
                    if other_idx == -1 or other_idx == idx:
                        continue
                    dist = self._distance(self.positions[other_idx], pos)
                    if interacts_ij and dist <= cutoff_ij:
                        kernel = self._interp(
                            self.death_x[species_id, other_species],
                            self.death_y[species_id, other_species],
                            dist,
                            self.death_inv_dx[species_id, other_species],
                        )
                        delta = strength_ij * kernel
                        if delta != 0.0:
                            self.death_rates[other_idx] += delta
                            self.cell_death_rate_by_species[neighbor_cell, other_species] += delta
                            self.cell_death_rate[neighbor_cell] += delta
                            self.total_death_rate += delta
                    if interacts_ji and dist <= cutoff_ji:
                        kernel = self._interp(
                            self.death_x[other_species, species_id],
                            self.death_y[other_species, species_id],
                            dist,
                            self.death_inv_dx[other_species, species_id],
                        )
                        added_death += strength_ji * kernel

        if added_death != 0.0:
            self.death_rates[idx] += added_death
            self.cell_death_rate_by_species[cell_idx, species_id] += added_death
            self.cell_death_rate[cell_idx] += added_death
            self.total_death_rate += added_death

        return True

    def kill_particle_index(self, idx: int) -> bool:
        if idx < 0 or idx >= self.population_total:
            return False

        species_id = self.species[idx]
        cell_idx = self.cell_index[idx]
        slot = self.slot_index[idx]
        pos = self.positions[idx]

        base_birth = self.b[species_id]
        particle_death = self.death_rates[idx]

        self.population_total = np.int64(self.population_total - 1)

        self.total_birth_rate -= base_birth
        self.total_death_rate -= particle_death
        self.cell_birth_rate_by_species[cell_idx, species_id] -= base_birth
        self.cell_birth_rate[cell_idx] -= base_birth
        self.cell_death_rate_by_species[cell_idx, species_id] -= particle_death
        self.cell_death_rate[cell_idx] -= particle_death
        self.cell_species_counts[cell_idx, species_id] -= 1

        spec_total = int(self.species_count)
        periodic = self.periodic_flag == 1
        for other_species in range(spec_total):
            cutoff_val = self.cutoff[species_id, other_species]
            strength = self.dd[species_id, other_species]
            if strength == 0.0 or cutoff_val <= 0.0:
                continue
            cull_range = int(self.cull[species_id, other_species])

            start, span = self._neighbor_window(cell_idx, cull_range)
            for offset in range(span):
                neighbor_cell = start + offset
                if periodic:
                    neighbor_cell = neighbor_cell % self.cell_count
                count = self.cell_counts[neighbor_cell]
                for slot_n in range(count):
                    other_idx = self.cell_particles[neighbor_cell, slot_n]
                    if other_idx == -1 or other_idx == idx:
                        continue
                    dist = self._distance(self.positions[other_idx], pos)
                    if dist <= cutoff_val:
                        kernel = self._interp(
                            self.death_x[species_id, other_species],
                            self.death_y[species_id, other_species],
                            dist,
                            self.death_inv_dx[species_id, other_species],
                        )
                        delta = strength * kernel
                        if delta != 0.0:
                            self.death_rates[other_idx] -= delta
                            self.cell_death_rate_by_species[neighbor_cell, other_species] -= delta
                            self.cell_death_rate[neighbor_cell] -= delta
                            self.total_death_rate -= delta

        cell_last_slot = self.cell_counts[cell_idx] - 1
        last_idx_in_cell = self.cell_particles[cell_idx, cell_last_slot]
        self.cell_particles[cell_idx, cell_last_slot] = -1
        self.cell_counts[cell_idx] = cell_last_slot
        if slot != cell_last_slot and last_idx_in_cell != -1:
            self.cell_particles[cell_idx, slot] = last_idx_in_cell
            self.slot_index[last_idx_in_cell] = slot

        last_idx = int(self.population_total)
        if idx != last_idx:
            last_cell = self.cell_index[last_idx]
            last_slot = self.slot_index[last_idx]

            self.positions[idx] = self.positions[last_idx]
            self.death_rates[idx] = self.death_rates[last_idx]
            self.species[idx] = self.species[last_idx]
            self.cell_index[idx] = last_cell
            self.slot_index[idx] = last_slot

            self.cell_particles[last_cell, last_slot] = idx

        self.positions[last_idx] = 0.0
        self.death_rates[last_idx] = 0.0
        self.species[last_idx] = -1
        self.cell_index[last_idx] = -1
        self.slot_index[last_idx] = -1

        if self.total_birth_rate < 0.0 and self.total_birth_rate > -1e-9:
            self.total_birth_rate = 0.0
        if self.total_death_rate < 0.0 and self.total_death_rate > -1e-9:
            self.total_death_rate = 0.0

        return True

    def attempt_birth_event(self) -> bool:
        cell_idx = self._sample_weighted_local(self.cell_birth_rate)
        if cell_idx < 0:
            return False

        species_id = self._sample_weighted_local(self.cell_birth_rate_by_species[cell_idx])
        if species_id < 0:
            return False

        parent_idx = self._sample_parent(cell_idx, species_id)
        if parent_idx < 0:
            return False

        parent_pos = self.positions[parent_idx]
        radius = self._interp(
            self.birth_x[species_id],
            self.birth_y[species_id],
            np.random.random(),
            self.birth_inv_dx[species_id],
        )
        direction = -1.0 if np.random.random() < 0.5 else 1.0
        child_pos = parent_pos + direction * radius
        if self.periodic_flag == 1 and self.area_length > 0.0:
            child_pos = child_pos - self.area_length * math.floor(child_pos / self.area_length)
            if child_pos < 0.0:
                child_pos += self.area_length

        return self.spawn_particle(species_id, child_pos)

    def attempt_death_event(self) -> bool:
        cell_idx = self._sample_weighted_local(self.cell_death_rate)
        if cell_idx < 0:
            return False

        species_id = self._sample_weighted_local(self.cell_death_rate_by_species[cell_idx])
        if species_id < 0:
            return False

        victim_idx = self._sample_victim(cell_idx, species_id)
        if victim_idx < 0:
            return False

        return self.kill_particle_index(victim_idx)

    def spawn_random(self) -> bool:
        return self.attempt_birth_event()

    def kill_random(self) -> bool:
        return self.attempt_death_event()

    def run_events(self, max_events: int) -> int:
        if max_events <= 0:
            return 0
        performed = 0
        for _ in range(max_events):
            total_rate = self.total_birth_rate + self.total_death_rate
            if total_rate <= 1e-12:
                break

            dt = -math.log(np.random.random()) / total_rate
            self.time += dt

            if np.random.random() * total_rate < self.total_birth_rate:
                if self.attempt_birth_event():
                    performed += 1
                    self.event_count += 1
                elif self.capacity_flag == 1:
                    break
            else:
                if self.attempt_death_event():
                    performed += 1
                    self.event_count += 1
        return performed

    def run_until_time(self, duration: float) -> int:
        if duration <= 0.0:
            return 0
        target_time = self.time + duration
        performed = 0
        while self.time < target_time:
            total_rate = self.total_birth_rate + self.total_death_rate
            if total_rate <= 1e-12:
                break
            dt = -math.log(np.random.random()) / total_rate
            if self.time + dt > target_time:
                self.time = target_time
                break
            self.time += dt
            if np.random.random() * total_rate < self.total_birth_rate:
                if self.attempt_birth_event():
                    performed += 1
                    self.event_count += 1
                elif self.capacity_flag == 1:
                    break
            else:
                if self.attempt_death_event():
                    performed += 1
                    self.event_count += 1
        return performed

    def make_event(self) -> bool:
        return self.run_events(1) > 0

    def reached_capacity(self) -> bool:
        return self.capacity_flag == 1

    def current_population(self) -> int:
        return int(self.population_total)

    def current_time(self) -> float:
        return self.time

    def current_event_count(self) -> int:
        return int(self.event_count)

    def reseed(self, seed: int) -> None:
        np.random.seed(int(seed))
        self.seed = np.int64(seed)


def make_ssa_state_1d(
    M: int,
    area_len: float,
    birth_rates: Sequence[float] | NDArray[np.float64],
    death_rates: Sequence[float] | NDArray[np.float64],
    dd_matrix: Sequence[Sequence[float]] | NDArray[np.float64],
    birth_x: NDArray[np.float64],
    birth_y: NDArray[np.float64],
    death_x: NDArray[np.float64],
    death_y: NDArray[np.float64],
    cutoffs: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    cell_count: int = 100,
    cell_capacity: int | None = None,
    is_periodic: bool = False,
    seed: int | None = None,
    initial_population: Sequence[Sequence[float]] | None = None,
) -> SSAState1D:
    species_count = int(M)
    area_len_f = float(area_len)
    cell_count_i = int(cell_count)

    b_arr = np.ascontiguousarray(birth_rates, dtype=np.float64)
    d_arr = np.ascontiguousarray(death_rates, dtype=np.float64)
    dd_matrix_arr = np.ascontiguousarray(dd_matrix, dtype=np.float64)
    cutoffs_arr = np.ascontiguousarray(cutoffs, dtype=np.float64)

    birth_x_table = np.ascontiguousarray(birth_x, dtype=np.float64)
    birth_y_table = np.ascontiguousarray(birth_y, dtype=np.float64)
    death_x_table = np.ascontiguousarray(death_x, dtype=np.float64)
    death_y_table = np.ascontiguousarray(death_y, dtype=np.float64)

    cell_length = area_len_f / float(cell_count_i)
    cull = np.zeros((species_count, species_count), dtype=np.int32)
    for s1 in range(species_count):
        for s2 in range(species_count):
            cutoff_val = cutoffs_arr[s1, s2]
            if cutoff_val <= 0.0:
                cull[s1, s2] = 0
            else:
                cull[s1, s2] = int(math.ceil(cutoff_val / cell_length))

    if cell_capacity is None:
        estimated = 0
        if initial_population is not None:
            for coords in initial_population:
                estimated += np.asarray(coords, dtype=np.float64).size
        if estimated == 0:
            estimated = species_count * 8 * max(1, cell_count_i // 10 + 1)
        per_cell = estimated // max(1, cell_count_i) + 16
        derived_capacity = max(32, per_cell * 2)
    else:
        derived_capacity = max(1, int(cell_capacity))

    if derived_capacity <= 0:
        raise ValueError("cell_capacity must be positive")

    seed_val = np.int64(seed if seed is not None else -1)

    state = SSAState1D(
        np.int32(species_count),
        area_len_f,
        np.int32(cell_count_i),
        np.uint8(1 if is_periodic else 0),
        np.int32(derived_capacity),
        seed_val,
        b_arr,
        d_arr,
        dd_matrix_arr,
        cutoffs_arr,
        cull,
        birth_x_table,
        birth_y_table,
        death_x_table,
        death_y_table,
    )

    if initial_population is not None:
        for species_id, coords in enumerate(initial_population):
            arr = np.asarray(coords, dtype=np.float64).reshape(-1)
            for pos in arr:
                if not state.spawn_particle(species_id, float(pos)):
                    raise RuntimeError("initial population exceeds configured capacity")

    return state





def get_all_particle_coords(state: SSAState1D) -> list[NDArray[np.float64]]:
    species_total = int(state.species_count)
    coords: list[NDArray[np.float64]] = []
    total = int(state.population_total)
    for species_id in range(species_total):
        collected = []
        for idx in range(total):
            if state.species[idx] == species_id:
                collected.append(state.positions[idx])
        coords.append(np.asarray(collected, dtype=np.float64))
    return coords


def get_all_particle_death_rates(state: SSAState1D) -> list[NDArray[np.float64]]:
    species_total = int(state.species_count)
    result: list[NDArray[np.float64]] = []
    total = int(state.population_total)
    for species_id in range(species_total):
        collected = []
        for idx in range(total):
            if state.species[idx] == species_id:
                collected.append(state.death_rates[idx])
        result.append(np.asarray(collected, dtype=np.float64))
    return result


__all__ = [
    "SSAState1D",
    "make_ssa_state_1d",
    "get_all_particle_coords",
    "get_all_particle_death_rates",
]
