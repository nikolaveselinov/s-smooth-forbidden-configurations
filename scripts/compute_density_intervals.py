#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array


def smooth_numbers(primes: tuple[int, ...], k: int) -> list[int]:
    seen = {1}
    heap = [1]
    arr: list[int] = []
    while len(arr) < k:
        x = heapq.heappop(heap)
        arr.append(x)
        for p in primes:
            y = x * p
            if y not in seen:
                seen.add(y)
                heapq.heappush(heap, y)
    return arr


def build_edges(primes: tuple[int, ...], arr: list[int]) -> list[tuple[int, ...]]:
    idx = {x: i for i, x in enumerate(arr)}
    edges: list[tuple[int, ...]] = []
    for x in arr:
        e = [idx[x]]
        ok = True
        for p in primes:
            y = x * p
            if y not in idx:
                ok = False
                break
            e.append(idx[y])
        if ok:
            edges.append(tuple(e))
    return edges


def prefix_optima(primes: tuple[int, ...], kmax: int, verbose: bool = False) -> tuple[list[int], list[int]]:
    arr = smooth_numbers(primes, kmax)
    edges = build_edges(primes, arr)
    vals: list[int] = []
    for k in range(1, kmax + 1):
        active = [e for e in edges if e[-1] < k]
        m = len(active)
        c = -np.ones(k, dtype=float)
        constraints = []
        if m:
            rows: list[int] = []
            cols: list[int] = []
            data: list[int] = []
            for r, e in enumerate(active):
                for col in e:
                    rows.append(r)
                    cols.append(col)
                    data.append(1)
            A = csc_array((data, (rows, cols)), shape=(m, k), dtype=float)
            constraints = [LinearConstraint(A, -np.inf * np.ones(m), len(primes) * np.ones(m))]
        bounds = Bounds(np.zeros(k), np.ones(k))
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(k))
        if res.status != 0:
            raise RuntimeError(f"MILP failed for primes={primes}, k={k}, status={res.status}, message={res.message}")
        vals.append(int(round(-res.fun)))
        if verbose and (k <= 10 or k % 20 == 0 or k == kmax):
            print(f"primes={primes} k={k} f(k)={vals[-1]}")
    return arr, vals


def floor_log_p_fraction(x: Fraction, p: int) -> int:
    assert x > 0
    power = 1
    a = 0
    while power * p <= x:
        power *= p
        a += 1
    return a


@lru_cache(maxsize=None)
def total_reciprocal_mass(primes: tuple[int, ...]) -> Fraction:
    out = Fraction(1, 1)
    for p in primes:
        out *= Fraction(p, p - 1)
    return out


@lru_cache(maxsize=None)
def tail_sum_fraction(primes: tuple[int, ...], num: int, den: int) -> Fraction:
    x = Fraction(num, den)
    if len(primes) == 1:
        p = primes[0]
        a = floor_log_p_fraction(x, p)
        return Fraction(1, (p - 1) * (p ** a))

    p = primes[0]
    rest = primes[1:]
    amax = floor_log_p_fraction(x, p)
    total = Fraction(0, 1)
    for a in range(amax + 1):
        sub_x = x / (p ** a)
        total += Fraction(1, p ** a) * tail_sum_fraction(rest, sub_x.numerator, sub_x.denominator)
    total += Fraction(1, (p - 1) * (p ** amax)) * total_reciprocal_mass(rest)
    return total


def decimal_of_fraction(frac: Fraction, digits: int = 30) -> str:
    getcontext().prec = digits + 10
    dec = Decimal(frac.numerator) / Decimal(frac.denominator)
    return format(dec, f".{digits}f")


@dataclass
class Result:
    primes: tuple[int, ...]
    kmax: int
    f_last: int
    smooth_last: int
    lower: str
    upper: str
    width: str
    smooth_prefix_density: str
    full_density_constant_prefactor: str
    deltas: list[int]
    smooth_numbers: list[int]
    optima: list[int]


def compute_interval(primes: tuple[int, ...], kmax: int, digits: int = 30, verbose: bool = False) -> Result:
    arr, vals = prefix_optima(primes, kmax, verbose=verbose)
    deltas: list[int] = []
    prev = 0
    lower_sum = Fraction(0, 1)
    for d, fk in zip(arr, vals):
        delta = fk - prev
        deltas.append(delta)
        lower_sum += Fraction(delta, d)
        prev = fk

    c_s = Fraction(1, 1)
    for p in primes:
        c_s *= Fraction(p - 1, p)

    lower = c_s * lower_sum
    tail = tail_sum_fraction(primes, arr[-1], 1)
    upper = lower + c_s * tail
    width = upper - lower

    return Result(
        primes=primes,
        kmax=kmax,
        f_last=vals[-1],
        smooth_last=arr[-1],
        lower=decimal_of_fraction(lower, digits),
        upper=decimal_of_fraction(upper, digits),
        width=decimal_of_fraction(width, digits),
        smooth_prefix_density=decimal_of_fraction(Fraction(vals[-1], kmax), digits),
        full_density_constant_prefactor=decimal_of_fraction(c_s, digits),
        deltas=deltas,
        smooth_numbers=arr,
        optima=vals,
    )


def parse_specs(raw_specs: Iterable[str]) -> list[tuple[tuple[int, ...], int]]:
    out = []
    for spec in raw_specs:
        left, right = spec.split(":")
        primes = tuple(sorted(int(x) for x in left.split(",") if x))
        out.append((primes, int(right)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute exact prefix optima and alpha-intervals for S-smooth corner-free problems.")
    parser.add_argument("--spec", action="append", required=True, help="Format: p1,p2,...:K, e.g. 2,3:200")
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = []
    for primes, kmax in parse_specs(args.spec):
        results.append(asdict(compute_interval(primes, kmax, digits=args.digits, verbose=args.verbose)))

    args.out.write_text(json.dumps({"results": results}, indent=2))
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
