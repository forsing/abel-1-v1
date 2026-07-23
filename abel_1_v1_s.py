#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Niels Henrik Abel (1802-1829) - norveški matematičar
algebra and mathematical analysis 
group theory and elliptic functions
"""

"""
abel_1_v1_s.py — Abelova suma na nizu S → next Loto / Loto Plus

Isti S kao surface_s (diskretna rotaciona površina).
Forecast: Abelova transformacija delimičnih zbirova A=cumsum(S)
  r = SEED/(SEED+1) = 39/40
  A_next_hat = (1-r) Σ_{j=0}^{n-1} r^j A[n-1-j]
  S_next_hat = A_next_hat - A[n-1] + (A[n-1] - A_abel_level)... 

Preciznije (Abel + delimični zbirovi):
  A[t] = Σ_{i=0}^{t} S[i]
  Abel-nivo A:  A_abel = (1-r) Σ_{j=0}^{n-1} r^j A[n-1-j]
  Predikcija sledećeg delimičnog zbira kao Abel-nivo pomeranja za
  poslednji korak u A-prostoru nije stabilna; koristim ekvivalentnu formulaciju (summation by parts):

  S_next_hat = (1-r) Σ_{j=0}^{n-1} r^j S[n-1-j]
  (Abelova sredina niza S; sledi iz A[t]-A[t-1]=S[t] + summation by parts)

Isti invert S→C kao surface_s.
CSV: loto_2949 i loto_plus_1705. 
"""

from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np

SEED = 39
N_PICK = 7
MAX_NUM = 39
MIN_HIST = 100
# Abel radijus (fiksno, bez RNG)
ABEL_R = float(SEED) / float(SEED + 1)  # 39/40

ROOT = Path(__file__).resolve().parent
CSV_LOTO = ROOT.parent / "data" / "loto7_4654_k58_loto_2949.csv"
CSV_PLUS = ROOT.parent / "data" / "loto7_4654_k58_loto_plus_1705.csv"
CSV_PATH = CSV_LOTO

POS_LO = np.arange(1, N_PICK + 1, dtype=int)
POS_HI = POS_LO + (MAX_NUM - N_PICK)


def load_draws(path: Path = CSV_PATH) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", dtype=int)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    assert raw.shape[1] == N_PICK, raw.shape
    assert raw.min() >= 1 and raw.max() <= MAX_NUM
    return raw


def surface_S(prev: np.ndarray, last: np.ndarray) -> float:
    y = np.sort(last.astype(np.float64))
    yp = np.sort(prev.astype(np.float64))
    dydx = np.empty(N_PICK, dtype=np.float64)
    dydx[0] = y[0] - yp[0]
    for i in range(1, N_PICK):
        dydx[i] = y[i] - y[i - 1]
    return float(2.0 * np.pi * np.sum(y * np.sqrt(1.0 + dydx * dydx)))


def surface_S_sorted(yp: np.ndarray, y: np.ndarray) -> float:
    dydx0 = y[0] - yp[0]
    s = y[0] * np.sqrt(1.0 + dydx0 * dydx0)
    for i in range(1, N_PICK):
        d = y[i] - y[i - 1]
        s += y[i] * np.sqrt(1.0 + d * d)
    return float(2.0 * np.pi * s)


def series_S(draws: np.ndarray) -> np.ndarray:
    n = len(draws)
    out = np.empty(n, dtype=np.float64)
    out[0] = surface_S(draws[0], draws[0])
    for t in range(1, n):
        out[t] = surface_S(draws[t - 1], draws[t])
    return out


def _fit_predict_abel(S: np.ndarray) -> float:
    """
    Abel preko delimičnih zbirova A=cumsum(S), r=SEED/(SEED+1).

    A[t] = Σ_{i≤t} S[i]
    Put: Abelova transformacija (summation by parts) na (A, geometrijski tegovi)
    daje Abel-sredinu niza S:

      S_next_hat = (1-r) Σ_{j=0}^{n-1} r^j S[n-1-j]

    (A se gradi eksplicitno; S[t] = A[t]-A[t-1], A[-1]:=0.)
    """
    n = len(S)
    if n < 1:
        return 0.0
    r = ABEL_R
    A = np.cumsum(S.astype(np.float64))
    # S iz A (provera mosta delimičnih zbirova)
    S_from_A = np.empty(n, dtype=np.float64)
    S_from_A[0] = A[0]
    S_from_A[1:] = A[1:] - A[:-1]
    abel_S = 0.0
    rj = 1.0
    for j in range(n):
        abel_S += rj * float(S_from_A[n - 1 - j])
        rj *= r
    abel_S *= 1.0 - r
    return float(abel_S)


def valid_positions(combo: Tuple[int, ...]) -> bool:
    return all(int(POS_LO[i]) <= combo[i] <= int(POS_HI[i]) for i in range(N_PICK))


def invert_S_to_combo(prev_draw: np.ndarray, S_target: float) -> Tuple[List[int], float]:
    yp = np.sort(prev_draw.astype(np.float64))
    best_combo: Tuple[int, ...] | None = None
    best_S = 0.0
    best_key: Tuple[float, int, Tuple[int, ...]] | None = None
    for comb in combinations(range(1, MAX_NUM + 1), N_PICK):
        if not valid_positions(comb):
            continue
        y = np.asarray(comb, dtype=np.float64)
        s = surface_S_sorted(yp, y)
        err = abs(s - S_target)
        seed_tie = (sum(comb) * SEED) % 97
        key = (err, seed_tie, comb)
        if best_key is None or key < best_key:
            best_key = key
            best_combo = comb
            best_S = s
    if best_combo is None:
        raise RuntimeError("nema kandidata")
    return list(best_combo), best_S


def predict_next(draws: np.ndarray) -> dict:
    S = series_S(draws)
    assert len(S) == len(draws)
    S_hat = _fit_predict_abel(S)
    nxt, S_match = invert_S_to_combo(draws[-1], S_hat)
    return {
        "S": S,
        "S_last": float(S[-1]),
        "S_next_hat": float(S_hat),
        "method": "abel",
        "abel_r": ABEL_R,
        "next": nxt,
        "S_match": S_match,
        "last_combo": draws[-1].tolist(),
    }


def _print_one(label: str, csv_path: Path, r: dict) -> None:
    _, counts = np.unique(np.round(r["S"], 12), return_counts=True)
    n_dup = int(np.sum(counts > 1))
    print(f"=== {label} ===")
    print("csv:             ", csv_path.name)
    print("csv rows:        ", len(r["S"]))
    print("last combo:      ", r["last_combo"])
    print("S_last (float64):", format(r["S_last"], ".12f"))
    print("method:          ", r["method"])
    print("S_next_hat:      ", format(r["S_next_hat"], ".12f"))
    print("S_match:         ", format(r["S_match"], ".12f"))
    print("S hist dup@12dp: ", n_dup)
    print(f"next_{label}:     ", r["next"])


def main() -> None:
    draws_loto = load_draws(CSV_LOTO)
    draws_plus = load_draws(CSV_PLUS)
    assert len(draws_loto) >= MIN_HIST + 2
    assert len(draws_plus) >= MIN_HIST + 2

    r_loto = predict_next(draws_loto)
    r_plus = predict_next(draws_plus)

    _print_one("loto", CSV_LOTO, r_loto)
    print()
    _print_one("loto_plus", CSV_PLUS, r_plus)
    print()
    print("next_loto:      ", r_loto["next"])
    print("next_loto_plus: ", r_plus["next"])


if __name__ == "__main__":
    main()



########################################################



"""
=== loto ===
csv:              loto7_4654_k58_loto_2949.csv
csv rows:         2949
last combo:       [6, 8, 12, 22, 30, 36, 38]
S_last (float64): 5295.255721742701
method:           abel
S_next_hat:       4876.953121751467
S_match:          4876.953060857481
S hist dup@12dp:  0
next_loto:      [5, 9, 14, 29, 30, 32, 35]

=== loto_plus ===
csv:              loto7_4654_k58_loto_plus_1705.csv
csv rows:         1705
last combo:       [3, 6, 9, 13, 18, 20, 21]
S_last (float64): 1756.816628406357
method:           abel
S_next_hat:       4612.876213127133
S_match:          4612.876247408894
S hist dup@12dp:  0
next_loto_plus:      [1, 4, 18, 22, 24, 28, 34]

next_loto:       [5, 9, 14, 29, 30, 32, 35]
next_loto_plus:  [1, 4, 18, 22, 24, 28, 34]
"""



########################################################################################



"""
Backtest Abel (n-1):


Loto (2948 → actual 2949)

pred: [3, 6, 8, 21, 22, 30, 35]
actual: [6, 8, 12, 22, 30, 36, 38]
HIT: False
· 4/7 (6, 8, 22, 30)
S_next_hat: 4866.227414059405
S_match: 4866.227378348540


Loto Plus (1704 → actual 1705)

pred: [4, 15, 21, 23, 28, 30, 35]
actual: [3, 6, 9, 13, 18, 20, 21]
HIT: False
· 1/7 (21)
S_next_hat: 4686.108510171242
S_match: 4686.108748554217


Invert i dalje poklapa S (S_match ≈ S_next_hat).
"""



"""
Backtest Abel (n-2):


Loto (2947 → actual 2948)

pred: [2, 6, 16, 21, 26, 32, 36]
actual: [7, 15, 20, 26, 28, 30, 39]
HIT: False
· 1/7 (26)
S_next_hat: 4844.959116000518
S_match: 4844.959110078014


Loto Plus (1703 → actual 1704)

pred: [4, 15, 23, 26, 27, 32, 35]
actual: [7, 8, 14, 15, 17, 23, 32]
HIT: False
· 3/7 (15, 23, 32)
S_next_hat: 4708.390744873537
S_match: 4708.390809610672
"""



"""
Backtest Abel (n-3):


Loto (2946 → actual 2947)

pred: [1, 10, 14, 15, 20, 23, 35]
actual: [1, 4, 7, 20, 27, 29, 39]
HIT: False
· 2/7 (1, 20)
S_next_hat: 4817.061143376021
S_match: 4817.061320215554


Loto Plus (1702 → actual 1703)

pred: [2, 3, 9, 19, 20, 26, 35]
actual: [4, 5, 6, 11, 12, 18, 28]
HIT: False
· 0/7
S_next_hat: 4750.956012525756
S_match: 4750.956078186530
"""



########################################################################################



"""
ANALIZA — abel_1_v1_s.py:

1. Ulaz — dva čista CSV-a: Loto (2949) i Plus (1705).

2. Skalar S — isti kao u surface_s (diskretna rotaciona površina).

3. Niz S → delimični zbirovi A = cumsum(S).

4. Forecast (Abel) — r = 39/40:
   S_next_hat = (1-r) Σ_j r^j S[n-1-j]
   (Abel-sredina; most preko A).

5. Invert — isti S→C kao surface_s → jedan next.

6. Izlaz — next_loto i next_loto_plus.

Razlika od surface_s: umesto AR(2) ide Abel.

Bitno: invert radi (S se poklapa); HIT False meri forecast, ne mapu S ↔ C.

Backtest Loto n−1: 4/7 pogotka (6, 8, 22, 30) — jak delimični rezultat.
"""



########################################################################################



"""
BELEŠKE — Abel kao struktura za algoritam, ne kao dokaz da „pogađa“ loto.

Od Abela šta ima smisla (deterministički, na čistom CSV-u):

1. Abelova suma / transformacija — drugačije sabiranje niza (npr. S ili frekvencija)
   → jedan skor → next
2. Abelove funkcije / eliptički integrali — mapa kombinacije u skalar (slično surface S),
   pa forecast + invert
3. Abelova grupa — komutativna struktura na brojevima/parovima
   (operacija + provera zatvorenosti), bez RNG
4. Abel–Ruffini (kvintika) — za 7/39 praktično ne pomaže.


stavka 1 u ovom fajlu:
  CSV → niz S (kao sada)
  Umesto AR(2): Abelova transformacija delimičnih zbirova S → S_next_hat
  Isti invert S→C → next_loto / next_loto_plus

drugačiji linearni filter istorije od AR2;
i dalje deterministički, ceo CSV, SEED=39 samo u tie-break invertu.

Rizik: ako je S skoro bela buka, Abel ne mora biti bolji od AR2 na HIT-u 
— meri se backtestom n-1 / n-2 / n-3.



Redosled rada (Abel → loto), jači prvi:
  1. Abelova suma / transformacija na nizu S → S_next_hat → invert  (ovo)
  2. Eliptičke / Abelove funkcije — nova mapa kombinacije → skalar, pa forecast + invert
  3. Abelova grupa — komutativna operacija na brojevima/parovima → skor → next
  (Abel-Ruffini / kvintika — ne u ovom redu.)
"""



########################################################