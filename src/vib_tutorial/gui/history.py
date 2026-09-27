"""Growable buffer of recent simulation samples for the strip charts."""

from __future__ import annotations

import numpy as np

# Columns of History.e: stored energy (with the parameters of the time) and the ledger.
ENERGY_COLUMNS = ("kinetic", "potential", "added", "work", "dissipated")


class History:
    def __init__(self, n: int, capacity: int = 300_000) -> None:
        self.capacity = capacity
        self.reset(n)

    def reset(self, n: int) -> None:
        self.n = n
        self.t = np.empty(self.capacity)
        self.x = np.empty((self.capacity, n))
        self.v = np.empty((self.capacity, n))
        self.f = np.empty(self.capacity)
        self.e = np.empty((self.capacity, len(ENERGY_COLUMNS)))
        self.size = 0

    def extend(self, ts: np.ndarray, xs: np.ndarray, vs: np.ndarray, fs: np.ndarray, es: np.ndarray) -> None:
        k = ts.size
        if k == 0:
            return
        if k >= self.capacity:
            ts, xs, vs, fs, es = (a[-self.capacity :] for a in (ts, xs, vs, fs, es))
            k = ts.size
        arrays = (self.t, self.x, self.v, self.f, self.e)
        if self.size + k > self.capacity:
            # Drop the oldest half (or more) to make room; amortized O(1).
            keep = min(self.size, self.capacity // 2, self.capacity - k)
            start = self.size - keep
            for a in arrays:
                a[:keep] = a[start : self.size]
            self.size = keep
        s = slice(self.size, self.size + k)
        for a, new in zip(arrays, (ts, xs, vs, fs, es)):
            a[s] = new
        self.size += k

    def window(self, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Views (t, x, v, f, e) of the samples within the last `seconds` of simulated time."""
        i0 = 0
        if self.size:
            i0 = int(np.searchsorted(self.t[: self.size], self.t[self.size - 1] - seconds))
        s = slice(i0, self.size)
        return self.t[s], self.x[s], self.v[s], self.f[s], self.e[s]
