"""Growable buffer of recent simulation samples for the strip charts."""

from __future__ import annotations

import numpy as np

# Columns of History.e: stored energy (with the parameters of the time) and the ledger.
ENERGY_COLUMNS = ("kinetic", "potential", "added", "work", "dissipated", "friction")
# History.s holds the element forces (with the parameters of the time): n springs, then n dampers.
# History.g holds the ground displacement and velocity (0 unless the input is a ground motion).
# History.offset numbers the samples: t[i] is sample offset + i of every sample ever stored.
# It never restarts, not even on reset, so caches keyed by it (the plot's Decimator) see new data.


class History:
    def __init__(self, n: int, capacity: int = 300_000) -> None:
        self.capacity = capacity
        self.offset = self.size = 0
        self.reset(n)

    def reset(self, n: int) -> None:
        self.offset += self.size
        self.n = n
        self.t = np.empty(self.capacity)
        self.x = np.empty((self.capacity, n))
        self.v = np.empty((self.capacity, n))
        self.f = np.empty(self.capacity)
        self.e = np.empty((self.capacity, len(ENERGY_COLUMNS)))
        self.s = np.empty((self.capacity, 2 * n))
        self.g = np.empty((self.capacity, 2))
        self.size = 0

    def extend(
        self,
        ts: np.ndarray,
        xs: np.ndarray,
        vs: np.ndarray,
        fs: np.ndarray,
        es: np.ndarray,
        ss: np.ndarray,
        gs: np.ndarray | None = None,
    ) -> None:
        new = (ts, xs, vs, fs, es, ss, np.zeros((ts.size, 2)) if gs is None else gs)
        k = ts.size
        if k == 0:
            return
        if k >= self.capacity:
            new = tuple(a[-self.capacity :] for a in new)
            self.offset += k - self.capacity  # samples never stored still count
            k = self.capacity
        arrays = (self.t, self.x, self.v, self.f, self.e, self.s, self.g)
        if self.size + k > self.capacity:
            # Drop the oldest half (or more) to make room; amortized O(1).
            keep = min(self.size, self.capacity // 2, self.capacity - k)
            start = self.size - keep
            for a in arrays:
                a[:keep] = a[start : self.size]
            self.size = keep
            self.offset += start
        s = slice(self.size, self.size + k)
        for a, b in zip(arrays, new):
            a[s] = b
        self.size += k

    def window(self, seconds: float) -> tuple[np.ndarray, ...]:
        """Views (t, x, v, f, e, s, g) of the samples within the last `seconds` of simulated time."""
        i0 = 0
        if self.size:
            i0 = int(np.searchsorted(self.t[: self.size], self.t[self.size - 1] - seconds))
        s = slice(i0, self.size)
        return self.t[s], self.x[s], self.v[s], self.f[s], self.e[s], self.s[s], self.g[s]
