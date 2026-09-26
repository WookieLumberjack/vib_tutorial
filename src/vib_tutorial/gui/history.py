"""Growable buffer of recent simulation samples for the strip charts."""

from __future__ import annotations

import numpy as np


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
        self.size = 0

    def extend(self, ts: np.ndarray, xs: np.ndarray, vs: np.ndarray, fs: np.ndarray) -> None:
        k = ts.size
        if k == 0:
            return
        if k >= self.capacity:
            ts, xs, vs, fs = (a[-self.capacity :] for a in (ts, xs, vs, fs))
            k = ts.size
        if self.size + k > self.capacity:
            # Drop the oldest half (or more) to make room; amortized O(1).
            keep = min(self.size, self.capacity // 2, self.capacity - k)
            start = self.size - keep
            self.t[:keep] = self.t[start : self.size]
            self.x[:keep] = self.x[start : self.size]
            self.v[:keep] = self.v[start : self.size]
            self.f[:keep] = self.f[start : self.size]
            self.size = keep
        s = slice(self.size, self.size + k)
        self.t[s], self.x[s], self.v[s], self.f[s] = ts, xs, vs, fs
        self.size += k

    def window(self, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Views (t, x, v, f) of the samples within the last `seconds` of simulated time."""
        if self.size == 0:
            return self.t[:0], self.x[:0], self.v[:0], self.f[:0]
        t = self.t[: self.size]
        i0 = int(np.searchsorted(t, t[-1] - seconds))
        s = slice(i0, self.size)
        return t[i0:], self.x[s], self.v[s], self.f[s]
