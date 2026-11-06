"""Bounded train-only kNN-LM probability kernel, CPU NumPy reference.

Based on Khandelwal et al., https://arxiv.org/abs/1911.00172.
Caller-supplied provenance is validated structurally, not independently proven.
A future extractor must verify actual training membership and frozen hashes.
"""
import re
import numpy as np


class TrainOnlyKnnMemory:
    MAX_ROWS = 20000
    MAX_DIM = 160
    MAX_BYTES = 64 * 2**20

    def __init__(self, keys, values, case_ids, *, source_split, checkpoint_sha256, vocab_size=260):
        if source_split != 'train':
            raise ValueError('only verified train provenance is supported')
        if not isinstance(checkpoint_sha256, str) or not re.fullmatch('[0-9a-f]{64}', checkpoint_sha256):
            raise ValueError('checkpoint binding required')
        if type(vocab_size) is not int or not 1 <= vocab_size <= 260:
            raise ValueError('bounded vocabulary required')
        if not isinstance(keys, np.ndarray) or keys.dtype != np.float32 or keys.ndim != 2:
            raise ValueError('preallocated float32 keys required')
        n, d = keys.shape
        if not 0 <= n <= self.MAX_ROWS or not 1 <= d <= self.MAX_DIM:
            raise ValueError('memory shape exceeds bounded reference domain')
        if not isinstance(values, np.ndarray) or values.dtype != np.int64 or values.shape != (n,):
            raise ValueError('one integer train target per key required')
        if not isinstance(case_ids, (list, tuple)) or len(case_ids) != n or any(not isinstance(x, str) or not x for x in case_ids):
            raise ValueError('one nonempty training case ID per key required')
        # Owned FP32 keys, FP64 distance cache, values, norms. Check before copying.
        expected = n * d * 12 + n * 16
        if expected > self.MAX_BYTES:
            raise ValueError('memory byte budget exceeded')
        if not np.isfinite(keys).all() or (values < 0).any() or (values >= vocab_size).any():
            raise ValueError('nonfinite keys or target outside vocabulary')
        self._keys = keys.copy()
        self._keys64 = keys.astype(np.float64)
        self._values = values.copy()
        self._norms = np.einsum('nd,nd->n', self._keys64, self._keys64)
        self.case_ids = tuple(case_ids)
        self.checkpoint_sha256 = checkpoint_sha256
        self.vocab_size = vocab_size
        self.dimension = d
        self.payload_bytes = sum(x.nbytes for x in (self._keys, self._keys64, self._values, self._norms))
        assert self.payload_bytes == expected
        for x in (self._keys, self._keys64, self._values, self._norms):
            x.flags.writeable = False

    def distribution(self, query, *, k=8, temperature=1.):
        if type(k) is not int or not 1 <= k <= 32:
            raise ValueError('bounded integer k required')
        if not np.isscalar(temperature) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('positive finite temperature required')
        q = np.asarray(query)
        if q.shape != (self.dimension,) or not np.isfinite(q).all():
            raise ValueError('finite query with matching dimension required')
        if not len(self._values):
            return None
        q = q.astype(np.float64)
        distances = np.maximum(0., self._norms - 2 * (self._keys64 @ q) + q @ q)
        # Stable tie-breaking by original train-row index.
        indices = np.lexsort((np.arange(len(distances)), distances))[:min(k, len(distances))]
        scores = -distances[indices] / float(temperature)
        weights = np.exp(scores - scores.max()); weights /= weights.sum()
        result = np.zeros(self.vocab_size, dtype=np.float64)
        np.add.at(result, self._values[indices], weights)
        return result

    def mix(self, lm_probs, query, *, weight=.25, k=8, temperature=1.):
        p = np.asarray(lm_probs)
        if p.shape != (self.vocab_size,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1., atol=1e-6):
            raise ValueError('normalized finite LM distribution required')
        if not np.isscalar(weight) or not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('memory weight must be in [0,1]')
        if weight == 0 or not len(self._values):
            return p.copy()
        memory = self.distribution(query, k=k, temperature=temperature)
        return (1 - weight) * p + weight * memory
