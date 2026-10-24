from collections import Counter, defaultdict
import math

import numpy as np

from .text import terms


class BM25:
    def __init__(self, data):
        self.postings = data["postings"]
        self.lengths = np.asarray(data["lengths"], dtype=np.float32)
        self.average = max(1.0, float(self.lengths.mean()))
        self.arrays = {}

    def _posting(self, term):
        if term not in self.arrays:
            posting = np.asarray(self.postings.get(term, []), dtype=np.int64).reshape(-1, 2)
            self.arrays[term] = (posting[:, 0], posting[:, 1].astype(np.float32))
        return self.arrays[term]

    @staticmethod
    def build(texts):
        postings, lengths = defaultdict(list), []
        for i, text in enumerate(texts):
            words = terms(text)
            lengths.append(len(words))
            for term, count in Counter(words).items():
                postings[term].append([i, count])
        return {"postings": dict(postings), "lengths": lengths}

    def scores(self, query):
        scores = np.zeros(len(self.lengths), dtype=np.float32)
        for term in set(terms(query)):
            indices, tf = self._posting(term)
            if not len(indices):
                continue
            idf = math.log(1 + (len(scores) - len(indices) + 0.5) / (len(indices) + 0.5))
            scores[indices] += (
                idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[indices] / self.average))
            )
        return scores


def rank(dense, sparse, allowed, limit, depth=None):
    """Reciprocal-rank fusion of dense and BM25 rankings, restricted to allowed units."""
    depth = depth or limit
    fused = defaultdict(float)
    for scores in (np.where(allowed, dense, -np.inf), np.where(allowed, sparse, -np.inf)):
        for position, index in enumerate(np.argsort(-scores, kind="stable")[:depth]):
            if scores[index] > 0:
                fused[int(index)] += 1 / (60 + position + 1)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]


class GroupedBM25(BM25):
    """BM25 over groups of units (e.g. sections) derived from the unit index: no extra postings stored."""

    def __init__(self, unit_index, groups, count):
        self.unit_index = unit_index
        self.groups = groups
        self.lengths = np.bincount(groups, weights=unit_index.lengths, minlength=count).astype(np.float32)
        self.average = max(1.0, float(self.lengths.mean()))
        self.arrays = {}

    def _posting(self, term):
        if term not in self.arrays:
            indices, tf = self.unit_index._posting(term)
            groups, inverse = np.unique(self.groups[indices], return_inverse=True)
            self.arrays[term] = (groups, np.bincount(inverse, weights=tf).astype(np.float32))
        return self.arrays[term]
