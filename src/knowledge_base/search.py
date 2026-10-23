from collections import Counter, defaultdict
import math

import numpy as np

from .text import terms


class BM25:
    def __init__(self, data):
        self.postings = data["postings"]
        self.lengths = np.asarray(data["lengths"], dtype=np.float32)
        self.average = max(1.0, float(self.lengths.mean()))

    @staticmethod
    def build(chunks):
        postings, lengths = defaultdict(list), []
        for i, chunk in enumerate(chunks):
            words = terms(chunk["text"])
            lengths.append(len(words))
            for term, count in Counter(words).items():
                postings[term].append([i, count])
        return {"postings": dict(postings), "lengths": lengths}

    def scores(self, query):
        scores = np.zeros(len(self.lengths), dtype=np.float32)
        for term in set(terms(query)):
            posting = self.postings.get(term, [])
            if not posting:
                continue
            idf = math.log(1 + (len(scores) - len(posting) + 0.5) / (len(posting) + 0.5))
            indices = np.array([p[0] for p in posting])
            tf = np.array([p[1] for p in posting])
            scores[indices] += (
                idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[indices] / self.average))
            )
        return scores


def rank(dense, sparse, allowed, limit, hybrid):
    dense = np.where(allowed, dense, -np.inf)
    if not hybrid:
        return [(int(i), float(dense[i])) for i in np.argsort(-dense, kind="stable")[:limit] if dense[i] > 0]
    fused = defaultdict(float)
    for scores in (dense, np.where(allowed, sparse, -np.inf)):
        for position, index in enumerate(np.argsort(-scores, kind="stable")[:limit]):
            if scores[index] > 0:
                fused[int(index)] += 1 / (60 + position + 1)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
