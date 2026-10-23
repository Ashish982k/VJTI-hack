"""Immutable collection snapshots, atomically published; no pickle deserialization."""

import json
import os
from pathlib import Path
import re
import uuid

import numpy as np


def validate_collection(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Collection must be 1-80 letters, digits, hyphens or underscores")
    return name


class Store:
    def __init__(self, root):
        self.root = Path(root)

    def location(self, collection):
        return self.root / "collections" / validate_collection(collection)

    def save(self, collection, payload, vectors, raw):
        location = self.location(collection)
        generation = uuid.uuid4().hex
        target = location / generation
        target.mkdir(parents=True)
        (target / "raw").mkdir()
        for doc_id, content in raw.items():
            (target / "raw" / f"{doc_id}.pdf").write_bytes(content)
        (target / "index.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(target / "vectors.npz", **vectors)
        pointer = location / f"current-{generation}.tmp"
        pointer.write_text(generation, encoding="ascii")
        os.replace(pointer, location / "CURRENT")
        return generation

    def generation(self, collection):
        try:
            value = (self.location(collection) / "CURRENT").read_text(encoding="ascii")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Collection {collection!r} has not been indexed") from e
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("Invalid index generation")
        return value

    def load(self, collection):
        generation = self.generation(collection)
        folder = self.location(collection) / generation
        payload = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        with np.load(folder / "vectors.npz", allow_pickle=False) as vectors:
            return generation, payload, {k: vectors[k] for k in vectors.files}
