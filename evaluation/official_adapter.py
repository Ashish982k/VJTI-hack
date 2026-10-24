"""Execute the provided evaluator definitions unchanged, excluding its demo invocation."""

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import time
import types

OFFICIAL_SHA256 = "f75493648896a79fc28499f04bb4909baea9bf717c1b040d0c3fb3974aa5f583"
DEFAULT_NOTEBOOK = Path(__file__).parent / "official" / "Evaluation.ipynb"


def _rate_limit_aware(requests_module, attempts=8):
    """HTTP-level shim for small free tiers: on 429, wait for the provider's reset and resend. The
    evaluator's prompts, parsing and formulas are untouched; it only sees the eventual response."""

    def post(*args, **kwargs):
        for _ in range(attempts):
            response = requests_module.post(*args, **kwargs)
            if response.status_code != 429:
                return response
            reset = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-tokens", "")
            match = re.fullmatch(r"(?:(\d+)m)?([\d.]+)(ms|s)?", reset.strip())
            wait = int(match[1] or 0) * 60 + float(match[2]) / (1000 if match[3] == "ms" else 1) if match else 10
            time.sleep(min(65.0, max(1.0, wait)))
        return response

    return types.SimpleNamespace(post=post, RequestException=requests_module.RequestException)


def load_official(path=DEFAULT_NOTEBOOK, model=None, base_url=None, api_key=None):
    """model/base_url/api_key override the configuration cell, which the notebook lets candidates change.
    Scores from any provider or model other than the notebook default are a proxy, not the official setup."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != OFFICIAL_SHA256:
        raise ValueError("Official evaluator checksum differs from the supplied original; refusing to score")
    notebook = json.loads(content)
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        is_imports = all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body)
        is_configuration = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "OPENCODE_API_KEY" for t in n.targets)
            for n in tree.body
        )
        if (
            is_imports
            or is_configuration
            or names & {"OpenCodeClient", "RAGMetricsEvaluator", "evaluate_dataset"}
        ):
            exec(compile(source, str(path), "exec"), namespace)
    key = api_key or os.getenv("OPENCODE_API_KEY")
    if not key:
        raise RuntimeError("Official evaluation requires OPENCODE_API_KEY")
    if base_url:
        namespace["requests"] = _rate_limit_aware(namespace["requests"])
    client = namespace["OpenCodeClient"](
        api_key=key,
        model=model or namespace["OPENCODE_MODEL"],
        base_url=base_url or namespace["OPENCODE_BASE_URL"],
    )
    return namespace["RAGMetricsEvaluator"](client), namespace["evaluate_dataset"]
