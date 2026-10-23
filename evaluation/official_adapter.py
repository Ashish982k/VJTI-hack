"""Execute the provided evaluator definitions unchanged, excluding its demo invocation."""
import ast
import hashlib
import json
import os
from pathlib import Path

OFFICIAL_SHA256 = "f75493648896a79fc28499f04bb4909baea9bf717c1b040d0c3fb3974aa5f583"
DEFAULT_NOTEBOOK = Path(__file__).parent / "official" / "Evaluation.ipynb"


def load_official(path=DEFAULT_NOTEBOOK):
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
        is_configuration = any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OPENCODE_API_KEY" for t in n.targets) for n in tree.body)
        if is_imports or is_configuration or names & {"OpenCodeClient", "RAGMetricsEvaluator", "evaluate_dataset"}:
            exec(compile(source, str(path), "exec"), namespace)
    key = os.getenv("OPENCODE_API_KEY")
    if not key:
        raise RuntimeError("Official evaluation requires OPENCODE_API_KEY")
    client = namespace["OpenCodeClient"](api_key=key)
    return namespace["RAGMetricsEvaluator"](client), namespace["evaluate_dataset"]
