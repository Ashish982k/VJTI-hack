"""Exercise real HTTP /index and /query against a temporary local server."""

import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

import httpx
import pymupdf


def main():
    with tempfile.TemporaryDirectory() as directory:
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((40, 50), "Acme revenue was INR 100 million in 2026.")
            content = pdf.tobytes()
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("Acme_Annual_Report_FY2026.pdf", content)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = {
            **os.environ,
            "KB_DATA_DIR": directory,
            "KB_EMBEDDING_BACKEND": "hash",
            "KB_LLM_BACKEND": "offline",
            "KB_OCR_BACKEND": "none",
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        }
        log_path = Path(directory) / "server.log"
        with log_path.open("w") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "knowledge_base.api:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                env=env,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False) as client:
                    for _ in range(60):
                        if server.poll() is not None:
                            raise RuntimeError(log_path.read_text())
                        try:
                            client.post("/query", json={"collection_name": "missing", "query": "revenue"})
                            break
                        except httpx.ConnectError:
                            time.sleep(0.25)
                    else:
                        raise RuntimeError("Server did not become ready")
                    result = client.post(
                        "/index",
                        data={"collection_name": "smoke"},
                        files={"file": ("docs.zip", archive.getvalue(), "application/zip")},
                    )
                    result.raise_for_status()
                    answer = client.post(
                        "/query", json={"collection_name": "smoke", "query": "Acme revenue in 2026"}
                    )
                    answer.raise_for_status()
                    response = answer.json()
                    assert "100" in response["answer"], response
                    assert response["chunks"][0]["page_numbers"] == [1]
                    assert all(
                        set(s) == {"step_name", "description", "input", "output"} for s in response["trace"]
                    )
                    print(
                        "HTTP smoke passed: ZIP index, original evidence, page citations and exact trace fields."
                    )
            finally:
                server.terminate()
                server.wait(timeout=15)


if __name__ == "__main__":
    main()
