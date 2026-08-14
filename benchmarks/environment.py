from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from typing import Any


def _run(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return completed.stdout.strip() or completed.stderr.strip() or None
    except Exception:
        return None


def _ollama_show(model: str) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=json.dumps({"name": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def collect_environment(llm_model: str, include_ollama: bool = True) -> dict[str, Any]:
    gpu = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])

    ollama_version = None
    ollama_runtime = None
    ollama_model = None
    ollama_llm_library = None
    if include_ollama:
        ollama_version = _run(["ollama", "--version"])
        ollama_runtime = _run(["ollama", "ps"])
        ollama_model = _ollama_show(llm_model)
        ollama_llm_library = os.environ.get("OLLAMA_LLM_LIBRARY")

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "nvidia_smi": gpu,
        "ollama_version": ollama_version,
        "ollama_runtime": ollama_runtime,
        "ollama_llm_library": ollama_llm_library,
        "ollama_model": ollama_model,
        "pip_freeze": pip_freeze.splitlines() if pip_freeze else None,
    }
