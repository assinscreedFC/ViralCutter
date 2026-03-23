"""Isolated pipeline worker — runs in a separate process.

IMPORTANT: This module must NOT import any heavy libraries (torch, whisperx,
onnxruntime) at the top level. On Windows, multiprocessing uses 'spawn',
which re-imports the target module in the child process. All heavy imports
happen INSIDE the worker function, after env vars are set.
"""
from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
import sys
import traceback
import warnings


def pipeline_worker(
    args_dict: dict,
    progress_q: multiprocessing.Queue,
    env_vars: dict,
    working_dir: str,
) -> None:
    """Run the pipeline in an isolated process.

    Args:
        args_dict: Flat dict of pipeline parameters (GUI values).
        progress_q: Queue for sending log records and control messages.
        env_vars: Environment variables to set before importing (API keys, etc.).
        working_dir: Project root directory.
    """
    # 1. Set environment FIRST (before any heavy import)
    os.environ.update(env_vars)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["ORT_LOGGING_LEVEL"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # 2. Ensure working directory is on sys.path
    if working_dir not in sys.path:
        sys.path.insert(0, working_dir)
    os.chdir(working_dir)

    # 3. Suppress warnings
    warnings.filterwarnings("ignore")

    # 4. Redirect ALL logging to the Queue
    root = logging.getLogger()
    root.handlers.clear()
    qh = logging.handlers.QueueHandler(progress_q)
    root.addHandler(qh)
    root.setLevel(logging.INFO)

    # 5. NOW import pipeline modules (heavy libs load here)
    try:
        from webui.pipeline_bridge import build_context_from_dict
        from scripts.pipeline.runner import run_pipeline

        ctx = build_context_from_dict(args_dict)
        run_pipeline(ctx)

        progress_q.put({
            "type": "done",
            "project_folder": ctx.project_folder or "",
        })
    except SystemExit as e:
        progress_q.put({
            "type": "error",
            "text": f"Pipeline exited with code {e.code}",
        })
    except Exception:
        progress_q.put({
            "type": "error",
            "text": traceback.format_exc(),
        })
