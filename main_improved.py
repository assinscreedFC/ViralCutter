import os
import sys

# Suppress unnecessary logs before importing heavy libs
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')

from scripts.pipeline.cli import build_parser
from scripts.pipeline.input_resolver import resolve_input
from scripts.pipeline.config_prompts import resolve_config
from scripts.pipeline.runner import run_pipeline


def main() -> None:
    args = build_parser().parse_args()
    ctx = resolve_input(args)
    ctx = resolve_config(ctx)
    run_pipeline(ctx)


if __name__ == "__main__":
    main()
