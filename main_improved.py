import os
import sys

# Suppress unnecessary logs before importing heavy libs
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')

logger = logging.getLogger(__name__)

from scripts.pipeline.cli import build_parser
from scripts.pipeline.config import ProcessingConfig
from scripts.pipeline.errors import PipelineError
from scripts.pipeline.input_resolver import resolve_input
from scripts.pipeline.config_prompts import resolve_config
from scripts.pipeline.runner import run_pipeline


def main() -> None:
    try:
        args = build_parser().parse_args()
        ctx = resolve_input(args)
        ctx = resolve_config(ctx)
        ctx.config = ProcessingConfig.from_namespace(ctx.args)
        run_pipeline(ctx)
    except PipelineError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
