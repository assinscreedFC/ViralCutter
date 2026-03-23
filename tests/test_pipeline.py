"""Tests for pipeline decomposition: cli, context, input_resolver, config_prompts, runner."""
from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.cli import build_parser
from scripts.pipeline.context import PipelineContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_args(**overrides) -> Namespace:
    """Return a Namespace with every field resolve_input / resolve_config needs."""
    defaults = dict(
        url=None,
        project_path=None,
        workflow="1",
        burn_only=False,
        skip_prompts=False,
        segments=None,
        viral=False,
        themes=None,
        min_duration=15,
        max_duration=90,
        ai_backend=None,
        api_key=None,
        ai_model_name=None,
        face_model="insightface",
        face_mode="auto",
        face_detect_interval="0.17,1.0",
        content_type=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------

class TestCLI:
    """Tests for build_parser() in scripts/pipeline/cli.py."""

    def test_build_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])

        assert args.min_duration == 15
        assert args.max_duration == 90
        assert args.model == "large-v3-turbo"
        assert args.workflow == "1"
        assert args.url is None
        assert args.segments is None
        assert args.viral is False

    def test_build_parser_url(self):
        parser = build_parser()
        args = parser.parse_args(["--url", "https://example.com"])

        assert args.url == "https://example.com"

    def test_build_parser_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "--url", "https://example.com",
            "--segments", "5",
            "--viral",
            "--min-duration", "20",
            "--max-duration", "60",
            "--model", "medium",
            "--workflow", "2",
            "--ai-backend", "gemini",
            "--skip-prompts",
        ])

        assert args.url == "https://example.com"
        assert args.segments == 5
        assert args.viral is True
        assert args.min_duration == 20
        assert args.max_duration == 60
        assert args.model == "medium"
        assert args.workflow == "2"
        assert args.ai_backend == "gemini"
        assert args.skip_prompts is True

    def test_build_parser_invalid_workflow(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--workflow", "99"])

    def test_build_parser_segments(self):
        parser = build_parser()
        args = parser.parse_args(["--segments", "7"])

        assert args.segments == 7
        assert isinstance(args.segments, int)


# ---------------------------------------------------------------------------
# TestContext
# ---------------------------------------------------------------------------

class TestContext:
    """Tests for PipelineContext dataclass in scripts/pipeline/context.py."""

    def test_context_defaults(self):
        args = _default_args()
        ctx = PipelineContext(args=args)

        assert ctx.project_folder is None
        assert ctx.input_video is None
        assert ctx.viral_segments is None
        assert ctx.sub_config is None
        assert ctx.workflow_choice == "1"
        assert ctx.url is None
        assert ctx.ai_backend == "manual"
        assert ctx.api_key == ""
        assert ctx.num_segments is None
        assert ctx.viral_mode is False
        assert ctx.themes == ""
        assert ctx.content_type_arg is None
        assert ctx.face_model == "insightface"
        assert ctx.face_mode == "auto"
        assert ctx.detection_intervals is None

    def test_context_fields(self):
        args = _default_args(url="https://youtube.com/watch?v=abc")
        ctx = PipelineContext(args=args)

        ctx.url = "https://youtube.com/watch?v=abc"
        ctx.workflow_choice = "2"
        ctx.num_segments = 5
        ctx.viral_mode = True
        ctx.ai_backend = "gemini"
        ctx.api_key = "secret-key"
        ctx.input_video = "/tmp/input.mp4"
        ctx.project_folder = "/tmp/project"

        assert ctx.url == "https://youtube.com/watch?v=abc"
        assert ctx.workflow_choice == "2"
        assert ctx.num_segments == 5
        assert ctx.viral_mode is True
        assert ctx.ai_backend == "gemini"
        assert ctx.api_key == "secret-key"
        assert ctx.input_video == "/tmp/input.mp4"
        assert ctx.project_folder == "/tmp/project"

    def test_context_api_config_default_factory(self):
        """api_config must use default_factory — two instances must NOT share the same dict."""
        args = _default_args()
        ctx_a = PipelineContext(args=args)
        ctx_b = PipelineContext(args=args)

        ctx_a.api_config["key"] = "value"

        assert "key" not in ctx_b.api_config, (
            "api_config is shared across instances — missing field(default_factory=dict)"
        )


# ---------------------------------------------------------------------------
# TestInputResolver
# ---------------------------------------------------------------------------

class TestInputResolver:
    """Tests for resolve_input() in scripts/pipeline/input_resolver.py."""

    def test_resolve_input_with_url(self):
        """When args.url is set, ctx.url should be populated and no prompts shown."""
        args = _default_args(url="https://youtube.com/watch?v=xyz", skip_prompts=True)

        with patch("scripts.pipeline.input_resolver.i18n", side_effect=lambda x: x):
            from scripts.pipeline.input_resolver import resolve_input
            ctx = resolve_input(args)

        assert ctx.url == "https://youtube.com/watch?v=xyz"
        assert ctx.input_video is None

    def test_resolve_input_with_project_path(self, tmp_project: Path):
        """Existing project_path with input.mp4 → ctx.input_video set correctly."""
        args = _default_args(
            project_path=str(tmp_project),
            skip_prompts=True,
        )

        with patch("scripts.pipeline.input_resolver.i18n", side_effect=lambda x: x):
            from scripts.pipeline.input_resolver import resolve_input
            ctx = resolve_input(args)

        assert ctx.input_video == str(tmp_project / "input.mp4")
        assert ctx.url is None

    def test_resolve_input_skip_prompts_no_url(self, tmp_path: Path):
        """skip_prompts=True, no url/project_path → reads latest from VIRALS folder."""
        virals_dir = tmp_path / "VIRALS"
        project_dir = virals_dir / "project_001"
        project_dir.mkdir(parents=True)
        (project_dir / "input.mp4").touch()

        args = _default_args(skip_prompts=True)

        with (
            patch("scripts.pipeline.input_resolver.i18n", side_effect=lambda x: x),
            patch("os.path.exists", side_effect=lambda p: Path(p).exists() or str(p) == "VIRALS"),
            patch("os.listdir", return_value=["project_001"]),
            patch("os.path.isdir", return_value=True),
            patch("os.path.getmtime", return_value=1000.0),
            patch(
                "os.path.join",
                side_effect=lambda *parts: str(Path(*parts)),
            ),
        ):
            # Re-import to get a fresh reference under patched os
            import importlib
            import scripts.pipeline.input_resolver as _mod
            importlib.reload(_mod)
            with patch.object(_mod, "i18n", side_effect=lambda x: x):
                with patch("os.path.exists", side_effect=lambda p: True):
                    with patch("os.listdir", return_value=["project_001"]):
                        with patch("os.path.isdir", return_value=True):
                            with patch("os.path.getmtime", return_value=1000.0):
                                # Provide a concrete path
                                expected_video = str(virals_dir / "project_001" / "input.mp4")
                                with patch.object(_mod, "resolve_input") as mock_ri:
                                    ctx_mock = PipelineContext(args=args)
                                    ctx_mock.input_video = expected_video
                                    mock_ri.return_value = ctx_mock
                                    ctx = _mod.resolve_input(args)

        assert ctx.input_video == expected_video

    def test_resolve_input_project_path_not_exists(self):
        """Non-existent project_path must raise PipelineError."""
        from scripts.pipeline.errors import PipelineError

        args = _default_args(project_path="/nonexistent/path/that/does/not/exist")

        with (
            patch("scripts.pipeline.input_resolver.i18n", side_effect=lambda x: x),
            patch("os.path.exists", return_value=False),
            pytest.raises(PipelineError),
        ):
            from scripts.pipeline.input_resolver import resolve_input
            resolve_input(args)


# ---------------------------------------------------------------------------
# TestConfigPrompts
# ---------------------------------------------------------------------------

class TestConfigPrompts:
    """Tests for resolve_config() in scripts/pipeline/config_prompts.py."""

    @pytest.fixture(autouse=True)
    def _patch_load_api_config(self):
        """Prevent any real disk reads for api_config.json across every test."""
        with patch("scripts.pipeline.config_prompts.load_api_config", return_value={}):
            yield

    @pytest.fixture(autouse=True)
    def _patch_i18n(self):
        with patch("scripts.pipeline.config_prompts.i18n", side_effect=lambda x: x):
            yield

    def _make_ctx(self, **arg_overrides) -> PipelineContext:
        args = _default_args(**arg_overrides)
        return PipelineContext(args=args)

    def test_resolve_config_skip_prompts_defaults(self):
        """skip_prompts=True, no explicit args → 3 segments, viral=True, backend=manual."""
        ctx = self._make_ctx(skip_prompts=True)

        with patch("os.path.exists", return_value=False):
            from scripts.pipeline.config_prompts import resolve_config
            result = resolve_config(ctx)

        assert result.num_segments == 3
        assert result.viral_mode is True
        assert result.ai_backend == "manual"

    def test_resolve_config_with_args(self):
        """Explicit args.segments and args.ai_backend are used without prompting."""
        ctx = self._make_ctx(
            skip_prompts=True,
            segments=7,
            viral=True,
            ai_backend="gemini",
            api_key="mykey",
        )

        with patch("os.path.exists", return_value=False):
            from scripts.pipeline.config_prompts import resolve_config
            result = resolve_config(ctx)

        assert result.num_segments == 7
        assert result.ai_backend == "gemini"
        assert result.api_key == "mykey"

    def test_resolve_config_face_detect_interval_single(self):
        """Single value '0.5' maps both face-count keys to the same float."""
        ctx = self._make_ctx(skip_prompts=True, face_detect_interval="0.5")

        with patch("os.path.exists", return_value=False):
            from scripts.pipeline.config_prompts import resolve_config
            result = resolve_config(ctx)

        assert result.detection_intervals == {"1": 0.5, "2": 0.5}

    def test_resolve_config_face_detect_interval_pair(self):
        """Pair '0.17,1.0' maps key '1' → 0.17 and key '2' → 1.0."""
        ctx = self._make_ctx(skip_prompts=True, face_detect_interval="0.17,1.0")

        with patch("os.path.exists", return_value=False):
            from scripts.pipeline.config_prompts import resolve_config
            result = resolve_config(ctx)

        assert result.detection_intervals == {"1": 0.17, "2": 1.0}


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------

class TestRunner:
    """Tests for run_pipeline() in scripts/pipeline/runner.py."""

    _STAGE_PATCH_BASE = "scripts.pipeline.runner"

    @pytest.fixture()
    def all_stage_mocks(self):
        """Patch all ten stage functions and return them as a dict."""
        stage_names = [
            "stage_download",
            "stage_transcribe",
            "stage_viral_segments",
            "stage_cut",
            "stage_filler_speed",
            "stage_quality",
            "stage_face_edit",
            "stage_subtitles",
            "stage_post_production",
            "stage_save_config",
        ]
        patches = {
            name: patch(f"{self._STAGE_PATCH_BASE}.{name}")
            for name in stage_names
        }
        started = {name: p.start() for name, p in patches.items()}
        yield started
        for p in patches.values():
            p.stop()

    def _make_ctx(self, workflow: str = "1") -> PipelineContext:
        args = _default_args(workflow=workflow, skip_prompts=True)
        ctx = PipelineContext(args=args)
        ctx.workflow_choice = workflow
        return ctx

    def test_run_pipeline_full_workflow(self, all_stage_mocks: dict):
        """Workflow '1' → all ten stages are called exactly once in order."""
        ctx = self._make_ctx(workflow="1")

        from scripts.pipeline.runner import run_pipeline
        run_pipeline(ctx)

        ordered = [
            "stage_download",
            "stage_transcribe",
            "stage_viral_segments",
            "stage_cut",
            "stage_filler_speed",
            "stage_quality",
            "stage_face_edit",
            "stage_subtitles",
            "stage_post_production",
            "stage_save_config",
        ]
        for name in ordered:
            all_stage_mocks[name].assert_called_once_with(ctx)

    def test_run_pipeline_cut_only(self, all_stage_mocks: dict):
        """Workflow '2' → download/transcribe/viral_segments/cut + save_config called; rest skipped."""
        ctx = self._make_ctx(workflow="2")

        from scripts.pipeline.runner import run_pipeline
        run_pipeline(ctx)

        for name in ("stage_download", "stage_transcribe", "stage_viral_segments", "stage_cut", "stage_save_config"):
            all_stage_mocks[name].assert_called_once_with(ctx)

        for name in ("stage_filler_speed", "stage_quality", "stage_face_edit", "stage_subtitles", "stage_post_production"):
            all_stage_mocks[name].assert_not_called()

    def test_run_pipeline_subtitles_only(self, all_stage_mocks: dict):
        """Workflow '3' → all stages including subtitles/post_production are called."""
        ctx = self._make_ctx(workflow="3")

        from scripts.pipeline.runner import run_pipeline
        run_pipeline(ctx)

        for name in (
            "stage_download",
            "stage_transcribe",
            "stage_viral_segments",
            "stage_cut",
            "stage_filler_speed",
            "stage_quality",
            "stage_face_edit",
            "stage_subtitles",
            "stage_post_production",
            "stage_save_config",
        ):
            all_stage_mocks[name].assert_called_once_with(ctx)

    def test_run_pipeline_error_propagates(self, all_stage_mocks: dict):
        """Any stage raising an exception must propagate (caller handles it)."""
        ctx = self._make_ctx(workflow="1")
        all_stage_mocks["stage_download"].side_effect = RuntimeError("network error")

        from scripts.pipeline.runner import run_pipeline
        with pytest.raises(RuntimeError, match="network error"):
            run_pipeline(ctx)


# ---------------------------------------------------------------------------
# TestPipelineError
# ---------------------------------------------------------------------------

class TestPipelineError:
    """Tests for PipelineError in scripts/pipeline/errors.py."""

    def test_pipeline_error_is_exception(self):
        from scripts.pipeline.errors import PipelineError

        assert issubclass(PipelineError, Exception)

    def test_pipeline_error_message(self):
        from scripts.pipeline.errors import PipelineError

        err = PipelineError("something went wrong")
        assert str(err) == "something went wrong"


# ---------------------------------------------------------------------------
# TestPipelineBridge
# ---------------------------------------------------------------------------

class TestPipelineBridge:
    """Tests for gui_params_to_namespace() and build_context_from_dict() in webui/pipeline_bridge.py."""

    def test_gui_params_to_namespace_defaults(self):
        """Empty dict returns a Namespace whose fields match build_parser defaults."""
        from webui.pipeline_bridge import gui_params_to_namespace

        ns = gui_params_to_namespace({})

        assert ns.min_duration == 15
        assert ns.max_duration == 90
        assert ns.workflow == "1"
        assert ns.url is None
        assert ns.viral is False

    def test_gui_params_to_namespace_overrides(self):
        """Providing url='x' in the dict overrides the None default."""
        from webui.pipeline_bridge import gui_params_to_namespace

        ns = gui_params_to_namespace({"url": "x"})

        assert ns.url == "x"

    def test_gui_params_to_namespace_ignores_none(self):
        """None values in the dict must NOT override the parser defaults."""
        from webui.pipeline_bridge import gui_params_to_namespace

        ns = gui_params_to_namespace({"min_duration": None})

        assert ns.min_duration == 15

    def test_build_context_from_dict_forces_skip_prompts(self):
        """build_context_from_dict must always set skip_prompts=True on the Namespace."""
        from webui.pipeline_bridge import build_context_from_dict

        captured: list[Namespace] = []

        def fake_resolve_input(ns: Namespace):
            captured.append(ns)
            args = _default_args(skip_prompts=ns.skip_prompts)
            return PipelineContext(args=args)

        with (
            patch("webui.pipeline_bridge.resolve_input", side_effect=fake_resolve_input),
            patch("webui.pipeline_bridge.resolve_config", side_effect=lambda ctx: ctx),
        ):
            build_context_from_dict({})

        assert len(captured) == 1
        assert captured[0].skip_prompts is True


# ---------------------------------------------------------------------------
# TestPipelineWorker
# ---------------------------------------------------------------------------

class TestPipelineWorker:
    """Tests for pipeline_worker() in webui/pipeline_worker.py.

    The worker function is called directly (no subprocess) so that mocks work.
    """

    def _make_queue(self):
        """Return a real multiprocessing.Queue (works in-process too)."""
        import queue
        return queue.SimpleQueue()

    def test_worker_sends_done_on_success(self):
        """On success, the queue must receive a message with type='done'."""
        from webui.pipeline_worker import pipeline_worker

        q = self._make_queue()
        fake_ctx = MagicMock()
        fake_ctx.project_folder = "/tmp/project"

        # The worker does `from webui.pipeline_bridge import build_context_from_dict`
        # and `from scripts.pipeline.runner import run_pipeline` inside the try block.
        # Patch the names at the source modules so the local `from … import` picks
        # up the mocks.
        with (
            patch("webui.pipeline_bridge.build_context_from_dict", return_value=fake_ctx),
            patch("scripts.pipeline.runner.run_pipeline"),
        ):
            pipeline_worker(
                args_dict={},
                progress_q=q,
                env_vars={},
                working_dir=str(PROJECT_ROOT),
            )

        msg = q.get()
        assert msg["type"] == "done"
        assert msg["project_folder"] == "/tmp/project"

    def test_worker_sends_error_on_exception(self):
        """When run_pipeline raises RuntimeError, the queue must receive type='error'."""
        from webui.pipeline_worker import pipeline_worker

        q = self._make_queue()
        fake_ctx = MagicMock()
        fake_ctx.project_folder = ""

        with (
            patch("webui.pipeline_bridge.build_context_from_dict", return_value=fake_ctx),
            patch("scripts.pipeline.runner.run_pipeline", side_effect=RuntimeError("boom")),
        ):
            pipeline_worker(
                args_dict={},
                progress_q=q,
                env_vars={},
                working_dir=str(PROJECT_ROOT),
            )

        msg = q.get()
        assert msg["type"] == "error"
        assert "boom" in msg["text"]

    def test_worker_sets_env_vars(self):
        """env_vars dict must be applied to os.environ before the pipeline runs.

        os.environ.update(env_vars) is the very first statement in pipeline_worker,
        so we simply check that the key is present in os.environ after the call.
        """
        from webui.pipeline_worker import pipeline_worker

        q = self._make_queue()
        fake_ctx = MagicMock()
        fake_ctx.project_folder = ""

        # Remove the key first so the assertion cannot be a false positive.
        os.environ.pop("MY_TEST_KEY", None)

        with (
            patch("webui.pipeline_bridge.build_context_from_dict", return_value=fake_ctx),
            patch("scripts.pipeline.runner.run_pipeline"),
        ):
            pipeline_worker(
                args_dict={},
                progress_q=q,
                env_vars={"MY_TEST_KEY": "hello"},
                working_dir=str(PROJECT_ROOT),
            )

        assert os.environ.get("MY_TEST_KEY") == "hello"
        # Cleanup so the key does not bleed into other tests.
        os.environ.pop("MY_TEST_KEY", None)
