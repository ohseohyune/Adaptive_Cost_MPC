"""Guards that a non-canonical MuJoCo build cannot run a canonical experiment.

The failure this prevents was measured: one D0 seed-7 checkpoint scored 1.00
under mujoco 3.4.0 and 0.92 under 3.3.7, and nothing in the pipeline noticed.
These tests check the gate itself -- that the mismatch stops the run *before*
a model, a checkpoint, or an episode exists, and that it stops it in a
subprocess too.

Run:  python tests/runtime_environment_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import acmpc.runtime_environment as runtime  # noqa: E402
from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, run_box_catch  # noqa: E402
from acmpc.main_acmpc_box_catch_curriculum import (  # noqa: E402
    CurriculumBoxCatchConfig,
    run_curriculum_box_catch,
)

TRAINER = ROOT / "acmpc" / "main_acmpc_box_catch_curriculum.py"
CHECKPOINT = (
    ROOT / "sweep_results/constraint_ablation_20260804/pilot/D0/seed_7/checkpoint.pt"
)
REQUIRED_STAMP_FIELDS = (
    "python_version",
    "python_executable",
    "mujoco_python_version",
    "mujoco_native_version",
    "torch_version",
    "numpy_version",
    "canonical_environment",
    "scene_xml_sha256",
    "robot_xml_sha256",
)


# Bound once, so a test that patches runtime.runtime_versions can still build
# its fake versions from the real ones instead of recursing into itself.
_REAL_VERSIONS = runtime.runtime_versions


def _versions(package: str, native: str) -> dict:
    versions = dict(_REAL_VERSIONS())
    versions["mujoco_python_version"] = package
    versions["mujoco_native_version"] = native
    return versions


def test_a_canonical_environment_passes() -> None:
    print("Test A: canonical environment")
    stamp = runtime.validate_runtime_environment(context="test A")
    assert stamp["canonical_environment"] is True, stamp
    assert stamp["mujoco_python_version"] == runtime.CANONICAL_MUJOCO_VERSION
    assert stamp["mujoco_native_version"] == runtime.CANONICAL_MUJOCO_VERSION
    for field in REQUIRED_STAMP_FIELDS:
        assert field in stamp, f"missing stamp field {field}"
    print(
        f"  package={stamp['mujoco_python_version']} "
        f"native={stamp['mujoco_native_version']} "
        f"interpreter={stamp['python_executable']}"
    )


def _expect_refusal(callable_, label: str, expected_fragments: tuple[str, ...]) -> None:
    try:
        callable_()
    except RuntimeError as error:
        message = str(error)
        for fragment in expected_fragments:
            assert fragment in message, f"{label}: message lacks {fragment!r}:\n{message}"
        print(f"  {label}: RuntimeError raised, message carries expected/actual")
        return
    raise AssertionError(f"{label}: no RuntimeError raised")


def test_b_package_mismatch() -> None:
    print("Test B: mujoco Python package version mismatch")
    _expect_refusal(
        lambda: runtime.validate_runtime_environment(
            context="test B", versions=_versions("3.3.7", "3.3.7")
        ),
        "package+native 3.3.7",
        ("expected=3.4.0", "actual=3.3.7", sys.executable),
    )

    # And through a real entry point: the episode must abort before MuJoCo is
    # touched. A mismatched build makes runtime_versions() return 3.3.7, so
    # patch that single seam and confirm nothing downstream ran.
    original = runtime.runtime_versions
    runtime.runtime_versions = lambda: _versions("3.3.7", "3.3.7")
    try:
        _expect_refusal(
            lambda: run_box_catch(
                AcmpcBoxCatchConfig(
                    seed=7,
                    device="cpu",
                    online_learning=False,
                    checkpoint_path=str(CHECKPOINT),
                )
            ),
            "run_box_catch",
            ("expected=3.4.0", "actual=3.3.7"),
        )
        _expect_refusal(
            lambda: run_curriculum_box_catch(
                CurriculumBoxCatchConfig(
                    episodes=1,
                    device="cpu",
                    online_learning=False,
                    load_checkpoint=True,
                    checkpoint_path=str(CHECKPOINT),
                    log_path=None,
                    use_wandb=False,
                )
            ),
            "run_curriculum_box_catch",
            ("expected=3.4.0", "actual=3.3.7"),
        )
    finally:
        runtime.runtime_versions = original


def test_c_native_mismatch() -> None:
    print("Test C: native runtime mismatch with a canonical Python package")
    try:
        runtime.validate_runtime_environment(
            context="test C", versions=_versions("3.4.0", "3.3.7")
        )
    except RuntimeError as error:
        message = str(error)
        assert "mujoco Python package  expected=3.4.0 actual=3.4.0" in message, message
        assert "MuJoCo native runtime  expected=3.4.0 actual=3.3.7" in message, message
        print("  package-only agreement does not pass; both lines reported")
        return
    raise AssertionError("native mismatch was not refused")


def test_d_subprocess_guard() -> None:
    print("Test D: subprocess refuses before simulating")
    injector = (
        "import sys;"
        f"sys.path.insert(0, {str(ROOT)!r});"
        "import acmpc.runtime_environment as r;"
        "_real = r.runtime_versions;"
        "r.runtime_versions = lambda: dict(_real(), "
        "mujoco_python_version='3.3.7', mujoco_native_version='3.3.7');"
        "sys.argv = ["
        f"{str(TRAINER)!r}, '--evaluation-replay', '--evaluation-episodes', '1',"
        f" '--device', 'cpu', '--checkpoint', {str(CHECKPOINT)!r}]; "
        "import runpy;"
        f"runpy.run_path({str(TRAINER)!r}, run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", injector], cwd=ROOT, capture_output=True, text=True
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, "child exited 0 in a non-canonical environment"
    assert "expected=3.4.0" in output and "actual=3.3.7" in output, output[-2000:]
    assert sys.executable in output, "child error does not name its interpreter"
    assert "episodes=" not in completed.stdout, "child ran episodes before failing"
    print(f"  child exit={completed.returncode}, no episode ran, interpreter reported")


def test_f_result_metadata() -> None:
    print("Test F: result.json metadata matches this interpreter")
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "result.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(TRAINER),
                "--evaluation-replay",
                "--evaluation-episodes",
                "1",
                "--evaluation-seed",
                "100000",
                "--device",
                "cpu",
                "--checkpoint",
                str(CHECKPOINT),
                "--log",
                str(log),
                "--weight-delta-fraction",
                "0.65",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
        assert "[runtime]" in completed.stdout, "no runtime line at process start"
        stored = json.loads(log.read_text())["environment"]
    current = runtime.runtime_versions()
    current["canonical_environment"] = True
    for field in REQUIRED_STAMP_FIELDS:
        assert field in stored, f"result.json environment lacks {field}"
        assert stored[field] == current[field], (
            f"{field}: log={stored[field]!r} current={current[field]!r}"
        )
    print(f"  all {len(REQUIRED_STAMP_FIELDS)} fields present and exact-matching")


def main() -> None:
    assert CHECKPOINT.exists(), f"missing checkpoint: {CHECKPOINT}"
    test_a_canonical_environment_passes()
    test_b_package_mismatch()
    test_c_native_mismatch()
    test_d_subprocess_guard()
    test_f_result_metadata()
    print("all runtime environment tests passed")


if __name__ == "__main__":
    main()
