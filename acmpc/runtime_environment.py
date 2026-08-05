"""Canonical runtime check for every AC-MPC training/evaluation entry point.

Why this exists (measured 2026-08-05): the same D0 seed-7 checkpoint scored
1.00 in the training process and 0.92 in a cold-start replay. Same code, same
seeds, same scenarios -- different interpreters. mujoco 3.4.0 and 3.3.7 solve
one box-catch contact to ~3.4e-13 N of each other, which is invisible before
contact and decisive across the 5 s HOLD: 4/50 success labels flipped. Nothing
stopped, or even recorded, the wrong-interpreter run.

So: no simulation, no checkpoint load, no subprocess spawn happens before
``validate_runtime_environment`` has confirmed both the mujoco Python package
*and* the native library it wraps are the canonical version. A mismatch raises
RuntimeError; it is never a warning.

The two versions are checked separately on purpose -- the pip package and the
bundled native library are independently installable, and a package-only check
would pass an environment whose physics is not the physics we validated.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# The version every canonical training run, evaluation and ablation must use.
# Keep in sync with requirements.txt.
CANONICAL_MUJOCO_VERSION = "3.4.0"

SCENE_XML = ROOT / "model/robotis_ffw/scene_ffw_sg2_fixed_base_box_dynamic_squeeze.xml"
ROBOT_XML = ROOT / "model/robotis_ffw/ffw_sg2.xml"

# One WARNING per process when a run opts out, not one per episode.
_noncanonical_warning_printed = False
_dirty_model_warning_printed = False


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def conda_environment_name() -> Optional[str]:
    """Best-effort conda environment name, for the error message."""

    # sys.prefix first, and CONDA_* only when it agrees with it: a subprocess
    # launched from a base shell into another env inherits CONDA_DEFAULT_ENV=
    # base, which would name the wrong environment in the one message that
    # exists to tell the user which environment they are in.
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        return prefix.name
    name = os.environ.get("CONDA_DEFAULT_ENV")
    if name and os.environ.get("CONDA_PREFIX") == str(prefix):
        return name
    return str(prefix)


def _git(repository: Path, *arguments: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


# The two files that define the physics: the box-catch scene and the robot it
# includes. "Relevant" dirtiness below means these, not the whole repository --
# unrelated working-tree edits elsewhere must not block a canonical run.
_RELEVANT_MODEL_PATHS = ("robotis_ffw/ffw_sg2.xml", SCENE_XML.name)


def git_state() -> dict:
    """Commit identity of the code and of the physics model.

    The XML hashes alone can only say *that* a model changed. These fields say
    *which* revision to check out to get it back, which is what a fresh clone
    on another machine needs.
    """

    model_directory = SCENE_XML.parent.parent
    is_submodule = _git(ROOT, "ls-files", "--stage", "--", str(model_directory.name))
    model_is_submodule = bool(is_submodule and is_submodule.startswith("160000"))
    pointer = is_submodule.split()[1] if model_is_submodule else None

    parent_status = _git(ROOT, "status", "--porcelain") or ""
    model_status = _git(model_directory, "status", "--porcelain") or ""
    relevant = [
        line
        for line in model_status.splitlines()
        if any(path in line for path in _RELEVANT_MODEL_PATHS)
    ]
    return {
        "parent_repo_commit": _git(ROOT, "rev-parse", "HEAD"),
        "parent_repo_branch": _git(ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
        "parent_repo_dirty": bool(parent_status),
        "parent_repo_tracked_dirty": bool(
            _git(ROOT, "status", "--porcelain", "--untracked-files=no")
        ),
        "parent_repo_untracked": any(
            line.startswith("??") for line in parent_status.splitlines()
        ),
        "model_is_submodule": model_is_submodule,
        "submodule_pointer_commit": pointer,
        "model_repo_commit": _git(model_directory, "rev-parse", "HEAD"),
        "model_repo_branch": _git(model_directory, "rev-parse", "--abbrev-ref", "HEAD"),
        "model_repo_dirty": bool(model_status),
        "model_repo_tracked_dirty": bool(
            _git(model_directory, "status", "--porcelain", "--untracked-files=no")
        ),
        "model_repo_untracked": any(
            line.startswith("??") for line in model_status.splitlines()
        ),
        "model_files_dirty": bool(relevant),
    }


def runtime_versions() -> dict:
    """Everything outside this repo that the simulation's bits depend on."""

    import mujoco
    import numpy
    import torch

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "conda_environment": conda_environment_name(),
        "mujoco_python_version": mujoco.__version__,
        "mujoco_native_version": mujoco.mj_versionString(),
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "scene_xml_sha256": _digest(SCENE_XML),
        "robot_xml_sha256": _digest(ROBOT_XML),
        **git_state(),
    }


def _mismatch_message(versions: dict, context: str) -> str:
    return "\n".join(
        [
            f"non-canonical MuJoCo runtime{f' ({context})' if context else ''}:",
            f"  mujoco Python package  expected={CANONICAL_MUJOCO_VERSION} "
            f"actual={versions['mujoco_python_version']}",
            f"  MuJoCo native runtime  expected={CANONICAL_MUJOCO_VERSION} "
            f"actual={versions['mujoco_native_version']}",
            f"  sys.executable         {versions['python_executable']}",
            f"  conda environment      {versions['conda_environment']}",
            "",
            "Results from different MuJoCo builds are not comparable: 3.3.7 and "
            "3.4.0 differ by ~3e-13 N per contact solve, which flipped 4/50 "
            "five-second HOLD outcomes on one D0 checkpoint.",
            f"Use the canonical environment with mujoco=={CANONICAL_MUJOCO_VERSION}.",
            "Forensic replay of an older result may pass "
            "--allow-noncanonical-mujoco; training and canonical evaluation may "
            "not.",
        ]
    )


def validate_runtime_environment(
    *,
    context: str = "",
    allow_noncanonical: bool = False,
    versions: Optional[dict] = None,
) -> dict:
    """Return the environment stamp, or raise before anything expensive runs.

    ``versions`` is injectable so a test can drive the mismatch branches
    without installing a second MuJoCo.
    """

    global _noncanonical_warning_printed

    versions = dict(versions if versions is not None else runtime_versions())
    canonical = (
        versions["mujoco_python_version"] == CANONICAL_MUJOCO_VERSION
        and versions["mujoco_native_version"] == CANONICAL_MUJOCO_VERSION
    )
    versions["canonical_environment"] = canonical
    if canonical:
        return versions
    if not allow_noncanonical:
        raise RuntimeError(_mismatch_message(versions, context))
    if not _noncanonical_warning_printed:
        _noncanonical_warning_printed = True
        print(
            "\n".join(
                [
                    "=" * 78,
                    "WARNING: NON-CANONICAL MuJoCo RUNTIME -- FORENSIC REPLAY ONLY",
                    _mismatch_message(versions, context),
                    "Results are recorded with canonical_environment=false and must "
                    "not be compared with, or aggregated into, canonical results.",
                    "=" * 78,
                ]
            ),
            file=sys.stderr,
            flush=True,
        )
    return versions


def _model_state_problems(state: dict) -> list[str]:
    """Reasons this working tree's physics model is not reproducible elsewhere."""

    model_root = SCENE_XML.parent.parent
    problems: list[str] = []
    for path in _RELEVANT_MODEL_PATHS:
        full = SCENE_XML.parent / Path(path).name
        relative = full.relative_to(model_root).as_posix()
        if not _git(model_root, "ls-files", "--", relative):
            problems.append(f"{full.name} is not tracked by git")
            continue
        # The file on disk must be the committed blob, byte for byte: a fresh
        # clone gets the blob, not the working tree.
        committed = _git(model_root, "hash-object", str(full))
        head = _git(model_root, "rev-parse", f"HEAD:{relative}")
        if committed != head:
            problems.append(
                f"{full.name} differs from its committed blob "
                f"(worktree={committed}, HEAD={head})"
            )
    if state["model_files_dirty"]:
        problems.append("the box-catch model files have uncommitted changes")
    if state["model_is_submodule"] and state["submodule_pointer_commit"] != state[
        "model_repo_commit"
    ]:
        problems.append(
            "submodule HEAD does not match the parent's gitlink "
            f"(HEAD={state['model_repo_commit']}, "
            f"pointer={state['submodule_pointer_commit']}) -- commit the "
            "pointer in the parent repository"
        )
    return problems


def validate_model_repository_state(
    *,
    context: str = "",
    allow_dirty_model: bool = False,
    state: Optional[dict] = None,
) -> bool:
    """Return whether the physics model is git-pinned; raise if it is not.

    ``state`` is injectable so a test can simulate a dirty model without
    touching the user's working tree.
    """

    global _dirty_model_warning_printed

    state = state if state is not None else git_state()
    problems = _model_state_problems(state)
    if not problems:
        return True
    message = "\n".join(
        [
            f"physics model is not git-pinned{f' ({context})' if context else ''}:",
            *(f"  - {problem}" for problem in problems),
            f"  scene {SCENE_XML.name} sha256={_digest(SCENE_XML)}",
            f"  robot {ROBOT_XML.name} sha256={_digest(ROBOT_XML)}",
            "",
            "A result produced from an uncommitted model cannot be reproduced "
            "from a fresh clone. Commit the model (and, for a submodule, the "
            "parent's pointer) before a canonical run, or pass "
            "--allow-dirty-model for a debug run whose numbers are not "
            "canonical.",
        ]
    )
    if not allow_dirty_model:
        raise RuntimeError(message)
    if not _dirty_model_warning_printed:
        _dirty_model_warning_printed = True
        print(
            "\n".join(
                [
                    "=" * 78,
                    "WARNING: UNPINNED PHYSICS MODEL -- DEBUG RUN ONLY",
                    message,
                    "Recorded with canonical_model=false.",
                    "=" * 78,
                ]
            ),
            file=sys.stderr,
            flush=True,
        )
    return False


def environment_stamp(
    *,
    allow_noncanonical: bool = False,
    allow_dirty_model: bool = False,
    versions: Optional[dict] = None,
) -> dict:
    """The stamp written into every result.json (validating on the way)."""

    stamp = validate_runtime_environment(
        context="result stamp", allow_noncanonical=allow_noncanonical, versions=versions
    )
    stamp["canonical_model"] = validate_model_repository_state(
        context="result stamp", allow_dirty_model=allow_dirty_model
    )
    return stamp


def log_runtime_environment(context: str, versions: dict) -> None:
    """One stdout line at process start, so a log always names its runtime."""

    print(
        f"[runtime] {context} interpreter={versions['python_executable']} "
        f"conda={versions['conda_environment']} "
        f"mujoco={versions['mujoco_python_version']}/"
        f"{versions['mujoco_native_version']} "
        f"torch={versions['torch_version']} numpy={versions['numpy_version']} "
        f"canonical={versions['canonical_environment']} "
        f"parent={str(versions.get('parent_repo_commit'))[:12]} "
        f"model={str(versions.get('model_repo_commit'))[:12]}",
        flush=True,
    )
