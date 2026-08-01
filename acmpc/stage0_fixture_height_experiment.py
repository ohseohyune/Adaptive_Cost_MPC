"""Stage 0 fixture-height comparison experiment (not a controller-tuning
change): does raising the box_launch_fixture's initial z (via
AcmpcBoxCatchConfig.stage0_fixture_box_z_offset_m) improve post-release
bilateral-contact retention, independent of the still-open CAPTURE
vertical-tracking gap (see the implementation report -- this experiment
does not claim to fix that, only isolates whether a friendlier initial
geometry helps on its own)?

Reuses the existing STAGE0_DEBUG_TRACE per-step instrumentation
(main_acmpc_box_catch.py) rather than adding new permanent BoxCatchSummary
fields -- this is a one-off comparison, not a metric anyone else needs
tracked forever.

Usage:
    python3 acmpc/stage0_fixture_height_experiment.py
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acmpc.main_acmpc_box_catch import AcmpcBoxCatchConfig, run_box_catch

OFFSETS_M = (0.00, 0.01, 0.02, 0.03, 0.04)

_CAPTURE2_RE = re.compile(
    r"CAPTURE2 t=(?P<t>[\d.]+) phase=(?P<phase>\S+) "
    r"left_contact_active=(?P<left_active>True|False) right_contact_active=(?P<right_active>True|False) "
    r"left_normal_force_n=(?P<left_force>[\d.]+) right_normal_force_n=(?P<right_force>[\d.]+)"
)


@dataclass
class OffsetResult:
    offset_cm: float
    fixture_release_time_s: Optional[float]
    release_left_force_n: float
    release_right_force_n: float
    post_release_bilateral_contact_duration_s: float
    contact_lost: bool
    recontact_occurred: bool
    recontact_time_s: Optional[float]
    recontact_hold_duration_s: float
    capture_entered: bool
    capture_time_s: Optional[float]
    hold_entered: bool
    hold_time_s: Optional[float]
    stable_hold_duration_s: float
    peak_contact_force_n: float
    first_contact_peak_force_n: float
    emergency_occurred: bool
    success: bool
    failure_reason: str


def _run_and_trace(offset_m: float):
    os.environ["STAGE0_DEBUG_TRACE"] = "1"
    cfg = AcmpcBoxCatchConfig(
        seed=7,
        device="cpu",
        online_learning=False,
        use_launch_fixture=True,
        release_fixture_on_bilateral_contact=True,
        stage0_fixture_box_z_offset_m=offset_m,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary = run_box_catch(cfg)
    return summary, buf.getvalue()


def _analyze(offset_m: float, summary, trace_text: str) -> OffsetResult:
    rows = []
    for line in trace_text.splitlines():
        m = _CAPTURE2_RE.match(line)
        if m:
            rows.append(
                (
                    float(m.group("t")),
                    m.group("phase"),
                    m.group("left_active") == "True",
                    m.group("right_active") == "True",
                    float(m.group("left_force")),
                    float(m.group("right_force")),
                )
            )

    release_time = summary.fixture_release_time_s
    peak_force = max((max(r[4], r[5]) for r in rows), default=0.0)

    post_release_rows = [r for r in rows if release_time is not None and r[0] >= release_time]
    post_release_bilateral_duration = 0.0
    contact_lost = False
    loss_time = None
    if post_release_rows:
        start_t = post_release_rows[0][0]
        for t, _phase, left_active, right_active, _lf, _rf in post_release_rows:
            if left_active and right_active:
                post_release_bilateral_duration = t - start_t
            else:
                contact_lost = True
                loss_time = t
                break

    recontact_occurred = False
    recontact_time = None
    recontact_hold_duration = 0.0
    if contact_lost and loss_time is not None:
        after_loss = [r for r in post_release_rows if r[0] > loss_time]
        recontact_start = None
        for t, _phase, left_active, right_active, _lf, _rf in after_loss:
            if left_active and right_active:
                if recontact_start is None:
                    recontact_occurred = True
                    recontact_time = t
                    recontact_start = t
                recontact_hold_duration = t - recontact_start
            else:
                recontact_start = None

    capture_rows = [r for r in rows if r[1] == "capture"]
    hold_rows = [r for r in rows if r[1] == "hold"]

    return OffsetResult(
        offset_cm=offset_m * 100.0,
        fixture_release_time_s=release_time,
        release_left_force_n=summary.fixture_release_left_force_n,
        release_right_force_n=summary.fixture_release_right_force_n,
        post_release_bilateral_contact_duration_s=post_release_bilateral_duration,
        contact_lost=contact_lost,
        recontact_occurred=recontact_occurred,
        recontact_time_s=recontact_time,
        recontact_hold_duration_s=recontact_hold_duration,
        capture_entered=bool(capture_rows),
        capture_time_s=capture_rows[0][0] if capture_rows else None,
        hold_entered=bool(hold_rows),
        hold_time_s=hold_rows[0][0] if hold_rows else None,
        stable_hold_duration_s=summary.hold_time_s,
        peak_contact_force_n=peak_force,
        first_contact_peak_force_n=summary.first_contact_peak_force_n,
        emergency_occurred="emergency" in summary.failure_reason.lower(),
        success=summary.success,
        failure_reason=summary.failure_reason,
    )


def _classify(result: OffsetResult, baseline: OffsetResult) -> str:
    if result.peak_contact_force_n > 30.0 or result.emergency_occurred:
        return "UNSAFE"
    if result.hold_entered or result.stable_hold_duration_s > 0.0:
        return "HOLD_IMPROVED"
    if result.recontact_occurred and not baseline.recontact_occurred:
        return "RECONTACT_IMPROVED"
    if result.post_release_bilateral_contact_duration_s > baseline.post_release_bilateral_contact_duration_s + 0.01:
        return "CONTACT_DELAY_ONLY"
    return "NO_EFFECT"


def run_experiment() -> list[OffsetResult]:
    results = []
    for offset_m in OFFSETS_M:
        summary, trace_text = _run_and_trace(offset_m)
        results.append(_analyze(offset_m, summary, trace_text))
    return results


def print_report(results: list[OffsetResult]) -> None:
    baseline = results[0]
    header = (
        f"{'offset_cm':>9} {'release_t':>9} {'L_N':>6} {'R_N':>6} "
        f"{'post_dur_s':>10} {'lost':>5} {'recontact':>9} {'recontact_t':>11} "
        f"{'capture':>7} {'hold':>5} {'hold_dur':>8} {'peak_N':>7} {'first_peak_N':>12} "
        f"{'emerg':>6} {'success':>7} classification"
    )
    print(header)
    for r in results:
        classification = _classify(r, baseline)
        print(
            f"{r.offset_cm:9.1f} {('' if r.fixture_release_time_s is None else f'{r.fixture_release_time_s:.3f}'):>9} "
            f"{r.release_left_force_n:6.2f} {r.release_right_force_n:6.2f} "
            f"{r.post_release_bilateral_contact_duration_s:10.3f} {str(r.contact_lost):>5} "
            f"{str(r.recontact_occurred):>9} "
            f"{('' if r.recontact_time_s is None else f'{r.recontact_time_s:.3f}'):>11} "
            f"{str(r.capture_entered):>7} {str(r.hold_entered):>5} "
            f"{r.stable_hold_duration_s:8.3f} {r.peak_contact_force_n:7.2f} "
            f"{r.first_contact_peak_force_n:12.2f} {str(r.emergency_occurred):>6} "
            f"{str(r.success):>7} {classification}  ({r.failure_reason})"
        )


if __name__ == "__main__":
    print_report(run_experiment())
