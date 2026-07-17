"""Grasp State Machine for contact-based gripper control.

States
------
APPROACH      arm moving toward object (free-space impedance, high K)
PRE_CONTACT   EE within pre_contact_dist — stiffness reduced to avoid impact spikes
GRASPING      gripper closing, contact detected (F_normal > contact_threshold)
GRASPED       stable grasp — F_normal > grasp_threshold for stable_time seconds
MANIPULATION  in-hand manipulation (entered via transition_to_manipulation())

Each state carries a recommended Cartesian translational stiffness K_pos so the
impedance controller can query self.K_pos without additional book-keeping.

Transitions
-----------
APPROACH     → PRE_CONTACT  : dist(EE, object) < pre_contact_dist
PRE_CONTACT  → GRASPING     : total_normal_force > contact_threshold
GRASPING     → GRASPED      : total_normal_force > grasp_threshold for stable_time s
                               (timer resets if force drops below threshold)
GRASPED      → MANIPULATION : explicit call to transition_to_manipulation()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from control.clik.contact import ContactInfo, total_normal_force


class GraspState(Enum):
    APPROACH     = "approach"
    PRE_CONTACT  = "pre_contact"
    GRASPING     = "grasping"
    GRASPED      = "grasped"
    MANIPULATION = "manipulation"


@dataclass
class GraspSMConfig:
    """Thresholds and stiffness values for the grasp state machine."""

    pre_contact_dist: float  = 0.05   # [m]  APPROACH → PRE_CONTACT
    contact_threshold: float = 0.5    # [N]  PRE_CONTACT → GRASPING
    grasp_threshold: float   = 3.0    # [N]  force required for stable grasp
    stable_time: float       = 0.5    # [s]  force must exceed grasp_threshold
    mu: float                = 0.5    # friction coefficient (for cone check)

    # Cartesian translational stiffness per state [N/m]
    K_pos_approach:    float = 500.0
    K_pos_pre_contact: float = 200.0
    K_pos_grasping:    float = 100.0
    K_pos_grasped:     float = 500.0


@dataclass
class StateTransition:
    time: float
    from_state: GraspState
    to_state: GraspState


class GraspStateMachine:
    """Event-driven grasp state machine.

    Call update() once per simulation step with the current EE position,
    object position, active contacts, and timestep dt.
    """

    def __init__(self, config: GraspSMConfig | None = None) -> None:
        self.config = config or GraspSMConfig()
        self.state  = GraspState.APPROACH
        self._stable_timer: float = 0.0
        self.history: list[StateTransition] = []

    # ── public properties ────────────────────────────────────────────────────────

    @property
    def K_pos(self) -> float:
        """Recommended Cartesian translational stiffness for the current state."""
        return {
            GraspState.APPROACH:     self.config.K_pos_approach,
            GraspState.PRE_CONTACT:  self.config.K_pos_pre_contact,
            GraspState.GRASPING:     self.config.K_pos_grasping,
            GraspState.GRASPED:      self.config.K_pos_grasped,
            GraspState.MANIPULATION: self.config.K_pos_grasped,
        }[self.state]

    # ── update ──────────────────────────────────────────────────────────────────

    def update(
        self,
        ee_pos: np.ndarray,
        object_pos: np.ndarray,
        contacts: list[ContactInfo],
        dt: float,
        t: float = 0.0,
    ) -> tuple[GraspState, bool]:
        """Advance the state machine by one timestep.

        Args:
            ee_pos:     EE position in world frame [3]
            object_pos: object centre in world frame [3]
            contacts:   ContactInfo list from get_contact_forces()
            dt:         simulation timestep [s]
            t:          current simulation time [s] (used in transition log)

        Returns:
            (current_state, state_changed_this_step)
        """
        old   = self.state
        dist  = float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(object_pos)))
        f_tot = total_normal_force(contacts)

        if self.state is GraspState.APPROACH:
            if dist < self.config.pre_contact_dist:
                self._transition(GraspState.PRE_CONTACT, t)

        elif self.state is GraspState.PRE_CONTACT:
            if f_tot > self.config.contact_threshold:
                self._transition(GraspState.GRASPING, t)

        elif self.state is GraspState.GRASPING:
            if f_tot > self.config.grasp_threshold:
                self._stable_timer += dt
                if self._stable_timer >= self.config.stable_time:
                    self._transition(GraspState.GRASPED, t)
            else:
                self._stable_timer = 0.0

        # GRASPED → MANIPULATION requires an explicit external call

        return self.state, self.state is not old

    # ── external triggers ────────────────────────────────────────────────────────

    def transition_to_manipulation(self, t: float = 0.0) -> bool:
        """Trigger GRASPED → MANIPULATION.

        Returns True if the transition was performed, False if already in
        MANIPULATION or not yet in GRASPED.
        """
        if self.state is GraspState.GRASPED:
            self._transition(GraspState.MANIPULATION, t)
            return True
        return False

    def reset(self) -> None:
        """Revert to APPROACH state and clear history."""
        self.state = GraspState.APPROACH
        self._stable_timer = 0.0
        self.history.clear()

    # ── private ─────────────────────────────────────────────────────────────────

    def _transition(self, new_state: GraspState, t: float) -> None:
        self.history.append(StateTransition(t, self.state, new_state))
        self.state = new_state
        self._stable_timer = 0.0
