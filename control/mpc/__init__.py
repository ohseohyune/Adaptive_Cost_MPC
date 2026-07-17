"""Model-predictive controllers and adaptive cost-learning components."""

from control.mpc.online_actor_critic import (
    ACMPCAction,
    BimanualPhase,
    DifferentiableBimanualMPC,
    DifferentiableMPCConfig,
    OnlineActorCriticACMPC,
    OnlineActorCriticConfig,
    OnlineUpdate,
    build_bimanual_observation,
    resolve_device,
)
from control.mpc.ppo_cost_adapter import (
    COST_PARAMETER_NAMES,
    GENERALIZATION_OBS_DIM,
    CostAdaptationAction,
    PPOCostAdapter,
    PPOCostConfig,
    PPORolloutBuffer,
    PPOUpdateSummary,
    apply_cost_action,
    build_generalization_observation,
    generalized_advantage_estimate,
)

__all__ = [
    "ACMPCAction",
    "BimanualPhase",
    "COST_PARAMETER_NAMES",
    "CostAdaptationAction",
    "DifferentiableBimanualMPC",
    "DifferentiableMPCConfig",
    "OnlineActorCriticACMPC",
    "OnlineActorCriticConfig",
    "OnlineUpdate",
    "GENERALIZATION_OBS_DIM",
    "PPOCostAdapter",
    "PPOCostConfig",
    "PPORolloutBuffer",
    "PPOUpdateSummary",
    "apply_cost_action",
    "build_bimanual_observation",
    "build_generalization_observation",
    "generalized_advantage_estimate",
    "resolve_device",
]
