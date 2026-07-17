"""NeuralCostMap: learned context → (w_pos, w_force, w_grasp).

Architecture (AC-MPC paper family)
------------------------------------
    ξ (13D) → [Linear(64)+ReLU] → [Linear(64)+ReLU] → [Linear(3)+Sigmoid×W_MAX]

Context vector ξ (INPUT_DIM = 13)
-----------------------------------
    [0:3]   x_EE − x_object      relative EE position [m]
    [3:6]   ẋ_object             object velocity [m/s]
    [6]     f_contact / 50.0     normalised total contact force [0, 1]
    [7]     min(ttc, 5) / 5.0    normalised time-to-contact [0, 1]
    [8:13]  GraspState one-hot   (Step 2 supervised; zero in Step 3 RL)

Step 2: supervised pre-training from COST_WEIGHTS lookup table.
    X, Y = generate_supervised_data()
    ncm  = NeuralCostMap()
    ncm.train(X, Y)

Step 3 (future): replace one-hot with learned embedding, train end-to-end
    with PPO through cvxpylayers (differentiable MPC).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from control.clik.grasp_state_machine import GraspState
from control.mpc.adaptive_cost_mpc import COST_WEIGHTS

# ── constants ──────────────────────────────────────────────────────────────────

W_MAX      = 10.0
INPUT_DIM  = 13
HIDDEN_DIM = 64
OUTPUT_DIM = 3        # (w_pos, w_force, w_grasp)

_STATES = list(GraspState)   # fixed ordering for one-hot encoding


# ── context builder ────────────────────────────────────────────────────────────

def build_context(
    x_ee:        np.ndarray,
    x_object:    np.ndarray,
    xdot_object: np.ndarray,
    f_contact:   float,
    ttc:         float,
    grasp_state: Optional[GraspState] = None,
) -> np.ndarray:
    """Build normalised 13D context vector ξ for NeuralCostMap.

    Args:
        x_ee:        EE position [m]
        x_object:    object position [m]
        xdot_object: object velocity [m/s]
        f_contact:   total contact force magnitude [N]
        ttc:         time-to-contact [s]  (np.inf → normalised as 1.0)
        grasp_state: set one-hot entry for Step 2; pass None in Step 3.

    Returns:
        ξ ∈ R^13, dtype=float64.
    """
    x_rel  = np.asarray(x_ee, float) - np.asarray(x_object, float)       # (3,)
    v_obj  = np.asarray(xdot_object, float)                               # (3,)
    f_norm = np.array([np.clip(float(f_contact), 0.0, 50.0) / 50.0])     # (1,)
    ttc_f  = float(ttc) if np.isfinite(float(ttc)) else 5.0
    ttc_n  = np.array([np.clip(ttc_f, 0.0, 5.0) / 5.0])                  # (1,)
    oh     = np.zeros(len(_STATES))
    if grasp_state is not None:
        oh[_STATES.index(grasp_state)] = 1.0                               # (5,)
    return np.concatenate([x_rel, v_obj, f_norm, ttc_n, oh])              # (13,)


# ── PyTorch network ────────────────────────────────────────────────────────────

class _CostMapNet(nn.Module):
    def __init__(
        self,
        input_dim:  int   = INPUT_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        w_max:      float = W_MAX,
    ) -> None:
        super().__init__()
        self.w_max = float(w_max)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, OUTPUT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.w_max


# ── public class ───────────────────────────────────────────────────────────────

class NeuralCostMap:
    """Learned cost-weight map: ξ (13D) → (w_pos, w_force, w_grasp) ∈ [0, W_MAX]³.

    Replaces the COST_WEIGHTS lookup table inside CartesianMPC.
    NumPy arrays in / out; PyTorch handles the forward pass and training.

    Usage::

        ncm = NeuralCostMap()
        X, Y = generate_supervised_data()
        ncm.train(X, Y, n_epochs=1000)

        ctx = build_context(x_ee, x_obj, xdot_obj, f_con, ttc, grasp_state)
        w   = ncm.predict(ctx)   # np.ndarray (3,): [w_pos, w_force, w_grasp]

        ncm.save("cost_map.pt")
        ncm.load("cost_map.pt")
    """

    def __init__(
        self,
        input_dim:  int   = INPUT_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        w_max:      float = W_MAX,
    ) -> None:
        self._train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._infer_device = torch.device("cpu")   # always infer on CPU (faster for batch=1)
        self._net  = _CostMapNet(input_dim, hidden_dim, w_max).to(self._infer_device)
        self._net.eval()
        self.w_max = float(w_max)

    # ── inference ──────────────────────────────────────────────────────────────

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Forward pass.  Input (INPUT_DIM,) or (B, INPUT_DIM) → (3,) or (B, 3)."""
        single = context.ndim == 1
        x = torch.tensor(np.atleast_2d(np.asarray(context, float)),
                          dtype=torch.float32)
        self._net.eval()
        with torch.no_grad():
            out = self._net(x).numpy()
        return out[0] if single else out

    def weights_for_state(self, grasp_state: GraspState) -> dict[str, float]:
        """Predict weights for a state using a neutral zero context."""
        ctx = build_context(
            x_ee=np.zeros(3), x_object=np.zeros(3),
            xdot_object=np.zeros(3), f_contact=0.0,
            ttc=float("inf"), grasp_state=grasp_state,
        )
        w = self.predict(ctx)
        return {"pos": float(w[0]), "force": float(w[1]), "grasp": float(w[2])}

    # ── training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_epochs:    int   = 1000,
        batch_size:  int   = 256,
        lr:          float = 2e-3,
        print_every: int   = 0,
    ) -> list[float]:
        """Adam mini-batch supervised training.

        Trains on self._train_device (GPU if available) for speed, then moves
        the weights back to self._infer_device (CPU) for real-time inference.

        Args:
            X:           (N, INPUT_DIM) context samples
            Y:           (N, 3)         target weights (from COST_WEIGHTS)
            n_epochs:    training epochs
            batch_size:  mini-batch size (256 to amortise GPU kernel-launch cost)
            lr:          Adam learning rate
            print_every: print loss every this many epochs (0 = silent)

        Returns:
            Per-epoch average MSE loss list.
        """
        dev = self._train_device
        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=dev)
        N   = len(X_t)

        net = self._net.to(dev)
        net.train()
        opt     = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        rng     = torch.Generator(device=dev if dev.type == "cuda" else "cpu")
        losses: list[float] = []

        for epoch in range(n_epochs):
            perm       = torch.randperm(N, generator=rng,
                                        device=dev if dev.type == "cuda" else "cpu")
            epoch_loss = 0.0
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X_t[idx], Y_t[idx]
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()) * len(idx)
            avg = epoch_loss / N
            losses.append(avg)
            if print_every > 0 and (epoch + 1) % print_every == 0:
                print(f"  epoch {epoch+1:4d}/{n_epochs}  loss={avg:.5f}")

        # move weights back to CPU for real-time inference
        self._net = net.to(self._infer_device)
        self._net.eval()
        return losses

    # ── persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        torch.save(self._net.state_dict(), str(path))

    def load(self, path: str | Path) -> None:
        self._net.load_state_dict(
            torch.load(str(path), map_location="cpu", weights_only=True)
        )
        self._net.eval()


# ── supervised data generation ─────────────────────────────────────────────────

def generate_supervised_data(
    n_per_state: int = 500,
    seed:        int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, Y) supervised pairs from the COST_WEIGHTS schedule.

    Each state contributes n_per_state context samples with randomised
    continuous features but the correct GraspState one-hot.  Targets are
    the corresponding (w_pos, w_force, w_grasp) from COST_WEIGHTS.

    Args:
        n_per_state: samples per GraspState
        seed:        RNG seed for reproducibility

    Returns:
        X: (5*n_per_state, INPUT_DIM)   context vectors
        Y: (5*n_per_state, 3)           target weight triplets
    """
    rng = np.random.default_rng(seed)
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []

    for state in GraspState:
        cw = COST_WEIGHTS[state]
        y  = np.array([cw["pos"], cw["force"], cw["grasp"]], dtype=float)
        for _ in range(n_per_state):
            x_rel   = rng.uniform(-0.3, 0.3, 3)
            v_obj   = rng.uniform(-1.5, 1.5, 3)
            f_con   = rng.uniform(0.0, 30.0)
            ttc_val = rng.uniform(0.0, 5.0)
            ctx = build_context(
                x_ee=x_rel, x_object=np.zeros(3),
                xdot_object=v_obj, f_contact=f_con,
                ttc=ttc_val, grasp_state=state,
            )
            X_list.append(ctx)
            Y_list.append(y)

    return np.array(X_list, dtype=float), np.array(Y_list, dtype=float)


# ── convenience factory ────────────────────────────────────────────────────────

def pretrained_neural_cost_map(
    n_per_state: int   = 200,
    n_epochs:    int   = 300,
    lr:          float = 2e-3,
    print_every: int   = 0,
) -> NeuralCostMap:
    """Train a NeuralCostMap supervised on COST_WEIGHTS and return it.

    Drop-in replacement for the COST_WEIGHTS lookup in CartesianMPC.
    Ready to use immediately; can be fine-tuned further with RL in Step 3.
    """
    ncm  = NeuralCostMap()
    X, Y = generate_supervised_data(n_per_state=n_per_state)
    ncm.train(X, Y, n_epochs=n_epochs, lr=lr, print_every=print_every)
    return ncm
