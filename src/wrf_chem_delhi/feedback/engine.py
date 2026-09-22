"""Aerosol-meteorology feedback engine.

Implements the physical feedback loop:
  PM2.5 (ug/m3)
      ↓
  aerosol optical depth
      ↓
  shortwave radiation reduction
      ↓
  surface heating reduction
      ↓
  PBL suppression
      ↓
  pollution trapping → higher PM2.5

Applied after the chemistry step and BEFORE the transport+chemistry
computation for the NEXT hour: this is what makes the system *coupled*.
A positive feedback: more PM → lower PBL → higher PM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Sensible parameter constants for the reduced-order feedback model
K_AOD = 0.23           # ug/m3 -> AOD
K_SW_REDUCTION = 0.30  # AOD -> SW fraction loss
K_PBL_SUSPENSION = 0.25 # SW loss -> PBL height reduction factor (max)
PM_BASELINE_UG = 80.0   # reference PM2.5 where feedback starts to bite
RETENTION_MAX = 0.35    # max concentration retention from PBL suppression


@dataclass
class FeedbackResult:
    pblh_adjusted: np.ndarray  # [nt,ny,nx] new PBL height after feedback
    pm25_coupled: np.ndarray   # [nt,ny,nx] PM2.5 after coupled feedback in the same hour
    feedback_strength: np.ndarray  # [nt,ny,nx] dimensionless 0..1
    description: str = "PM25->AOD->SW-loss->PBL-suppression->PM25 trapping"


def compute_feedback(
    pm25_ug: np.ndarray,
    pblh_base: np.ndarray,
    pbl_min: float = 150.0,
) -> FeedbackResult:
    """Apply the aerosol-radiation-PBL feedback to a *single* PM2.5 field.

    Returns the adjusted PBL and a "coupled" PM2.5 for the same hour. The
    coupled PM2.5 is: PM2.5 * (1 + excess / baseline) where excess arises
    from the additional surface concentration kept when PBL is suppressed.

    Parameters
    ----------
    pm25_ug : [ny,nx] current PM2.5 in µg/m³
    pblh_base : [ny,nx] base (met-only) PBL height in m
    pbl_min : minimum PBL height floor after feedback (m)

    This function is called PER TIME STEP to form the coupled loop.
    """
    # 1. Aerosol optical depth (approximate from PM)
    aod = K_AOD * pm25_ug / 1000.0  # typical summer AOD ~ 0.1..2.0
    # 2. Shortwave reduction
    sw_loss = K_SW_REDUCTION * np.clip(aod, 0, 5)
    # 3. PBL suppression factor (dimensionless, 0..1)
    pbl_factor = 1.0 - K_PBL_SUSPENSION * np.clip(sw_loss, 0, 1)
    pbl_adj = np.maximum(pblh_base * pbl_factor, pbl_min)
    # 4. Excess concentration (more surface-bound than PBL)
    excess_ratio = np.clip(pblh_base / pbl_adj - 1.0, 0, 2.0)
    pm25_coupled = pm25_ug * (1.0 + RETENTION_MAX * np.clip(excess_ratio, 0, 1))
    feedback_strength = np.clip(K_PBL_SUSPENSION * sw_loss, 0, 1)

    return FeedbackResult(
        pblh_adjusted=pbl_adj,
        pm25_coupled=pm25_coupled,
        feedback_strength=feedback_strength,
    )


def apply_feedback_sequence(
    pm25: np.ndarray, pblh: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full sequence across a time series.

    Parameters
    ----------
    pm25  [nt,ny,nx]
    pblh  [nt,ny,nx]

    Returns (pblh_adjusted, pm25_coupled, feedback_strength) all [nt,ny,nx].
    """
    nt = pm25.shape[0]
    pbl_out = pblh.copy()
    pm_out = pm25.copy()
    fb_out = np.zeros_like(pm25)
    for t in range(nt):
        fb = compute_feedback(pm25[t], pblh[t])
        pbl_out[t] = fb.pblh_adjusted
        pm_out[t] = fb.pm25_coupled
        fb_out[t] = fb.feedback_strength
    return pbl_out, pm_out, fb_out