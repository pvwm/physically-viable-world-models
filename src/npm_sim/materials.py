from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialPreset:
    density: float
    friction: float
    restitution: float


MATERIALS: dict[str, MaterialPreset] = {
    "rubber": MaterialPreset(density=1100.0, friction=1.10, restitution=0.20),
    "wood": MaterialPreset(density=700.0, friction=0.70, restitution=0.05),
    "steel": MaterialPreset(density=7850.0, friction=0.45, restitution=0.05),
}
