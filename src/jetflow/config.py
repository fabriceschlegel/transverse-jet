from dataclasses import asdict, dataclass
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class Config:
    # x streamwise, y wall normal, z spanwise. D = U_inf = rho = 1.
    h: float = 0.25
    x_min: float = -3.0
    x_max: float = 7.0
    pipe_depth: float = 2.0
    height: float = 4.0
    half_width: float = 2.0
    velocity_ratio: float = 2.0
    re_jet: float = 100.0
    schmidt: float = 1.0
    delta99: float = 0.75
    cfl: float = 0.45
    dt_max: float = 0.02
    end_time: float = 4.0
    output_interval: float = 0.5
    sample_interval: float = 0.1
    average_start: float = 2.0
    pressure_rtol: float = 1e-10
    pressure_maxiter: int = 1000
    momentum_advection: str = "centered"
    growth_guard_window: float = 0.1
    max_velocity_growth_factor: float = 3.0
    max_energy_growth_factor: float = 1.5

    @property
    def nu(self):
        return self.velocity_ratio / self.re_jet

    @property
    def kappa(self):
        return self.nu / self.schmidt

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        for key in ("h", "pipe_depth", "height", "half_width", "velocity_ratio",
                    "re_jet", "schmidt", "delta99", "cfl", "dt_max", "end_time",
                    "output_interval", "sample_interval", "pressure_rtol",
                    "growth_guard_window"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.momentum_advection not in {"centered", "skew-symmetric"}:
            raise ValueError("momentum_advection must be 'centered' or 'skew-symmetric'")
        if self.max_velocity_growth_factor <= 1 or self.max_energy_growth_factor <= 1:
            raise ValueError("rapid-growth factors must exceed one")
        if self.x_min >= -0.5 or self.x_max <= 1.0 or self.half_width <= 0.5:
            raise ValueError("Domain must surround the diameter-one nozzle")
        if self.h > 0.25:
            raise ValueError("Use at least 4 cells per diameter, even for a smoke test")
        if not 0 < self.cfl <= 0.5:
            raise ValueError("cfl must be in (0, 0.5]")
        if self.average_start < 0 or self.average_start >= self.end_time:
            raise ValueError("average_start must lie in [0, end_time)")
        if self.pressure_maxiter < 1:
            raise ValueError("pressure_maxiter must be positive")
        for name, length in (("x_min", -self.x_min), ("x_max", self.x_max),
                             ("pipe_depth", self.pipe_depth), ("height", self.height),
                             ("half_width", self.half_width)):
            if abs(length / self.h - round(length / self.h)) > 1e-8:
                raise ValueError(f"{name} must be an integer multiple of h")
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()
