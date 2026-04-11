"""Middleware configuration. All behavior controlled via environment variables."""

import os
from dataclasses import dataclass, field


@dataclass
class MiddlewareConfig:
    # Master switch: ENABLE_MIDDLEWARE=0 (default, silent) or 1 (active)
    enabled: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_MIDDLEWARE", "0") == "1"
    )

    # Per-deficiency toggles: MIDDLEWARE_DEFICIENCIES=B1,B2,B3,B5,M1,M2,M3,M4
    active_deficiencies: set[str] = field(
        default_factory=lambda: set(
            os.environ.get(
                "MIDDLEWARE_DEFICIENCIES", "B1,B2,B3,B5,M1,M2,M3,M4"
            ).split(",")
        )
    )

    # Process detection parameters
    # Checkpoints: query counts at which to run mid-process checks.
    # Default [37, 44] based on precision analysis of correct/wrong case distributions.
    # Override via MW_CHECK_POINTS=37,44 (comma-separated)
    check_points: list[int] = field(
        default_factory=lambda: [
            int(x) for x in os.environ.get("MW_CHECK_POINTS", "37,44").split(",")
        ]
    )
    max_interventions: int = int(os.environ.get("MW_MAX_INTERVENTIONS", "3"))

    # Conclusion detection parameters
    check_before_conclusion: bool = True
    max_conclusion_interventions: int = int(
        os.environ.get("MW_MAX_CONCLUSION", "1")
    )

    # Dedup: same dimension intervened at most N times
    max_per_dimension: int = int(os.environ.get("MW_MAX_PER_DIM", "1"))

    @property
    def process_deficiencies(self) -> set[str]:
        return self.active_deficiencies & {"B1", "B2", "B3", "B5", "M1", "M2"}

    @property
    def conclusion_deficiencies(self) -> set[str]:
        return self.active_deficiencies & {"M1", "M2", "M3", "M4"}
