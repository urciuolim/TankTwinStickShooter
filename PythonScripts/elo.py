# Shim (M0 A6): the ELO math moved into the installable package `tank_twin.elo`.
# This module re-exports it so 2021 importers (`from elo import elo_change`,
# `from elo import elo_prob`) keep working byte-for-byte. Behavior is identical;
# the implementation now lives in src/tank_twin/elo.py. Migrate callers to
# `from tank_twin.elo import ...` over time, then delete this shim.
from tank_twin.elo import elo_change, elo_prob

__all__ = ["elo_change", "elo_prob"]
