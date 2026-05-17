"""Back-compat shim. Real implementation lives in `signals_carry_unified`."""
from strategy.signals_carry_unified import CarryHarvesterUnifiedV2 as CarryHarvesterV2

__all__ = ["CarryHarvesterV2"]
