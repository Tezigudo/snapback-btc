"""Back-compat shim. Real implementation lives in `signals_carry_unified`."""
from strategy.signals_carry_unified import CarryHarvesterUnifiedV3 as CarryHarvesterV3

__all__ = ["CarryHarvesterV3"]
