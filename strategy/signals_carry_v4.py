"""Back-compat shim. Real implementation lives in `signals_carry_unified`."""
from strategy.signals_carry_unified import CarryHarvesterUnifiedV4 as CarryHarvesterV4

__all__ = ["CarryHarvesterV4"]
