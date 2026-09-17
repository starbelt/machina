"""
machina.agents — Layer 3 agent types.

Public API
----------
AgentType             -- abstract base class for all agent types
QuantityDeclaration   -- dataclass describing a quantity an agent declares
ConstraintDeclaration -- dataclass describing a constraint an agent returns
"""

from .agent_type import AgentType, ConstraintDeclaration, QuantityDeclaration
from .single_sat_coverage import SingleSatCoverage

__all__ = ['AgentType', 'QuantityDeclaration', 'ConstraintDeclaration',
           'SingleSatCoverage']
