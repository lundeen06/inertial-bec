"""
Physics modules for BEC simulation including Hamiltonians, interactions,
and trap potentials.
"""

from .hamiltonian import Hamiltonian
from .interactions import InteractionModel
from .acceleration import AccelerationSensor
from .trap_potential import TrapPotential

__all__ = [
    "Hamiltonian",
    "InteractionModel",
    "AccelerationSensor",
    "TrapPotential"
]