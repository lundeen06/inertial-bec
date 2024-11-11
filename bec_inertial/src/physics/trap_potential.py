import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
from ..utils.constants import to_dimensionless, to_physical

@dataclass
class TrapPotential:
    """Class representing the trapping potential for the BEC."""
    
    # Trap parameters
    frequency: float  # Trap frequency in Hz
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Trap center in meters
    anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Relative frequencies
    time_dependence: Optional[Callable[[float], float]] = None
    
    def __post_init__(self):
        """Convert physical parameters to simulation units."""
        self.center_dimless = tuple(
            to_dimensionless(x, 'length', self.frequency) for x in self.center
        )
    
    def harmonic_potential(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Compute harmonic trapping potential.
        x: position grid (dimensionless)
        t: time (seconds)
        Returns: potential in dimensionless units
        """
        # Center coordinates
        x_centered = x - self.center_dimless[0]
        
        # Basic harmonic potential
        V = 0.5 * self.anisotropy[0] * x_centered**2
        
        # Apply time dependence if specified
        if self.time_dependence is not None:
            V *= self.time_dependence(t)
        
        return V
    
    def add_noise(self, V: np.ndarray, amplitude: float = 0.01) -> np.ndarray:
        """Add technical noise to potential."""
        noise = amplitude * np.random.randn(*V.shape)
        return V * (1 + noise)
    
    def add_roughness(self, V: np.ndarray, scale: float = 0.1, 
                     correlation_length: float = 0.1) -> np.ndarray:
        """Add spatial roughness to potential."""
        x = np.arange(len(V))
        roughness = scale * np.exp(-0.5 * (np.subtract.outer(x, x)/correlation_length)**2)
        noise = np.random.multivariate_normal(np.zeros_like(x), roughness)
        return V + noise
    
    def time_varying_frequency(self, t: float, 
                             modulation_depth: float = 0.1,
                             modulation_frequency: float = 1.0) -> float:
        """Time-dependent modulation of trap frequency."""
        return 1.0 + modulation_depth * np.sin(2 * np.pi * modulation_frequency * t)
    
    def calculate_trap_depth(self, x: np.ndarray) -> float:
        """Calculate the effective trap depth."""
        V = self.harmonic_potential(x)
        return np.max(V) - np.min(V)
    
    def calculate_characteristic_length(self) -> float:
        """Calculate characteristic length scale of the trap."""
        return np.sqrt(to_physical(1.0, 'length', self.frequency))
    
    def calculate_trapping_frequencies(self) -> Tuple[float, float, float]:
        """Get trapping frequencies in all directions."""
        return tuple(self.frequency * a for a in self.anisotropy)
    
    def get_potential_with_tilt(self, x: np.ndarray, 
                              acceleration: float) -> np.ndarray:
        """
        Get trap potential with added linear gradient from acceleration.
        acceleration: acceleration in m/s²
        """
        acc_dimless = to_dimensionless(acceleration, 'acceleration', self.frequency)
        V_harmonic = self.harmonic_potential(x)
        V_tilt = -acc_dimless * x
        return V_harmonic + V_tilt
    
    def get_turning_points(self, x: np.ndarray, energy: float) -> Tuple[float, float]:
        """Calculate classical turning points for given energy."""
        V = self.harmonic_potential(x)
        # Find points where V(x) = E
        idx_left = np.argmin(np.abs(V[:len(V)//2] - energy))
        idx_right = len(V)//2 + np.argmin(np.abs(V[len(V)//2:] - energy))
        return x[idx_left], x[idx_right]
    
    def is_stable(self, x: np.ndarray, 
                 acceleration: float, 
                 threshold: float = 1.0) -> bool:
        """
        Check if trap is stable under given acceleration.
        threshold: maximum allowed tilt relative to trap depth
        """
        V_total = self.get_potential_with_tilt(x, acceleration)
        trap_depth = self.calculate_trap_depth(x)
        tilt = np.max(np.abs(np.gradient(V_total)))
        return tilt < threshold * trap_depth