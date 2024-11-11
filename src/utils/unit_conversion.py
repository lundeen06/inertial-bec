import numpy as np
from typing import Union, Dict, Tuple
from dataclasses import dataclass
from ..utils.constants import HBAR, M_RB87, KB

@dataclass
class UnitConverter:
    """
    Class for handling unit conversions between physical and simulation units.
    Provides comprehensive unit conversion for BEC simulations.
    """
    
    # Reference scales
    trap_frequency: float  # Hz
    length_scale: float   # meters
    atom_number: float    # number of atoms
    
    def __post_init__(self):
        """Initialize derived unit scales."""
        # Derived time and energy scales
        self.time_scale = 1.0 / (2 * np.pi * self.trap_frequency)
        self.energy_scale = HBAR * 2 * np.pi * self.trap_frequency
        
        # Derived velocity and momentum scales
        self.velocity_scale = self.length_scale / self.time_scale
        self.momentum_scale = M_RB87 * self.velocity_scale
        
        # Derived acceleration and force scales
        self.acceleration_scale = self.length_scale / (self.time_scale ** 2)
        self.force_scale = M_RB87 * self.acceleration_scale
        
        # Density scale
        self.density_scale = self.atom_number / self.length_scale
        
        # Initialize conversion dictionaries
        self._to_dimensionless = {
            'length': self.length_to_dimensionless,
            'time': self.time_to_dimensionless,
            'energy': self.energy_to_dimensionless,
            'velocity': self.velocity_to_dimensionless,
            'momentum': self.momentum_to_dimensionless,
            'acceleration': self.acceleration_to_dimensionless,
            'force': self.force_to_dimensionless,
            'density': self.density_to_dimensionless,
            'temperature': self.temperature_to_dimensionless,
            'frequency': self.frequency_to_dimensionless
        }
        
        self._to_physical = {
            'length': self.length_to_physical,
            'time': self.time_to_physical,
            'energy': self.energy_to_physical,
            'velocity': self.velocity_to_physical,
            'momentum': self.momentum_to_physical,
            'acceleration': self.acceleration_to_physical,
            'force': self.force_to_physical,
            'density': self.density_to_physical,
            'temperature': self.temperature_to_physical,
            'frequency': self.frequency_to_physical
        }
    
    def to_dimensionless(self, value: Union[float, np.ndarray], 
                        unit_type: str) -> Union[float, np.ndarray]:
        """Convert from physical to dimensionless units."""
        if unit_type not in self._to_dimensionless:
            raise ValueError(f"Unknown unit type: {unit_type}")
        return self._to_dimensionless[unit_type](value)
    
    def to_physical(self, value: Union[float, np.ndarray], 
                   unit_type: str) -> Union[float, np.ndarray]:
        """Convert from dimensionless to physical units."""
        if unit_type not in self._to_physical:
            raise ValueError(f"Unknown unit type: {unit_type}")
        return self._to_physical[unit_type](value)
    
    # Length conversions
    def length_to_dimensionless(self, length: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert length from meters to dimensionless units."""
        return length / self.length_scale
    
    def length_to_physical(self, length_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert length from dimensionless to meters."""
        return length_dimless * self.length_scale
    
    # Time conversions
    def time_to_dimensionless(self, time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert time from seconds to dimensionless units."""
        return time / self.time_scale
    
    def time_to_physical(self, time_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert time from dimensionless to seconds."""
        return time_dimless * self.time_scale
    
    # Energy conversions
    def energy_to_dimensionless(self, energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert energy from Joules to dimensionless units."""
        return energy / self.energy_scale
    
    def energy_to_physical(self, energy_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert energy from dimensionless to Joules."""
        return energy_dimless * self.energy_scale
    
    # Velocity conversions
    def velocity_to_dimensionless(self, velocity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert velocity from m/s to dimensionless units."""
        return velocity / self.velocity_scale
    
    def velocity_to_physical(self, velocity_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert velocity from dimensionless to m/s."""
        return velocity_dimless * self.velocity_scale
    
    # Momentum conversions
    def momentum_to_dimensionless(self, momentum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert momentum from kg⋅m/s to dimensionless units."""
        return momentum / self.momentum_scale
    
    def momentum_to_physical(self, momentum_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert momentum from dimensionless to kg⋅m/s."""
        return momentum_dimless * self.momentum_scale
    
    # Acceleration conversions
    def acceleration_to_dimensionless(self, accel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert acceleration from m/s² to dimensionless units."""
        return accel / self.acceleration_scale
    
    def acceleration_to_physical(self, accel_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert acceleration from dimensionless to m/s²."""
        return accel_dimless * self.acceleration_scale
    
    # Force conversions
    def force_to_dimensionless(self, force: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert force from Newtons to dimensionless units."""
        return force / self.force_scale
    
    def force_to_physical(self, force_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert force from dimensionless to Newtons."""
        return force_dimless * self.force_scale
    
    # Density conversions
    def density_to_dimensionless(self, density: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert density from m^-1 to dimensionless units."""
        return density / self.density_scale
    
    def density_to_physical(self, density_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert density from dimensionless to m^-1."""
        return density_dimless * self.density_scale
    
    # Temperature conversions
    def temperature_to_dimensionless(self, temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert temperature from Kelvin to dimensionless units."""
        return KB * temp / self.energy_scale
    
    def temperature_to_physical(self, temp_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert temperature from dimensionless to Kelvin."""
        return temp_dimless * self.energy_scale / KB
    
    # Frequency conversions
    def frequency_to_dimensionless(self, freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert frequency from Hz to dimensionless units."""
        return freq * self.time_scale
    
    def frequency_to_physical(self, freq_dimless: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert frequency from dimensionless to Hz."""
        return freq_dimless / self.time_scale
    
    def get_scales(self) -> Dict[str, float]:
        """Return dictionary of all scale factors."""
        return {
            'length': self.length_scale,
            'time': self.time_scale,
            'energy': self.energy_scale,
            'velocity': self.velocity_scale,
            'momentum': self.momentum_scale,
            'acceleration': self.acceleration_scale,
            'force': self.force_scale,
            'density': self.density_scale
        }
    
    def dimensionless_parameters(self) -> Dict[str, float]:
        """Return characteristic dimensionless parameters of the system."""
        return {
            'interaction_parameter': self.atom_number * self.length_scale,
            'quantum_parameter': HBAR / (M_RB87 * self.velocity_scale * self.length_scale),
            'trap_parameter': self.trap_frequency * self.time_scale
        }