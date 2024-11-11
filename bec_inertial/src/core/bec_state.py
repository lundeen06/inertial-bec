import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from ..utils.constants import (
    SPATIAL_POINTS, DEFAULT_TRAP_FREQUENCY,
    DEFAULT_INTERACTION_STRENGTH, DEFAULT_ATOM_NUMBER,
    to_dimensionless, to_physical
)

@dataclass
class BECState:
    """Class representing the quantum state of a Bose-Einstein condensate."""
    
    # Grid parameters
    grid_points: int = SPATIAL_POINTS
    box_size: float = 20.0  # in units of characteristic length
    
    # Physical parameters
    trap_frequency: float = DEFAULT_TRAP_FREQUENCY
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH
    atom_number: float = DEFAULT_ATOM_NUMBER
    
    # State storage
    _wavefunction: Optional[np.ndarray] = None
    _grid: Optional[np.ndarray] = None
    _k_grid: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize grids and default wavefunction if not provided."""
        # Create spatial grid
        self._grid = np.linspace(-self.box_size/2, self.box_size/2, self.grid_points)
        self.dx = self._grid[1] - self._grid[0]
        
        # Create momentum grid
        self._k_grid = 2 * np.pi * np.fft.fftfreq(self.grid_points, self.dx)
        
        # Initialize ground state if wavefunction not provided
        if self._wavefunction is None:
            self._initialize_ground_state()
    
    def _initialize_ground_state(self):
        """Initialize the ground state wavefunction (Gaussian approximation)."""
        # Harmonic oscillator ground state
        x = self._grid
        sigma = 1.0  # Ground state width in harmonic units
        psi = (1/(np.pi*sigma**2))**(1/4) * np.exp(-x**2/(2*sigma**2))
        
        # Normalize
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        
        # Scale for correct atom number
        self._wavefunction = np.sqrt(self.atom_number) * psi
    
    @property
    def wavefunction(self) -> np.ndarray:
        """Get the current wavefunction."""
        return self._wavefunction
    
    @wavefunction.setter
    def wavefunction(self, psi: np.ndarray):
        """Set the wavefunction with automatic normalization."""
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        self._wavefunction = psi * np.sqrt(self.atom_number) / norm
    
    @property
    def density(self) -> np.ndarray:
        """Get the particle density."""
        return np.abs(self._wavefunction)**2
    
    @property
    def phase(self) -> np.ndarray:
        """Get the phase of the wavefunction."""
        return np.angle(self._wavefunction)
    
    def momentum_space_wavefunction(self) -> np.ndarray:
        """Get the wavefunction in momentum space."""
        return np.fft.fft(self._wavefunction) * self.dx / np.sqrt(2*np.pi)
    
    def kinetic_energy(self) -> float:
        """Calculate the kinetic energy of the system."""
        psi_k = self.momentum_space_wavefunction()
        return np.sum(0.5 * self._k_grid**2 * np.abs(psi_k)**2) * 2*np.pi/self.box_size
    
    def potential_energy(self, V: np.ndarray) -> float:
        """Calculate the potential energy for a given potential."""
        return np.sum(V * self.density) * self.dx
    
    def interaction_energy(self) -> float:
        """Calculate the interaction energy."""
        return 0.5 * self.interaction_strength * np.sum(self.density**2) * self.dx
    
    def total_energy(self, V: np.ndarray) -> float:
        """Calculate the total energy of the system."""
        return (self.kinetic_energy() + 
                self.potential_energy(V) + 
                self.interaction_energy())
    
    def time_evolve(self, dt: float, V: np.ndarray) -> None:
        """Evolve the state forward in time using split-step Fourier method."""
        # Half step in position space
        self._wavefunction *= np.exp(-0.5j * dt * 
            (V + self.interaction_strength * np.abs(self._wavefunction)**2))
        
        # Full step in momentum space
        psi_k = np.fft.fft(self._wavefunction)
        psi_k *= np.exp(-0.5j * dt * self._k_grid**2)
        self._wavefunction = np.fft.ifft(psi_k)
        
        # Final half step in position space
        self._wavefunction *= np.exp(-0.5j * dt * 
            (V + self.interaction_strength * np.abs(self._wavefunction)**2))
    
    def get_physical_grid(self) -> np.ndarray:
        """Get the spatial grid in physical units (meters)."""
        return to_physical(self._grid, 'length', self.trap_frequency)
    
    def get_physical_density(self) -> np.ndarray:
        """Get the density in physical units (atoms/m)."""
        return self.density / to_physical(1.0, 'length', self.trap_frequency)
    
    def save_state(self, filename: str):
        """Save the current state to a file."""
        np.savez(filename,
                 wavefunction=self._wavefunction,
                 grid=self._grid,
                 k_grid=self._k_grid,
                 parameters=np.array([
                     self.trap_frequency,
                     self.interaction_strength,
                     self.atom_number,
                     self.box_size
                 ]))
    
    @classmethod
    def load_state(cls, filename: str) -> 'BECState':
        """Load a state from a file."""
        data = np.load(filename)
        params = data['parameters']
        
        state = cls(
            trap_frequency=params[0],
            interaction_strength=params[1],
            atom_number=params[2],
            box_size=params[3]
        )
        
        state._wavefunction = data['wavefunction']
        state._grid = data['grid']
        state._k_grid = data['k_grid']
        
        return state