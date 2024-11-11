import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from ..utils.constants import HBAR, M_RB87
from scipy.fftpack import fft, ifft

@dataclass
class WaveFunction:
    """
    Class representing the quantum wavefunction of the BEC.
    Handles operations, transformations, and analysis of the quantum state.
    """
    
    # Core properties
    psi: np.ndarray  # Complex wavefunction array
    grid: np.ndarray  # Spatial grid points
    k_grid: Optional[np.ndarray] = None  # Momentum grid points
    
    def __post_init__(self):
        """Initialize derived quantities and grids."""
        self.dx = self.grid[1] - self.grid[0]
        if self.k_grid is None:
            self.k_grid = 2 * np.pi * np.fft.fftfreq(len(self.grid), self.dx)
        self.normalize()
    
    def normalize(self) -> None:
        """Normalize the wavefunction."""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        if norm > 0:
            self.psi /= norm
    
    @property
    def density(self) -> np.ndarray:
        """Get probability density."""
        return np.abs(self.psi)**2
    
    @property
    def phase(self) -> np.ndarray:
        """Get phase of wavefunction."""
        return np.angle(self.psi)
    
    def momentum_space(self) -> np.ndarray:
        """Transform to momentum space."""
        return fft(self.psi) * self.dx / np.sqrt(2*np.pi)
    
    def position_expectation(self) -> float:
        """Calculate expectation value of position."""
        return np.sum(self.grid * self.density) * self.dx
    
    def momentum_expectation(self) -> float:
        """Calculate expectation value of momentum."""
        psi_k = self.momentum_space()
        return np.sum(self.k_grid * np.abs(psi_k)**2) * (2*np.pi/self.grid[-1])
    
    def kinetic_energy(self) -> float:
        """Calculate kinetic energy."""
        psi_k = self.momentum_space()
        return np.sum(0.5 * self.k_grid**2 * np.abs(psi_k)**2) * (2*np.pi/self.grid[-1])
    
    def potential_energy(self, V: np.ndarray) -> float:
        """Calculate potential energy for given potential."""
        return np.sum(V * self.density) * self.dx
    
    def total_energy(self, V: np.ndarray) -> float:
        """Calculate total energy."""
        return self.kinetic_energy() + self.potential_energy(V)
    
    def translate(self, displacement: float) -> 'WaveFunction':
        """Translate wavefunction by given displacement."""
        psi_k = self.momentum_space()
        translated_psi_k = psi_k * np.exp(-1j * self.k_grid * displacement)
        translated_psi = ifft(translated_psi_k) * np.sqrt(2*np.pi) / self.dx
        return WaveFunction(translated_psi, self.grid, self.k_grid)
    
    def boost(self, momentum: float) -> 'WaveFunction':
        """Apply momentum boost to wavefunction."""
        boosted_psi = self.psi * np.exp(1j * momentum * self.grid)
        return WaveFunction(boosted_psi, self.grid, self.k_grid)
    
    def uncertainty_product(self) -> float:
        """Calculate position-momentum uncertainty product."""
        x_mean = self.position_expectation()
        p_mean = self.momentum_expectation()
        
        x_variance = np.sum((self.grid - x_mean)**2 * self.density) * self.dx
        
        psi_k = self.momentum_space()
        p_variance = np.sum((self.k_grid - p_mean)**2 * 
                          np.abs(psi_k)**2) * (2*np.pi/self.grid[-1])
        
        return np.sqrt(x_variance * p_variance)
    
    def probability_current(self) -> np.ndarray:
        """Calculate probability current density."""
        dx = self.grid[1] - self.grid[0]
        dpsi_dx = np.gradient(self.psi, dx)
        return np.imag(np.conj(self.psi) * dpsi_dx) / M_RB87
    
    def add_vortex(self, charge: int, core_size: float) -> 'WaveFunction':
        """Add a vortex to the wavefunction."""
        x = self.grid
        r = np.abs(x)
        theta = np.angle(x + 1j*0)  # For 1D, just use x axis
        
        vortex_factor = (r/core_size)**abs(charge) * np.exp(-r**2/(2*core_size**2))
        vortex_factor *= np.exp(1j * charge * theta)
        
        new_psi = self.psi * vortex_factor
        return WaveFunction(new_psi, self.grid, self.k_grid)
    
    def overlap(self, other: 'WaveFunction') -> complex:
        """Calculate overlap with another wavefunction."""
        if not np.array_equal(self.grid, other.grid):
            raise ValueError("Grid points must match for overlap calculation")
        return np.sum(np.conj(self.psi) * other.psi) * self.dx
    
    def coherent_state(self, x0: float, p0: float, width: float) -> 'WaveFunction':
        """Create a coherent state centered at (x0, p0)."""
        norm = 1/(2*np.pi*width**2)**(1/4)
        psi = norm * np.exp(-(self.grid - x0)**2/(4*width**2) + 1j*p0*self.grid)
        return WaveFunction(psi, self.grid, self.k_grid)
    
    def to_dict(self) -> dict:
        """Convert wavefunction to dictionary for saving."""
        return {
            'psi_real': np.real(self.psi).tolist(),
            'psi_imag': np.imag(self.psi).tolist(),
            'grid': self.grid.tolist(),
            'k_grid': self.k_grid.tolist() if self.k_grid is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WaveFunction':
        """Create wavefunction from dictionary."""
        psi = np.array(data['psi_real']) + 1j * np.array(data['psi_imag'])
        grid = np.array(data['grid'])
        k_grid = np.array(data['k_grid']) if data['k_grid'] is not None else None
        return cls(psi, grid, k_grid)
    
    def clone(self) -> 'WaveFunction':
        """Create a copy of the wavefunction."""
        return WaveFunction(self.psi.copy(), self.grid.copy(), 
                          self.k_grid.copy() if self.k_grid is not None else None)