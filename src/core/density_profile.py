import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.interpolate import interp1d
from ..utils.constants import to_physical, to_dimensionless

@dataclass
class DensityProfile:
    """
    Class for handling and analyzing BEC density profiles.
    Provides methods for density calculations, analysis, and transformations.
    """
    
    grid: np.ndarray  # Spatial grid points
    density: np.ndarray  # Density values
    trap_frequency: float  # Trap frequency in Hz
    
    def __post_init__(self):
        """Initialize derived quantities."""
        self.dx = self.grid[1] - self.grid[0]
        self.normalized_density = self.density / np.sum(self.density) / self.dx
    
    @property
    def total_atoms(self) -> float:
        """Calculate total number of atoms."""
        return np.sum(self.density) * self.dx
    
    def center_of_mass(self) -> float:
        """Calculate center of mass position."""
        return np.sum(self.grid * self.density) * self.dx / self.total_atoms
    
    def rms_width(self) -> float:
        """Calculate RMS width of the density distribution."""
        com = self.center_of_mass()
        return np.sqrt(np.sum((self.grid - com)**2 * self.density) * 
                      self.dx / self.total_atoms)
    
    def peak_density(self) -> Tuple[float, float]:
        """Find peak density and its position."""
        peak_idx = np.argmax(self.density)
        return self.grid[peak_idx], self.density[peak_idx]
    
    def thomas_fermi_radius(self, threshold: float = 0.05) -> float:
        """
        Calculate Thomas-Fermi radius.
        threshold: fraction of peak density to define edge
        """
        peak_pos, peak_val = self.peak_density()
        threshold_density = peak_val * threshold
        
        # Find points where density crosses threshold
        above_threshold = self.density > threshold_density
        edges = np.where(np.diff(above_threshold))[0]
        
        if len(edges) >= 2:
            return (self.grid[edges[-1]] - self.grid[edges[0]]) / 2
        else:
            return self.rms_width()
    
    def get_physical_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to physical units (meters and atoms/meter)."""
        x_physical = to_physical(self.grid, 'length', self.trap_frequency)
        n_physical = self.density / to_physical(1.0, 'length', self.trap_frequency)
        return x_physical, n_physical
    
    def interpolate(self, new_grid: np.ndarray) -> 'DensityProfile':
        """Interpolate density onto new grid points."""
        interpolator = interp1d(self.grid, self.density, 
                              kind='cubic', bounds_error=False, 
                              fill_value=0)
        new_density = interpolator(new_grid)
        return DensityProfile(new_grid, new_density, self.trap_frequency)
    
    def smooth(self, window_size: int = 5) -> 'DensityProfile':
        """Apply smoothing to density profile."""
        window = np.ones(window_size) / window_size
        smoothed_density = np.convolve(self.density, window, mode='same')
        return DensityProfile(self.grid, smoothed_density, self.trap_frequency)
    
    def integrate_region(self, x_min: float, x_max: float) -> float:
        """Integrate density over specified region."""
        mask = (self.grid >= x_min) & (self.grid <= x_max)
        return np.sum(self.density[mask]) * self.dx
    
    def find_nodes(self, threshold: float = 1e-6) -> List[float]:
        """Find positions of nodes in density profile."""
        below_threshold = self.density < (np.max(self.density) * threshold)
        transitions = np.where(np.diff(below_threshold))[0]
        return [self.grid[idx] for idx in transitions]
    
    def moment(self, order: int) -> float:
        """Calculate nth order moment of density distribution."""
        com = self.center_of_mass()
        return np.sum((self.grid - com)**order * self.density) * self.dx / self.total_atoms
    
    def entropy_estimate(self, bins: int = 50) -> float:
        """
        Estimate entropy of density distribution using binned Shannon entropy.
        """
        hist, _ = np.histogram(self.density, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log(hist))
    
    def get_density_fluctuations(self, window_size: int = 20) -> np.ndarray:
        """Calculate local density fluctuations."""
        smoothed = self.smooth(window_size).density
        return (self.density - smoothed) / smoothed
    
    def overlap(self, other: 'DensityProfile') -> float:
        """Calculate overlap integral with another density profile."""
        if not np.array_equal(self.grid, other.grid):
            other = other.interpolate(self.grid)
        
        n1 = self.normalized_density
        n2 = other.normalized_density
        
        return np.abs(np.sum(np.sqrt(n1 * n2)) * self.dx)
    
    def to_dict(self) -> dict:
        """Convert profile to dictionary for saving."""
        return {
            'grid': self.grid.tolist(),
            'density': self.density.tolist(),
            'trap_frequency': self.trap_frequency
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DensityProfile':
        """Create profile from dictionary."""
        return cls(
            grid=np.array(data['grid']),
            density=np.array(data['density']),
            trap_frequency=data['trap_frequency']
        )