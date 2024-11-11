import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
from ..utils.constants import HBAR, M_RB87
from ..core.wave_function import WaveFunction

@dataclass
class InteractionModel:
    """
    Class handling atomic interactions in the BEC.
    Implements various interaction models including contact interactions,
    finite-range interactions, and loss processes.
    """
    
    # Basic interaction parameters
    scattering_length: float  # s-wave scattering length
    atom_number: int         # Number of atoms
    
    # Optional parameters for advanced models
    effective_range: Optional[float] = None  # For finite-range interactions
    three_body_coefficient: Optional[float] = None  # For three-body losses
    temperature: Optional[float] = None  # For temperature-dependent effects
    
    def __post_init__(self):
        """Initialize derived quantities."""
        # Calculate basic interaction strength
        self.g_1d = 2 * HBAR**2 * self.scattering_length / (M_RB87)
        
        # Effective 1D interaction strength (if in quasi-1D regime)
        if self.temperature is not None:
            self.thermal_wavelength = np.sqrt(2 * np.pi * HBAR**2 / (M_RB87 * self.temperature))
    
    def contact_interaction(self, psi: WaveFunction) -> np.ndarray:
        """
        Standard contact interaction term.
        Returns: g|ψ|²ψ
        """
        return self.g_1d * np.abs(psi.psi)**2 * psi.psi
    
    def finite_range_interaction(self, psi: WaveFunction) -> np.ndarray:
        """
        Finite-range interaction using effective range expansion.
        More accurate for stronger interactions.
        """
        if self.effective_range is None:
            raise ValueError("Effective range not specified for finite-range interaction")
            
        density = np.abs(psi.psi)**2
        k_typical = np.sqrt(np.mean(np.abs(np.gradient(psi.psi, psi.grid[1] - psi.grid[0]))**2))
        
        # Effective coupling including momentum dependence
        g_eff = self.g_1d * (1 + 0.5 * self.effective_range * k_typical**2)
        
        return g_eff * density * psi.psi
    
    def three_body_loss(self, psi: WaveFunction, dt: float) -> Tuple[np.ndarray, float]:
        """
        Calculate three-body loss effect and atom loss rate.
        Returns: (modified wavefunction, number of atoms lost)
        """
        if self.three_body_coefficient is None:
            raise ValueError("Three-body loss coefficient not specified")
            
        density = np.abs(psi.psi)**2
        loss_rate = self.three_body_coefficient * density**2
        
        # Update wave function with losses
        psi_new = psi.psi * np.exp(-0.5 * loss_rate * dt)
        atoms_lost = self.atom_number * (1 - np.sum(np.abs(psi_new)**2) / 
                                       np.sum(np.abs(psi.psi)**2))
        
        return psi_new, atoms_lost
    
    def thermal_effects(self, psi: WaveFunction, T: float) -> np.ndarray:
        """
        Include thermal effects in interaction strength.
        Valid in the quasi-condensate regime.
        """
        # Thermal phase fluctuations
        phase_fluct = np.sqrt(T * M_RB87 / (HBAR**2 * self.atom_number)) * np.random.randn(*psi.psi.shape)
        
        # Modified interaction strength due to thermal depletion
        g_T = self.g_1d * (1 + 2 * np.sqrt(T * M_RB87 / (HBAR**2 * self.atom_number)))
        
        return g_T * np.abs(psi.psi)**2 * psi.psi * np.exp(1j * phase_fluct)
    
    def beyond_mean_field(self, psi: WaveFunction) -> np.ndarray:
        """
        Include beyond-mean-field corrections (Lee-Huang-Yang term).
        Important for strong interactions.
        """
        density = np.abs(psi.psi)**2
        na3 = self.scattering_length**3 * density
        
        # Lee-Huang-Yang correction
        lhy_factor = 1 + (32/3) * np.sqrt(na3/np.pi)
        
        return self.g_1d * density * psi.psi * lhy_factor
    
    def interaction_energy(self, psi: WaveFunction, 
                         include_beyond_mf: bool = False) -> float:
        """
        Calculate total interaction energy.
        Option to include beyond-mean-field corrections.
        """
        if include_beyond_mf:
            int_term = self.beyond_mean_field(psi)
        else:
            int_term = self.contact_interaction(psi)
        
        return 0.5 * np.real(np.sum(np.conj(psi.psi) * int_term)) * (psi.grid[1] - psi.grid[0])
    
    def calculate_chemical_potential(self, psi: WaveFunction) -> float:
        """Calculate chemical potential including interaction effects."""
        density = np.abs(psi.psi)**2
        return self.g_1d * density.max()  # Peak mean-field energy
    
    def healing_length(self, peak_density: float) -> float:
        """Calculate healing length at given density."""
        return np.sqrt(HBAR**2 / (2 * M_RB87 * self.g_1d * peak_density))
    
    def sound_speed(self, peak_density: float) -> float:
        """Calculate sound speed in the condensate."""
        return np.sqrt(self.g_1d * peak_density / M_RB87)
    
    def is_quasi_1d(self, trap_frequency_perp: float) -> bool:
        """
        Check if system is in quasi-1D regime.
        Requires: interaction energy << transverse excitation energy
        """
        e_int = self.g_1d * self.atom_number  # Typical interaction energy
        e_perp = HBAR * trap_frequency_perp    # Transverse excitation energy
        return e_int < e_perp