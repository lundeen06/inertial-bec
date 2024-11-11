import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
from ..core.wave_function import WaveFunction
from ..utils.constants import HBAR, M_RB87

@dataclass
class Hamiltonian:
    """
    Class representing the Hamiltonian operator for the BEC system.
    Combines kinetic energy, potential energy, and interactions.
    """
    
    # Grid parameters
    grid: np.ndarray
    k_grid: np.ndarray
    
    # Interaction parameters
    interaction_strength: float
    
    # Optional time-dependent parameters
    time_dependent_potential: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    
    def __post_init__(self):
        """Initialize derived quantities."""
        self.dx = self.grid[1] - self.grid[0]
        self.dk = self.k_grid[1] - self.k_grid[0]
        
        # Precompute kinetic energy operator in k-space
        self.kinetic_operator = 0.5 * HBAR**2 * self.k_grid**2 / M_RB87
    
    def kinetic_energy(self, psi: WaveFunction) -> np.ndarray:
        """
        Apply kinetic energy operator.
        Returns: -ℏ²/2m ∇² ψ
        """
        psi_k = psi.momentum_space()
        return np.fft.ifft(self.kinetic_operator * psi_k)
    
    def potential_energy(self, psi: WaveFunction, V: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Apply potential energy operator.
        Includes both static and time-dependent parts if specified.
        """
        V_total = V.copy()
        if self.time_dependent_potential is not None:
            V_total += self.time_dependent_potential(self.grid, t)
        return V_total * psi.psi
    
    def interaction_energy(self, psi: WaveFunction) -> np.ndarray:
        """
        Apply interaction energy operator.
        Returns: g|ψ|²ψ for contact interactions
        """
        return self.interaction_strength * np.abs(psi.psi)**2 * psi.psi
    
    def apply(self, psi: WaveFunction, V: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Apply full Hamiltonian operator.
        H = T + V + g|ψ|²
        """
        return (self.kinetic_energy(psi) + 
                self.potential_energy(psi, V, t) + 
                self.interaction_energy(psi))
    
    def energy_expectation(self, psi: WaveFunction, V: np.ndarray) -> float:
        """Calculate expectation value of energy."""
        return np.real(np.sum(np.conj(psi.psi) * self.apply(psi, V)) * self.dx)
    
    def commutator(self, op1: Callable, op2: Callable, psi: WaveFunction) -> np.ndarray:
        """
        Calculate commutator of two operators: [op1, op2]
        """
        return op1(op2(psi)) - op2(op1(psi))
    
    def time_evolution_operator(self, dt: float, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate split-step time evolution operators.
        Returns: (exp(-iVdt/2ℏ), exp(-iTdt/ℏ))
        """
        exp_V = np.exp(-0.5j * dt * V / HBAR)
        exp_T = np.exp(-1j * dt * self.kinetic_operator / HBAR)
        return exp_V, exp_T
    
    def get_ground_state(self, V: np.ndarray, 
                        initial_guess: Optional[WaveFunction] = None,
                        max_iter: int = 1000,
                        tolerance: float = 1e-10) -> Tuple[WaveFunction, float]:
        """
        Find ground state using imaginary time evolution.
        Returns: (ground state wavefunction, ground state energy)
        """
        if initial_guess is None:
            # Create Gaussian initial guess
            sigma = np.sqrt(HBAR / (M_RB87 * 100.0))  # Characteristic length
            psi = np.exp(-self.grid**2 / (4*sigma**2))
            psi /= np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            initial_guess = WaveFunction(psi, self.grid, self.k_grid)
        
        psi = initial_guess.clone()
        E_old = self.energy_expectation(psi, V)
        
        dt = 1e-5  # Imaginary time step
        exp_V, exp_T = self.time_evolution_operator(dt, V)
        
        for _ in range(max_iter):
            # Split-step imaginary time evolution
            psi.psi = exp_V * psi.psi
            psi.psi = np.fft.ifft(exp_T * np.fft.fft(psi.psi))
            psi.psi = exp_V * psi.psi
            
            # Normalize
            psi.normalize()
            
            # Check convergence
            E_new = self.energy_expectation(psi, V)
            if abs(E_new - E_old) < tolerance:
                break
            E_old = E_new
        
        return psi, E_new
    
    def excitation_spectrum(self, psi: WaveFunction, V: np.ndarray,
                          n_modes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Bogoliubov excitation spectrum.
        Returns: (excitation energies, mode functions)
        """
        # Construct Bogoliubov-de Gennes matrix
        N = len(self.grid)
        M = np.zeros((2*N, 2*N), dtype=complex)
        
        # Kinetic + Potential + 2g|ψ|² terms
        diag = np.fft.ifft(self.kinetic_operator * np.fft.fft(np.eye(N), axis=1), axis=1)
        diag += np.diag(V + 2 * self.interaction_strength * np.abs(psi.psi)**2)
        
        # Off-diagonal terms
        off_diag = np.diag(self.interaction_strength * psi.psi**2)
        
        M[:N, :N] = diag
        M[:N, N:] = off_diag
        M[N:, :N] = -np.conj(off_diag)
        M[N:, N:] = -np.conj(diag)
        
        # Find eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(M)
        
        # Sort and select positive energy modes
        idx = np.argsort(np.abs(eigenvals))[-n_modes:]
        energies = eigenvals[idx]
        modes = eigenvecs[:, idx]
        
        return energies, modes