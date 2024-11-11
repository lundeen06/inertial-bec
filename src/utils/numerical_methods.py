import numpy as np
from typing import Tuple, Callable, Optional
from scipy.fftpack import fft, ifft
from scipy.integrate import complex_ode
from dataclasses import dataclass

@dataclass
class GPESolver:
    """
    Solver for the Gross-Pitaevskii equation using various numerical methods.
    Implements split-operator, Runge-Kutta, and imaginary time evolution methods.
    """
    
    # Grid parameters
    Nx: int  # Number of spatial points
    dx: float  # Spatial step size
    dt: float  # Time step
    
    # Physical parameters
    g: float  # Interaction strength
    
    def __post_init__(self):
        """Initialize grids and operators."""
        # Spatial grid
        self.x = np.linspace(-self.Nx*self.dx/2, self.Nx*self.dx/2, self.Nx)
        
        # Momentum grid
        self.k = 2 * np.pi * np.fft.fftfreq(self.Nx, self.dx)
        
        # Kinetic energy operator in k-space
        self.K = 0.5 * self.k**2
        
        # Initialize FFT plans (if using numpy.fft)
        self.forward_transform = lambda x: np.fft.fft(x)
        self.inverse_transform = lambda x: np.fft.ifft(x)
    
    def split_step(self, psi: np.ndarray, V: np.ndarray, 
                  steps: int = 1) -> np.ndarray:
        """
        Evolve wavefunction using split-operator method.
        
        Args:
            psi: Initial wavefunction
            V: Potential energy array
            steps: Number of time steps
            
        Returns:
            Evolved wavefunction
        """
        # Prepare exponential operators
        exp_K = np.exp(-0.5j * self.dt * self.K)
        
        for _ in range(steps):
            # Half step in position space
            psi *= np.exp(-0.5j * self.dt * 
                         (V + self.g * np.abs(psi)**2))
            
            # Full step in momentum space
            psi_k = self.forward_transform(psi)
            psi_k *= exp_K
            psi = self.inverse_transform(psi_k)
            
            # Final half step in position space
            psi *= np.exp(-0.5j * self.dt * 
                         (V + self.g * np.abs(psi)**2))
        
        return psi
    
    def runge_kutta4(self, psi: np.ndarray, V: np.ndarray, 
                    steps: int = 1) -> np.ndarray:
        """
        Evolve wavefunction using 4th order Runge-Kutta method.
        
        Args:
            psi: Initial wavefunction
            V: Potential energy array
            steps: Number of time steps
            
        Returns:
            Evolved wavefunction
        """
        def compute_derivative(psi_current: np.ndarray) -> np.ndarray:
            """Compute right-hand side of GPE."""
            # Kinetic energy term
            psi_k = self.forward_transform(psi_current)
            T_psi = self.inverse_transform(-self.K * psi_k)
            
            # Potential and interaction terms
            V_total = V + self.g * np.abs(psi_current)**2
            
            return -1j * (T_psi + V_total * psi_current)
        
        for _ in range(steps):
            # RK4 steps
            k1 = compute_derivative(psi)
            k2 = compute_derivative(psi + 0.5*self.dt*k1)
            k3 = compute_derivative(psi + 0.5*self.dt*k2)
            k4 = compute_derivative(psi + self.dt*k3)
            
            # Update wavefunction
            psi += (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Normalize
            psi /= np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        
        return psi
    
    def imaginary_time(self, psi_initial: np.ndarray, V: np.ndarray,
                      max_iter: int = 1000, 
                      tolerance: float = 1e-10) -> Tuple[np.ndarray, float]:
        """
        Find ground state using imaginary time evolution.
        
        Args:
            psi_initial: Initial guess for ground state
            V: Potential energy array
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance for energy
            
        Returns:
            Tuple of (ground state wavefunction, ground state energy)
        """
        # Convert to imaginary time by making dt imaginary
        dt_imag = -1j * self.dt
        psi = psi_initial.copy()
        
        # Energy convergence
        E_old = self.compute_energy(psi, V)
        
        for iter in range(max_iter):
            # Evolution step
            psi = self.split_step(psi, V)
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            psi /= norm
            
            # Check energy convergence
            E_new = self.compute_energy(psi, V)
            if abs(E_new - E_old) < tolerance:
                break
            E_old = E_new
        
        return psi, E_new
    
    def compute_energy(self, psi: np.ndarray, V: np.ndarray) -> float:
        """
        Compute total energy of the system.
        
        Args:
            psi: Wavefunction
            V: Potential energy array
            
        Returns:
            Total energy
        """
        # Kinetic energy
        psi_k = self.forward_transform(psi)
        E_kin = np.sum(self.K * np.abs(psi_k)**2) * self.dx
        
        # Potential energy
        E_pot = np.sum(V * np.abs(psi)**2) * self.dx
        
        # Interaction energy
        E_int = 0.5 * self.g * np.sum(np.abs(psi)**4) * self.dx
        
        return E_kin + E_pot + E_int
    
    def adaptive_step(self, psi: np.ndarray, V: np.ndarray,
                     error_tolerance: float = 1e-6) -> Tuple[np.ndarray, float]:
        """
        Evolve wavefunction with adaptive time stepping.
        
        Args:
            psi: Initial wavefunction
            V: Potential energy array
            error_tolerance: Maximum allowed error per step
            
        Returns:
            Tuple of (evolved wavefunction, optimal time step)
        """
        # Try full step
        psi1 = self.split_step(psi, V)
        
        # Try two half steps
        psi_half = self.split_step(psi, V, steps=1)
        psi2 = self.split_step(psi_half, V, steps=1)
        
        # Estimate error
        error = np.max(np.abs(psi1 - psi2))
        
        # Adjust time step
        if error > error_tolerance:
            self.dt *= 0.95 * (error_tolerance/error)**(1/3)
            return self.adaptive_step(psi, V, error_tolerance)
        elif error < 0.1 * error_tolerance:
            self.dt *= 1.05
        
        return psi2, self.dt
    
    def vortex_initial_state(self, charge: int = 1) -> np.ndarray:
        """
        Generate initial state with vortex of given charge.
        
        Args:
            charge: Vortex winding number
            
        Returns:
            Wavefunction with vortex
        """
        r = np.sqrt(self.x[:, np.newaxis]**2 + self.x[np.newaxis, :]**2)
        theta = np.arctan2(self.x[np.newaxis, :], self.x[:, np.newaxis])
        
        psi = r**abs(charge) * np.exp(-r**2/4) * np.exp(1j*charge*theta)
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * self.dx**2)
        
        return psi
    
    def compute_phase_gradient(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute phase gradient of wavefunction.
        
        Args:
            psi: Wavefunction
            
        Returns:
            Phase gradient array
        """
        phase = np.angle(psi)
        return np.gradient(np.unwrap(phase), self.dx)
    
    def compute_current_density(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute probability current density.
        
        Args:
            psi: Wavefunction
            
        Returns:
            Current density array
        """
        phase_gradient = self.compute_phase_gradient(psi)
        return np.abs(psi)**2 * phase_gradient