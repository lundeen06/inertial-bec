import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from ..core.bec_state import BECState
from ..utils.constants import to_dimensionless, to_physical

@dataclass
class AccelerationSensor:
    """
    Class implementing BEC-based acceleration sensing.
    Uses center of mass motion and phase evolution to detect acceleration.
    """
    
    bec_state: BECState
    sensitivity: float = 1e-6  # Acceleration sensitivity in m/s²
    measurement_time: float = 0.1  # Measurement time in seconds
    noise_level: float = 1e-3  # Relative noise level
    
    def __post_init__(self):
        """Initialize sensor parameters and history."""
        self.history = {
            'time': [],
            'com_position': [],
            'com_velocity': [],
            'phase_gradient': [],
            'detected_acceleration': []
        }
        
        # Convert sensitivity to dimensionless units
        self.dimensionless_sensitivity = to_dimensionless(
            self.sensitivity, 'acceleration', self.bec_state.trap_frequency
        )
    
    def compute_com_position(self) -> float:
        """Compute center of mass position."""
        density = self.bec_state.density
        grid = self.bec_state._grid
        return np.sum(grid * density) * self.bec_state.dx / np.sum(density) / self.bec_state.dx
    
    def compute_com_velocity(self) -> float:
        """Compute center of mass velocity using phase gradient."""
        psi = self.bec_state.wavefunction
        dx = self.bec_state.dx
        density = np.abs(psi)**2
        phase = np.angle(psi)
        
        # Compute phase gradient (handle discontinuities)
        phase_gradient = np.gradient(np.unwrap(phase), dx)
        
        # Average velocity from phase gradient
        return np.sum(phase_gradient * density) * dx / np.sum(density) / dx
    
    def compute_phase_gradient(self) -> np.ndarray:
        """Compute spatial phase gradient."""
        phase = np.angle(self.bec_state.wavefunction)
        return np.gradient(np.unwrap(phase), self.bec_state.dx)
    
    def measure_acceleration(self, time_steps: int = 100) -> Tuple[float, float]:
        """
        Measure acceleration using both COM motion and phase evolution.
        Returns (acceleration, uncertainty)
        """
        dt = self.measurement_time / time_steps
        
        # Initial position and phase
        initial_com = self.compute_com_position()
        initial_phase = np.angle(self.bec_state.wavefunction)
        
        # Evolution loop
        for _ in range(time_steps):
            # Get current potential (including acceleration)
            V = self.get_total_potential()
            
            # Evolve state
            self.bec_state.time_evolve(dt, V)
            
            # Record measurements
            current_time = len(self.history['time']) * dt
            self.history['time'].append(current_time)
            self.history['com_position'].append(self.compute_com_position())
            self.history['com_velocity'].append(self.compute_com_velocity())
            self.history['phase_gradient'].append(np.mean(self.compute_phase_gradient()))
        
        # Compute acceleration from COM motion
        com_positions = np.array(self.history['com_position'])
        times = np.array(self.history['time'])
        
        # Fit quadratic to COM motion
        coeffs = np.polyfit(times, com_positions, 2)
        com_acceleration = 2 * coeffs[0]
        
        # Compute acceleration from phase evolution
        phase_gradients = np.array(self.history['phase_gradient'])
        phase_acceleration = np.gradient(phase_gradients, dt).mean()
        
        # Combine measurements with weights based on noise
        weighted_acc = (com_acceleration + phase_acceleration) / 2
        
        # Estimate uncertainty
        uncertainty = self.noise_level * np.abs(weighted_acc)
        
        # Convert to physical units
        acc_physical = to_physical(weighted_acc, 'acceleration', self.bec_state.trap_frequency)
        uncertainty_physical = to_physical(uncertainty, 'acceleration', self.bec_state.trap_frequency)
        
        return acc_physical, uncertainty_physical
    
    def get_total_potential(self, external_acceleration: Optional[float] = None) -> np.ndarray:
        """
        Compute total potential including trap and acceleration.
        external_acceleration in m/s² if provided
        """
        grid = self.bec_state._grid
        
        # Harmonic trap potential
        V_trap = 0.5 * grid**2
        
        # Acceleration potential (if provided)
        if external_acceleration is not None:
            acc_dimensionless = to_dimensionless(
                external_acceleration, 'acceleration', self.bec_state.trap_frequency
            )
            V_acc = -acc_dimensionless * grid
        else:
            V_acc = 0
            
        return V_trap + V_acc
    
    def calibrate(self, known_accelerations: List[float]) -> float:
        """
        Calibrate sensor using known accelerations.
        Returns calibration factor.
        """
        measured = []
        actual = []
        
        for acc in known_accelerations:
            # Reset state
            self.bec_state._initialize_ground_state()
            self.history = {key: [] for key in self.history}
            
            # Measure with known acceleration
            measured_acc, _ = self.measure_acceleration()
            measured.append(measured_acc)
            actual.append(acc)
        
        # Compute calibration factor
        measured = np.array(measured)
        actual = np.array(actual)
        calibration_factor = np.mean(actual / measured)
        
        return calibration_factor
    
    def get_sensitivity_limit(self) -> float:
        """
        Calculate theoretical sensitivity limit based on quantum noise.
        Returns minimum detectable acceleration in m/s².
        """
        N = self.bec_state.atom_number
        T = self.measurement_time
        omega = 2 * np.pi * self.bec_state.trap_frequency
        
        # Quantum noise limit for acceleration measurement
        a_min = np.sqrt(omega / (2 * N * T**2))
        
        return to_physical(a_min, 'acceleration', self.bec_state.trap_frequency)