import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple
from ..core.bec_state import BECState
from ..physics.acceleration import AccelerationSensor

class BECVisualizer:
    """Class for visualizing BEC states and dynamics."""
    
    def __init__(self, bec_state: BECState):
        self.bec_state = bec_state
        self.fig = None
        self.axes = None
        self._animation = None
        
    def plot_density(self, ax: Optional[plt.Axes] = None, title: str = "BEC Density Profile") -> Tuple[plt.Figure, plt.Axes]:
        """Plot current density profile."""
        if ax is None:
            if self.fig is None:
                self.fig, self.axes = plt.subplots(figsize=(10, 6))
            ax = self.axes
        else:
            self.axes = ax
            self.fig = ax.figure
            
        ax.clear()
        density = self.bec_state.density
        grid = self.bec_state.get_physical_grid() * 1e6  # Convert to micrometers
        
        ax.plot(grid, density, 'b-', label='Density')
        ax.fill_between(grid, density, alpha=0.3)
        
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Density (a.u.)')
        ax.set_title(title)
        ax.grid(True)
        
        return self.fig, ax
    
    def plot_phase(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot current phase profile."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        phase = self.bec_state.phase
        grid = self.bec_state.get_physical_grid() * 1e6
        
        ax.plot(grid, np.unwrap(phase), 'r-', label='Phase')
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title('BEC Phase Profile')
        ax.grid(True)
        
        return fig, ax
    
    def plot_wavefunction_3d(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create 3D plot of wavefunction amplitude and phase."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        grid = self.bec_state.get_physical_grid() * 1e6
        psi = self.bec_state.wavefunction
        
        # Plot amplitude
        ax.plot(grid, np.real(psi), np.imag(psi), 'b-', label='Wavefunction')
        
        # Project onto walls
        ax.plot(grid, np.real(psi), np.min(np.imag(psi))*np.ones_like(grid), 
                'b--', alpha=0.3)
        ax.plot(grid, np.min(np.real(psi))*np.ones_like(grid), np.imag(psi), 
                'b--', alpha=0.3)
        
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Re(ψ)')
        ax.set_zlabel('Im(ψ)')
        ax.set_title('BEC Wavefunction')
        
        return fig, ax
    
    def create_evolution_animation(self, 
                                 times: List[float], 
                                 states: List[np.ndarray],
                                 interval: int = 50) -> FuncAnimation:
        """Create animation of time evolution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        grid = self.bec_state.get_physical_grid() * 1e6
        line, = ax.plot([], [], 'b-', label='Density')
        fill = ax.fill_between([], [], alpha=0.3)
        
        ax.set_xlim(grid.min(), grid.max())
        ax.set_ylim(0, 1.2 * max([np.max(np.abs(psi)**2) for psi in states]))
        
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Density (a.u.)')
        ax.set_title('BEC Evolution')
        ax.grid(True)
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(i):
            density = np.abs(states[i])**2
            line.set_data(grid, density)
            
            # Update fill
            fill.remove()
            ax.fill_between(grid, density, alpha=0.3)
            
            ax.set_title(f'Time: {times[i]*1e3:.1f} ms')
            return line,
        
        self._animation = FuncAnimation(fig, animate, init_func=init,
                                      frames=len(times), interval=interval,
                                      blit=True)
        
        return self._animation
    
    def plot_acceleration_measurement(self, 
                                   sensor: AccelerationSensor,
                                   true_acceleration: Optional[float] = None):
        """Plot acceleration measurement results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot center of mass motion
        times = np.array(sensor.history['time']) * 1e3  # Convert to ms
        com = np.array(sensor.history['com_position'])
        
        ax1.plot(times, com, 'b.-', label='COM Position')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Position (a.u.)')
        ax1.set_title('Center of Mass Motion')
        ax1.grid(True)
        
        # Plot measured acceleration
        acc = np.array(sensor.history['detected_acceleration'])
        ax2.plot(times[:-1], acc, 'r.-', label='Measured')
        
        if true_acceleration is not None:
            ax2.axhline(y=true_acceleration, color='k', linestyle='--',
                       label='True Value')
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Acceleration (m/s²)')
        ax2.set_title('Acceleration Measurement')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_sensitivity_analysis(self, 
                                sensor: AccelerationSensor,
                                accelerations: np.ndarray,
                                measurements: np.ndarray,
                                uncertainties: np.ndarray):
        """Plot sensitivity analysis results."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(accelerations, measurements, yerr=uncertainties,
                   fmt='bo-', capsize=5, label='Measurements')
        
        # Plot ideal response
        ax.plot(accelerations, accelerations, 'k--', label='Ideal Response')
        
        ax.set_xlabel('Applied Acceleration (m/s²)')
        ax.set_ylabel('Measured Acceleration (m/s²)')
        ax.set_title('Sensor Response and Sensitivity')
        ax.grid(True)
        ax.legend()
        
        # Add sensitivity limit
        sensitivity = sensor.get_sensitivity_limit()
        ax.axhspan(-sensitivity, sensitivity, color='r', alpha=0.2,
                  label=f'Sensitivity Limit: {sensitivity:.2e} m/s²')
        
        return fig, ax
    
    def save_plots(self, base_filename: str):
        """Save all current plots to files."""
        # Density plot
        fig_density, _ = self.plot_density()
        fig_density.savefig(f'{base_filename}_density.png')
        
        # Phase plot
        fig_phase, _ = self.plot_phase()
        fig_phase.savefig(f'{base_filename}_phase.png')
        
        # 3D wavefunction plot
        fig_3d, _ = self.plot_wavefunction_3d()
        fig_3d.savefig(f'{base_filename}_3d.png')
        
        plt.close('all')