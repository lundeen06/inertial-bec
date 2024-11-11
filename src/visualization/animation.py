import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
from ..core.wave_function import WaveFunction
from ..utils.constants import to_physical

@dataclass
class BECAnimator:
    """
    Class for creating animations of BEC dynamics.
    Supports various types of animations including density evolution,
    phase dynamics, and real-time measurements.
    """
    
    def __init__(self):
        """Initialize animation settings."""
        self.fig = None
        self.ax = None
        self._animation = None
        self.is_running = False
        
        # Default style settings
        plt.style.use('seaborn')
        self.default_figsize = (10, 6)
        self.default_fps = 30
        
    def setup_figure(self, title: str = "") -> None:
        """Set up the figure and axes for animation."""
        self.fig, self.ax = plt.subplots(figsize=self.default_figsize)
        self.ax.set_title(title)
        self.ax.grid(True)
    
    def density_evolution(self, 
                        wave_functions: List[WaveFunction],
                        times: np.ndarray,
                        interval: int = 50) -> FuncAnimation:
        """
        Create animation of density evolution.
        
        Args:
            wave_functions: List of wavefunctions at different times
            times: Array of time points
            interval: Time between frames in milliseconds
        """
        if self.fig is None:
            self.setup_figure("BEC Density Evolution")
            
        # Convert to physical units
        x_physical = to_physical(wave_functions[0].grid, 'length', 1.0) * 1e6  # μm
        times_ms = times * 1e3  # Convert to milliseconds
        
        # Set up plot elements
        line, = self.ax.plot([], [], 'b-', lw=2, label='Density')
        fill = self.ax.fill_between([], [], alpha=0.3, color='blue')
        time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        
        # Set axis labels and limits
        self.ax.set_xlabel('Position (μm)')
        self.ax.set_ylabel('Density')
        self.ax.set_xlim(x_physical.min(), x_physical.max())
        max_density = max([np.max(wf.density) for wf in wave_functions])
        self.ax.set_ylim(0, 1.1 * max_density)
        
        def init():
            """Initialize animation."""
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
            
        def animate(i):
            """Update animation frame."""
            density = wave_functions[i].density
            line.set_data(x_physical, density)
            
            # Update fill
            fill.remove()
            self.ax.fill_between(x_physical, density, alpha=0.3, color='blue')
            
            time_text.set_text(f'Time: {times_ms[i]:.1f} ms')
            return line, time_text
        
        self._animation = FuncAnimation(
            self.fig, animate, init_func=init,
            frames=len(wave_functions), interval=interval, blit=True
        )
        
        return self._animation
    
    def phase_evolution(self,
                       wave_functions: List[WaveFunction],
                       times: np.ndarray,
                       interval: int = 50) -> FuncAnimation:
        """Create animation of phase evolution."""
        if self.fig is None:
            self.setup_figure("BEC Phase Evolution")
            
        x_physical = to_physical(wave_functions[0].grid, 'length', 1.0) * 1e6
        times_ms = times * 1e3
        
        # Create phase plot
        line, = self.ax.plot([], [], 'r-', lw=2, label='Phase')
        time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        
        self.ax.set_xlabel('Position (μm)')
        self.ax.set_ylabel('Phase (rad)')
        self.ax.set_xlim(x_physical.min(), x_physical.max())
        
        # Find phase range
        all_phases = [np.unwrap(wf.phase) for wf in wave_functions]
        phase_min = min([np.min(p) for p in all_phases])
        phase_max = max([np.max(p) for p in all_phases])
        self.ax.set_ylim(phase_min, phase_max)
        
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
            
        def animate(i):
            phase = np.unwrap(wave_functions[i].phase)
            line.set_data(x_physical, phase)
            time_text.set_text(f'Time: {times_ms[i]:.1f} ms')
            return line, time_text
        
        self._animation = FuncAnimation(
            self.fig, animate, init_func=init,
            frames=len(wave_functions), interval=interval, blit=True
        )
        
        return self._animation
    
    def real_time_measurement(self,
                            wave_functions: List[WaveFunction],
                            times: np.ndarray,
                            measurement_func: Callable[[WaveFunction], float],
                            measurement_label: str,
                            interval: int = 50) -> FuncAnimation:
        """
        Create animation showing real-time measurement evolution.
        
        Args:
            measurement_func: Function that computes measurement from wavefunction
            measurement_label: Label for the measured quantity
        """
        if self.fig is None:
            self.setup_figure(f"Real-time {measurement_label}")
        
        times_ms = times * 1e3
        measurements = [measurement_func(wf) for wf in wave_functions]
        
        # Set up both wavefunction and measurement plots
        self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Wavefunction plot
        density_line, = ax1.plot([], [], 'b-', lw=2, label='Density')
        ax1.set_xlabel('Position (μm)')
        ax1.set_ylabel('Density')
        
        # Measurement plot
        measurement_line, = ax2.plot([], [], 'g-', lw=2, label=measurement_label)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel(measurement_label)
        
        # Set axis limits
        x_physical = to_physical(wave_functions[0].grid, 'length', 1.0) * 1e6
        ax1.set_xlim(x_physical.min(), x_physical.max())
        ax1.set_ylim(0, 1.1 * max([np.max(wf.density) for wf in wave_functions]))
        
        ax2.set_xlim(times_ms[0], times_ms[-1])
        ax2.set_ylim(min(measurements) * 0.9, max(measurements) * 1.1)
        
        def init():
            density_line.set_data([], [])
            measurement_line.set_data([], [])
            return density_line, measurement_line
            
        def animate(i):
            # Update density plot
            density_line.set_data(x_physical, wave_functions[i].density)
            
            # Update measurement plot
            measurement_line.set_data(times_ms[:i+1], measurements[:i+1])
            return density_line, measurement_line
        
        self._animation = FuncAnimation(
            self.fig, animate, init_func=init,
            frames=len(wave_functions), interval=interval, blit=True
        )
        
        return self._animation
    
    def save_animation(self, filename: str, fps: int = 30, 
                      dpi: int = 200) -> None:
        """Save animation to file."""
        if self._animation is None:
            raise ValueError("No animation to save. Create an animation first.")
            
        writer = PillowWriter(fps=fps)
        self._animation.save(filename, writer=writer, dpi=dpi)
    
    def combine_measurements(self,
                           wave_functions: List[WaveFunction],
                           times: np.ndarray,
                           measurements: List[Tuple[Callable, str]],
                           interval: int = 50) -> FuncAnimation:
        """
        Create animation with multiple synchronized measurements.
        
        Args:
            measurements: List of (measurement_function, label) tuples
        """
        if self.fig is None:
            self.setup_figure("Combined Measurements")
        
        n_measurements = len(measurements)
        fig, axs = plt.subplots(n_measurements + 1, 1, 
                               figsize=(10, 4*(n_measurements + 1)))
        
        times_ms = times * 1e3
        x_physical = to_physical(wave_functions[0].grid, 'length', 1.0) * 1e6
        
        # Setup density plot
        density_line, = axs[0].plot([], [], 'b-', lw=2, label='Density')
        axs[0].set_xlabel('Position (μm)')
        axs[0].set_ylabel('Density')
        axs[0].set_xlim(x_physical.min(), x_physical.max())
        axs[0].set_ylim(0, 1.1 * max([np.max(wf.density) for wf in wave_functions]))
        
        # Setup measurement plots
        measurement_lines = []
        measurement_data = []
        
        for i, (func, label) in enumerate(measurements, 1):
            line, = axs[i].plot([], [], '-', lw=2, label=label)
            measurement_lines.append(line)
            measurement_data.append([func(wf) for wf in wave_functions])
            
            axs[i].set_xlabel('Time (ms)')
            axs[i].set_ylabel(label)
            axs[i].set_xlim(times_ms[0], times_ms[-1])
            axs[i].set_ylim(min(measurement_data[-1]) * 0.9,
                           max(measurement_data[-1]) * 1.1)
            axs[i].grid(True)
            
        def init():
            density_line.set_data([], [])
            for line in measurement_lines:
                line.set_data([], [])
            return [density_line] + measurement_lines
            
        def animate(i):
            # Update density plot
            density_line.set_data(x_physical, wave_functions[i].density)
            
            # Update measurement plots
            for line, data in zip(measurement_lines, measurement_data):
                line.set_data(times_ms[:i+1], data[:i+1])
            
            return [density_line] + measurement_lines
        
        self._animation = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(wave_functions), interval=interval, blit=True
        )
        
        plt.tight_layout()
        return self._animation