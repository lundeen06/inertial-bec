import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List
from dataclasses import dataclass
from ..core.wave_function import WaveFunction
from ..utils.constants import to_physical

@dataclass
class PhaseVisualizer:
    """
    Class for visualizing phase-related properties of the BEC wavefunction.
    Includes phase profiles, phase gradients, and vortex detection.
    """
    
    def __init__(self):
        """Initialize custom colormaps and plotting defaults."""
        # Create custom colormap for phase plots
        colors = ['#ff0000', '#ff8000', '#ffff00', '#80ff00', '#00ff00',
                 '#00ff80', '#00ffff', '#0080ff', '#0000ff', '#8000ff',
                 '#ff00ff', '#ff0080', '#ff0000']
        self.phase_cmap = LinearSegmentedColormap.from_list('phase', colors)
        
        # Default figure settings
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'figure.figsize': [10, 6]
        })
    
    def plot_phase_profile(self, wf: WaveFunction, 
                          title: str = "Phase Profile") -> Tuple[plt.Figure, plt.Axes]:
        """Plot the phase profile of the wavefunction."""
        fig, ax = plt.subplots()
        
        # Get physical grid
        x_physical = to_physical(wf.grid, 'length', 1.0) * 1e6  # Convert to micrometers
        
        # Calculate phase and unwrap it
        phase = np.unwrap(wf.phase)
        
        # Plot phase
        ax.plot(x_physical, phase, 'b-', label='Phase')
        
        # Add density overlay for reference (scaled)
        density = wf.density / np.max(wf.density) * np.pi  # Scale to match phase range
        ax.plot(x_physical, density, 'r--', alpha=0.5, label='Normalized Density')
        
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        
        return fig, ax
    
    def plot_phase_gradient(self, wf: WaveFunction, 
                          smoothing: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the phase gradient (local momentum)."""
        fig, ax = plt.subplots()
        
        # Calculate phase gradient
        dx = wf.grid[1] - wf.grid[0]
        phase = np.unwrap(wf.phase)
        gradient = np.gradient(phase, dx)
        
        if smoothing:
            # Apply smoothing to reduce noise
            window_size = 5
            gradient = np.convolve(gradient, np.ones(window_size)/window_size, mode='same')
        
        # Convert to physical units
        x_physical = to_physical(wf.grid, 'length', 1.0) * 1e6
        gradient_physical = to_physical(gradient, 'momentum', 1.0)
        
        # Plot gradient
        ax.plot(x_physical, gradient_physical, 'g-', label='Phase Gradient')
        
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Local Momentum (ℏ/μm)')
        ax.set_title('Phase Gradient Profile')
        ax.grid(True)
        
        return fig, ax
    
    def plot_phase_space(self, wf: WaveFunction) -> Tuple[plt.Figure, plt.Axes]:
        """Create phase space (position-momentum) plot."""
        fig, ax = plt.subplots()
        
        # Calculate physical quantities
        x_physical = to_physical(wf.grid, 'length', 1.0) * 1e6
        p = np.gradient(np.unwrap(wf.phase), wf.grid[1] - wf.grid[0])
        p_physical = to_physical(p, 'momentum', 1.0)
        
        # Create scatter plot with density-dependent color
        scatter = ax.scatter(x_physical, p_physical, c=wf.density,
                           cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Density')
        
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Momentum (ℏ/μm)')
        ax.set_title('Phase Space Distribution')
        
        return fig, ax
    
    def detect_phase_singularities(self, wf: WaveFunction, 
                                 threshold: float = 0.1) -> List[float]:
        """
        Detect phase singularities (potential vortices).
        Returns list of positions where singularities are found.
        """
        # Calculate phase gradient
        dx = wf.grid[1] - wf.grid[0]
        phase_gradient = np.gradient(np.unwrap(wf.phase), dx)
        
        # Look for sudden phase jumps
        singularities = []
        for i in range(len(phase_gradient)-1):
            if (abs(phase_gradient[i+1] - phase_gradient[i]) > threshold and 
                wf.density[i] < np.max(wf.density) * threshold):
                singularities.append(wf.grid[i])
        
        return singularities
    
    def plot_phase_correlation(self, wf: WaveFunction, 
                             max_distance: Optional[float] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot phase correlation function."""
        fig, ax = plt.subplots()
        
        if max_distance is None:
            max_distance = (wf.grid[-1] - wf.grid[0]) / 2
            
        # Calculate correlation function
        phase = wf.phase
        correlations = []
        distances = []
        
        center_idx = len(wf.grid) // 2
        for i in range(len(wf.grid) // 2):
            if wf.grid[center_idx + i] - wf.grid[center_idx] > max_distance:
                break
            
            corr = np.cos(phase[center_idx + i] - phase[center_idx])
            correlations.append(corr)
            distances.append(wf.grid[center_idx + i] - wf.grid[center_idx])
        
        # Convert to physical units and plot
        distances_physical = to_physical(np.array(distances), 'length', 1.0) * 1e6
        ax.plot(distances_physical, correlations, 'b-')
        
        ax.set_xlabel('Distance (μm)')
        ax.set_ylabel('Phase Correlation')
        ax.set_title('Phase Correlation Function')
        ax.grid(True)
        
        return fig, ax
    
    def plot_phase_histogram(self, wf: WaveFunction, 
                           bins: int = 50) -> Tuple[plt.Figure, plt.Axes]:
        """Plot histogram of phase values."""
        fig, ax = plt.subplots()
        
        # Create histogram of phase values
        phase = wf.phase % (2 * np.pi)  # Wrap to [0, 2π]
        weights = wf.density / np.sum(wf.density)  # Weight by density
        
        ax.hist(phase, bins=bins, weights=weights, 
                density=True, alpha=0.7, color='blue')
        
        ax.set_xlabel('Phase (rad)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Phase Distribution')
        ax.grid(True)
        
        return fig, ax
    
    def save_all_plots(self, wf: WaveFunction, base_filename: str):
        """Save all phase-related plots to files."""
        # Phase profile
        fig_phase, _ = self.plot_phase_profile(wf)
        fig_phase.savefig(f'{base_filename}_phase_profile.png')
        
        # Phase gradient
        fig_gradient, _ = self.plot_phase_gradient(wf)
        fig_gradient.savefig(f'{base_filename}_phase_gradient.png')
        
        # Phase space
        fig_space, _ = self.plot_phase_space(wf)
        fig_space.savefig(f'{base_filename}_phase_space.png')
        
        # Phase correlation
        fig_corr, _ = self.plot_phase_correlation(wf)
        fig_corr.savefig(f'{base_filename}_phase_correlation.png')
        
        # Phase histogram
        fig_hist, _ = self.plot_phase_histogram(wf)
        fig_hist.savefig(f'{base_filename}_phase_histogram.png')
        
        plt.close('all')