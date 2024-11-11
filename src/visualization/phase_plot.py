import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List
from dataclasses import dataclass
from ..core.bec_state import BECState

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
    
    def plot_phase_profile(self, bec: BECState, 
                          ax: Optional[plt.Axes] = None,
                          title: str = "Phase Profile") -> Tuple[plt.Figure, plt.Axes]:
        """Plot the phase profile of the BEC."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
            ax.clear()
        
        # Get grid and phase
        grid = bec._grid
        phase = np.unwrap(np.angle(bec._wavefunction))
        
        # Plot phase
        ax.plot(grid, phase, 'b-', label='Phase')
        
        # Add density overlay for reference (scaled)
        density = bec.density / np.max(bec.density) * np.pi
        ax.plot(grid, density, 'r--', alpha=0.5, label='Normalized Density')
        
        ax.set_xlabel('Position (trap units)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        
        return fig, ax
    
    def plot_phase_space(self, bec: BECState,
                        ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create phase space (position-momentum) plot."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
            ax.clear()
        
        # Calculate quantities
        grid = bec._grid
        phase = np.unwrap(np.angle(bec._wavefunction))
        p = np.gradient(phase, grid[1] - grid[0])
        
        # Create scatter plot with density-dependent color
        scatter = ax.scatter(grid, p, c=bec.density,
                           cmap='viridis', alpha=0.6)
        fig.colorbar(scatter, ax=ax, label='Density')
        
        ax.set_xlabel('Position (trap units)')
        ax.set_ylabel('Momentum (ℏ/a₀)')
        ax.set_title('Phase Space Distribution')
        
        return fig, ax
    
    def plot_phase_correlation(self, bec: BECState,
                             max_distance: Optional[float] = None,
                             ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot phase correlation function."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
            ax.clear()
            
        if max_distance is None:
            max_distance = (bec._grid[-1] - bec._grid[0]) / 2
            
        # Calculate correlation function
        phase = np.angle(bec._wavefunction)
        correlations = []
        distances = []
        
        center_idx = len(bec._grid) // 2
        for i in range(len(bec._grid) // 2):
            if bec._grid[center_idx + i] - bec._grid[center_idx] > max_distance:
                break
            
            corr = np.cos(phase[center_idx + i] - phase[center_idx])
            correlations.append(corr)
            distances.append(bec._grid[center_idx + i] - bec._grid[center_idx])
        
        ax.plot(distances, correlations, 'b-')
        ax.set_xlabel('Distance (trap units)')
        ax.set_ylabel('Phase Correlation')
        ax.set_title('Phase Correlation Function')
        ax.grid(True)
        
        return fig, ax
    
    def plot_phase_histogram(self, bec: BECState,
                           bins: int = 50,
                           ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot histogram of phase values."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
            ax.clear()
        
        # Create histogram of phase values
        phase = np.angle(bec._wavefunction) % (2 * np.pi)
        weights = bec.density / np.sum(bec.density)
        
        ax.hist(phase, bins=bins, weights=weights, 
                density=True, alpha=0.7, color='blue')
        
        ax.set_xlabel('Phase (rad)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Phase Distribution')
        ax.grid(True)
        
        return fig, ax
    
    def detect_phase_singularities(self, bec: BECState,
                                 threshold: float = 0.1) -> List[float]:
        """
        Detect phase singularities (potential vortices).
        Returns list of positions where singularities are found.
        """
        # Calculate phase gradient
        dx = bec._grid[1] - bec._grid[0]
        phase = np.angle(bec._wavefunction)
        phase_gradient = np.gradient(np.unwrap(phase), dx)
        
        # Look for sudden phase jumps
        singularities = []
        for i in range(len(phase_gradient)-1):
            if (abs(phase_gradient[i+1] - phase_gradient[i]) > threshold and 
                bec.density[i] < np.max(bec.density) * threshold):
                singularities.append(bec._grid[i])
        
        return singularities
    
    def save_all_plots(self, bec: BECState, base_filename: str):
        """Save all phase-related plots to files."""
        # Phase profile
        fig_phase, _ = self.plot_phase_profile(bec)
        fig_phase.savefig(f'{base_filename}_phase_profile.png')
        
        # Phase space
        fig_space, _ = self.plot_phase_space(bec)
        fig_space.savefig(f'{base_filename}_phase_space.png')
        
        # Phase correlation
        fig_corr, _ = self.plot_phase_correlation(bec)
        fig_corr.savefig(f'{base_filename}_phase_correlation.png')
        
        # Phase histogram
        fig_hist, _ = self.plot_phase_histogram(bec)
        fig_hist.savefig(f'{base_filename}_phase_histogram.png')
        
        plt.close('all')