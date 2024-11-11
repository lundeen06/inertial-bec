import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple
import matplotlib.animation as animation
from dataclasses import dataclass
from ..core.bec_state import BECState
from ..utils.numerical_methods import GPESolver


@dataclass
class BECAnimator:
    """Class for creating animations of BEC evolution."""
    
    def create_evolution_animation(self,
                             times: np.ndarray,
                             states: List[np.ndarray],
                             interval: int = 50,
                             fig: Optional[plt.Figure] = None,
                             mode: str = 'density') -> animation.Animation:
        """
        Create animation of BEC evolution.
        
        Args:
            times: Array of time points
            states: List of wavefunctions at each time
            interval: Time between frames in milliseconds
            fig: Optional figure to use
            mode: Type of visualization ('density', 'phase', or 'both')
        """
        # Input validation
        if not states or len(states) == 0:
            raise ValueError("States list cannot be empty")
        if len(times) != len(states):
            raise ValueError(f"Mismatch between times ({len(times)}) and states ({len(states)})")
        
        # Create figure if not provided
        if fig is None:
            if mode == 'both':
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            else:
                fig, ax1 = plt.subplots(figsize=(8, 5))
                ax2 = None
        
        # Initialize plot lines
        grid = np.linspace(-10, 10, len(states[0]))
        density_line, = ax1.plot([], [], 'b-', label='Density')
        
        # Store fill artist for proper removal
        fill_artist = ax1.fill_between(grid, np.zeros_like(grid), alpha=0.3, color='blue')
        
        if mode == 'both':
            phase_line, = ax2.plot([], [], 'r-', label='Phase')
        
        # Set axis properties
        max_density = max([np.max(np.abs(psi)**2) for psi in states])
        ax1.set_xlim(grid.min(), grid.max())
        ax1.set_ylim(0, 1.2 * max_density)
        ax1.set_xlabel('Position (trap units)')
        ax1.set_ylabel('Density')
        ax1.grid(True)
        
        if mode == 'both':
            ax2.set_xlim(grid.min(), grid.max())
            ax2.set_ylim(-np.pi, np.pi)
            ax2.set_xlabel('Position (trap units)')
            ax2.set_ylabel('Phase (rad)')
            ax2.grid(True)
        
        # Time display
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
        
        def safe_remove_artist(artist):
            """Safely remove an artist if it exists in the axes"""
            try:
                artist.remove()
            except:
                pass
        
        def init():
            """Initialize animation"""
            density_line.set_data([], [])
            safe_remove_artist(fill_artist)
            
            if mode == 'both':
                phase_line.set_data([], [])
                
            time_text.set_text('')
            artists = [density_line, time_text]
            if mode == 'both':
                artists.append(phase_line)
            return artists
        
        def animate(frame):
            """Animation function"""
            try:
                # Update density plot
                density = np.abs(states[frame])**2
                density_line.set_data(grid, density)
                
                # Update fill_between
                for coll in ax1.collections:
                    safe_remove_artist(coll)
                ax1.fill_between(grid, density, alpha=0.3, color='blue')
                
                if mode == 'both':
                    # Update phase plot
                    phase = np.angle(states[frame])
                    phase_line.set_data(grid, phase)
                
                # Update time display
                time_text.set_text(f'Time: {times[frame]*1e3:.1f} ms')
                
                artists = [density_line, time_text]
                if mode == 'both':
                    artists.append(phase_line)
                return artists
                
            except Exception as e:
                print(f"Error in animation frame {frame}: {str(e)}")
                return []
        
        try:
            anim = FuncAnimation(
                fig=fig,
                func=animate,
                init_func=init,
                frames=len(times),
                interval=interval,
                blit=True,
                repeat=False  # Add this if you don't want the animation to loop
            )
            
            plt.close(fig)  # Prevent display of static figure
            return anim
            
        except Exception as e:
            print(f"Error creating animation: {str(e)}")
            plt.close(fig)
            raise
    def create_phase_space_animation(self,
                                   times: np.ndarray,
                                   states: List[np.ndarray],
                                   interval: int = 50,
                                   momentum_spread: float = 0.1) -> animation.Animation:
        """
        Create animation of phase space evolution.
        
        Args:
            times: Array of time points
            states: List of wavefunctions at each time
            interval: Time between frames in milliseconds
            momentum_spread: Width of momentum uncertainty
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        grid = np.linspace(-10, 10, len(states[0]))
        dx = grid[1] - grid[0]
        
        # Set up scatter plot
        scatter = ax.scatter([], [], c=[], cmap='viridis', alpha=0.6, s=1)
        mean_line, = ax.plot([], [], 'r--', alpha=0.5, label='Mean trajectory')
        
        # Set axis properties
        max_pos = np.max(np.abs(grid))
        max_mom = 2.0  # Estimated maximum momentum
        limit = max(max_pos, max_mom) * 1.1
        
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add uncertainty principle guidelines
        x_range = np.linspace(-limit, limit, 100)
        uncertainty_bound = 0.5/np.abs(x_range)
        ax.plot(x_range, uncertainty_bound, 'k:', alpha=0.3, label='ΔxΔp = ℏ/2')
        ax.plot(x_range, -uncertainty_bound, 'k:', alpha=0.3)
        
        ax.set_xlabel('Position (trap units)')
        ax.set_ylabel('Momentum (ℏ/a₀)')
        ax.legend()
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            scatter.set_offsets(np.zeros((0, 2)))
            scatter.set_array(np.array([]))
            mean_line.set_data([], [])
            time_text.set_text('')
            return [scatter, mean_line, time_text]
        
        def animate(i):
            # Calculate phase space distribution
            phase = np.unwrap(np.angle(states[i]))
            p_mean = np.gradient(phase, dx)
            density = np.abs(states[i])**2
            
            # Create points with quantum spread
            n_points = 1000
            x_points = []
            p_points = []
            weights = []
            
            for x, p, rho in zip(grid, p_mean, density):
                if rho > 1e-6 * np.max(density):
                    n_local = int(n_points * rho / np.max(density))
                    x_local = np.random.normal(x, 0.1, n_local)
                    p_local = np.random.normal(p, momentum_spread, n_local)
                    
                    x_points.extend(x_local)
                    p_points.extend(p_local)
                    weights.extend([rho/np.max(density)] * n_local)
            
            # Update plots
            points = np.column_stack((x_points, p_points))
            scatter.set_offsets(points)
            scatter.set_array(np.array(weights))
            
            mean_line.set_data(grid, p_mean)
            time_text.set_text(f'Time: {times[i]*1e3:.1f} ms')
            
            return [scatter, mean_line, time_text]
        
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(times), interval=interval,
                           blit=True)
        
        plt.close(fig)
        return anim
    
    def create_density_phase_animation(self,
                                     bec: BECState,
                                     times: np.ndarray,
                                     solver: 'GPESolver',
                                     potential: np.ndarray,
                                     interval: int = 50) -> animation.Animation:
        """
        Create real-time animation during BEC evolution.
        
        Args:
            bec: Initial BEC state
            times: Time points for evolution
            solver: GPE solver instance
            potential: Trap potential
            interval: Time between frames in milliseconds
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Initialize plots
        grid = bec._grid
        density_line, = ax1.plot([], [], 'b-', label='Density')
        phase_line, = ax2.plot([], [], 'r-', label='Phase')
        
        # Set axis properties
        ax1.set_xlim(grid.min(), grid.max())
        ax1.set_ylim(0, 1.2 * np.max(bec.density))
        ax1.set_xlabel('Position (trap units)')
        ax1.set_ylabel('Density')
        ax1.grid(True)
        
        ax2.set_xlim(grid.min(), grid.max())
        ax2.set_ylim(-np.pi, np.pi)
        ax2.set_xlabel('Position (trap units)')
        ax2.set_ylabel('Phase (rad)')
        ax2.grid(True)
        
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
        
        def animate(frame):
            t = times[frame]
            
            # Evolve state
            solver.split_step(bec._wavefunction, potential)
            
            # Update plots
            density_line.set_data(grid, bec.density)
            phase_line.set_data(grid, np.angle(bec._wavefunction))
            time_text.set_text(f'Time: {t*1e3:.1f} ms')
            
            return [density_line, phase_line, time_text]
        
        anim = FuncAnimation(fig, animate, frames=len(times),
                           interval=interval, blit=True)
        
        plt.close(fig)
        return anim