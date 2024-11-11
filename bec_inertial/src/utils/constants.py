import numpy as np

# Physical constants (SI units)
HBAR = 1.054571817e-34  # Reduced Planck constant (J⋅s)
KB = 1.380649e-23       # Boltzmann constant (J/K)
M_RB87 = 1.443160648e-25  # Mass of Rb87 atom (kg)

# Characteristic scales for the simulation
CHARACTERISTIC_LENGTH = 1e-6  # 1 μm
CHARACTERISTIC_TIME = 1e-3    # 1 ms
CHARACTERISTIC_ENERGY = HBAR * 2 * np.pi * 1e3  # Based on 1kHz frequency

# Derived units
CHARACTERISTIC_VELOCITY = CHARACTERISTIC_LENGTH / CHARACTERISTIC_TIME
CHARACTERISTIC_ACCELERATION = CHARACTERISTIC_LENGTH / (CHARACTERISTIC_TIME ** 2)

# Numerical parameters
SPATIAL_POINTS = 256  # Number of points in spatial grid
TEMPORAL_POINTS = 1000  # Number of points in time grid
SAFETY_FACTOR = 0.9  # For adaptive time stepping

# Trap parameters (defaults)
DEFAULT_TRAP_FREQUENCY = 100.0  # Hz
DEFAULT_INTERACTION_STRENGTH = 1.0  # Dimensionless g
DEFAULT_ATOM_NUMBER = 1e5

def dimensionless_coupling(scattering_length, atom_number, trap_frequency):
    """Calculate dimensionless coupling constant for the GPE."""
    omega = 2 * np.pi * trap_frequency
    osc_length = np.sqrt(HBAR / (M_RB87 * omega))
    return 4 * np.pi * scattering_length * atom_number / osc_length

def characteristic_scales(trap_frequency):
    """Return characteristic scales for given trap frequency."""
    omega = 2 * np.pi * trap_frequency
    length_scale = np.sqrt(HBAR / (M_RB87 * omega))
    time_scale = 1 / omega
    energy_scale = HBAR * omega
    return length_scale, time_scale, energy_scale

def to_dimensionless(value, unit_type, trap_frequency):
    """Convert physical values to dimensionless units."""
    l_scale, t_scale, e_scale = characteristic_scales(trap_frequency)
    
    conversions = {
        'length': l_scale,
        'time': t_scale,
        'energy': e_scale,
        'velocity': l_scale / t_scale,
        'acceleration': l_scale / (t_scale ** 2)
    }
    
    return value / conversions[unit_type]

def to_physical(value, unit_type, trap_frequency):
    """Convert dimensionless units to physical values."""
    l_scale, t_scale, e_scale = characteristic_scales(trap_frequency)
    
    conversions = {
        'length': l_scale,
        'time': t_scale,
        'energy': e_scale,
        'velocity': l_scale / t_scale,
        'acceleration': l_scale / (t_scale ** 2)
    }
    
    return value * conversions[unit_type]