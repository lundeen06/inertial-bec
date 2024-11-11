# BEC Inertial Sensor Simulation
A Python-based simulation framework for modeling Bose-Einstein Condensate (BEC) based inertial sensors, with applications in inertial navigation systems in high-precision vehicles.

## Overview
This project implements numerical simulations of BEC dynamics under acceleration, solving the time-dependent Gross-Pitaevskii equation with external potentials. The system models how a trapped BEC responds to inertial forces, providing high-precision acceleration measurements.

## Project Structure

```mermaid
graph TD
    A[BEC Inertial] --> B[src]
    A --> C[tests]
    A --> D[notebooks]
    A --> E[docs]
    A --> F[config]
    A --> G[data]
    
    B --> H[core]
    B --> I[physics]
    B --> J[visualization]
    B --> K[utils]
    
    H --> L[bec_state.py]
    H --> M[wave_function.py]
    H --> N[density_profile.py]
    
    I --> O[hamiltonian.py]
    I --> P[trap_potential.py]
    I --> Q[interactions.py]
    I --> R[acceleration.py]
    
    J --> S[density_plot.py]
    J --> T[phase_plot.py]
    J --> U[animation.py]
```

## System Architecture

```mermaid
flowchart LR
    A[Input Parameters] --> B[BEC State]
    B --> C{Physics Engine}
    C --> D[Trap Potential]
    C --> E[Interactions]
    C --> F[Acceleration]
    D & E & F --> G[Time Evolution]
    G --> H[Visualization]
    G --> I[Analysis]
    H & I --> J[Output Metrics]
```

## Data Flow

```mermaid
sequenceDiagram
    participant Config
    participant BECState
    participant Physics
    participant Visualizer
    participant Analysis
    
    Config->>BECState: Initialize Parameters
    loop Time Evolution
        BECState->>Physics: Current State
        Physics->>Physics: Compute Forces
        Physics->>BECState: Update State
        BECState->>Visualizer: State Data
        BECState->>Analysis: Metrics
    end
    Analysis->>Analysis: Process Results
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bec_inertial.git
cd bec_inertial

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from src.core.bec_state import BECState
from src.physics.acceleration import AccelerationSensor

# Initialize BEC state
bec = BECState(atoms=1e5, trap_frequency=100.0)

# Create sensor
sensor = AccelerationSensor(bec)

# Run simulation
results = sensor.simulate(duration=1.0, dt=0.001)
```

## Features

1. **Core Simulation**
   - Time-dependent Gross-Pitaevskii equation solver
   - Adaptive step size integration
   - Conservation law monitoring

2. **Physics Modules**
   - Harmonic trap potential
   - Mean-field interactions
   - External acceleration forces
   - Quantum pressure terms

3. **Analysis Tools**
   - Phase space analysis
   - Density profile extraction
   - Response function calculation
   - Sensitivity metrics

4. **Visualization**
   - Real-time density plots
   - Phase evolution
   - Animation capabilities
   - Metric tracking

## Configuration

Simulation parameters are stored in `config/simulation_params.yaml`. Example configuration:

```yaml
simulation:
  atoms: 1e5
  trap_frequency: 100.0
  interaction_strength: 1.0
  grid_points: 256
  box_size: 20.0
  
physics:
  acceleration_range: [-10.0, 10.0]
  measurement_time: 1.0
  temperature: 1e-9
```

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Testing

Run the test suite:

```bash
pytest tests/
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/bec_inertial

## Acknowledgments

- References to key BEC papers and theoretical frameworks
- Credits to numerical methods used
- Any other acknowledgments

## References

1. Key paper on BEC-based inertial sensing
2. Gross-Pitaevskii equation numerical methods
3. Relevant experimental implementations