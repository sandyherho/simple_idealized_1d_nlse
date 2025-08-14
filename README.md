# Simple Idealized 1D NLSE Solver

[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)


A flexible and efficient solver for the 1D Nonlinear Schrödinger Equation (NLSE)

## 📋 Overview

This package provides a comprehensive framework for solving the 1D NLSE:

$ i \frac{\partial \psi}{\partial t} + \frac{1}{2} \frac{\partial^2 \psi}{\partial x^2 } + |\psi|^2 \psi = 0,$

where $\psi$ is the complex wave function.

## 🚀 Features

- **Multiple Scenarios**: Single soliton, two-soliton collision, breather solutions, modulation instability
- **Enhanced Numerical Stability**: DOP853 solver, adaptive time-stepping, conservation law monitoring
- **High Performance**: Optional Numba JIT compilation for speed
- **Flexible Configuration**: Support for both YAML and TXT configuration files
- **Progress Tracking**: Real-time progress bars with tqdm
- **Comprehensive Logging**: Detailed logs for each simulation
- **Professional Visualization**: FiveThirtyEight-style animations
- **Data Management**: NetCDF output format with institutional metadata
- **Conservation Monitoring**: Real-time tracking of conserved quantities

## 📦 Installation

### From PyPI

```bash
pip install simple-idealized-1d-nlse
```


### From source

```bash
git clone https://github.com/samuderasains/simple-idealized-1d-nlse.git
cd simple-idealized-1d-nlse
pip install -e .
```

## 🎯 Quick Start

### Command Line Interface

```bash
# Run single scenario with YAML config
nlse-simulate single_soliton

# Run with TXT configuration file
nlse-simulate --config configs/txt/single_soliton.txt

# Run all predefined scenarios
nlse-simulate --all

# Run with verbose output
nlse-simulate single_soliton --verbose
```

### Python API

```python
from simple_idealized_1d_nlse import NLSESolver, SingleSoliton

# Create solver with enhanced stability
solver = NLSESolver(
    domain_length=50, 
    num_points=512, 
    use_adaptive=True,  # Adaptive stepping
    verbose=True
)

# Set initial condition
initial = SingleSoliton(amplitude=2.0, position=0.0, velocity=1.0)

# Run simulation with monitoring
result = solver.solve(
    initial_condition=initial,
    t_final=20.0,
    n_snapshots=100,
    monitor_conservation=True,
    show_progress=True
)

# Results include wave function ψ
print(f"Max |ψ|: {result['psi_abs'].max():.4f}")
print(f"Conservation error: {result['conservation']['conservation_error']:.2e}")
```

## 📁 Project Structure

```
simple_idealized_1d_nlse/
├── src/simple_idealized_1d_nlse/
│   ├── core/           # Core solver and numerical methods
│   ├── utils/          # Utility functions and helpers
│   ├── visualization/  # FiveThirtyEight-style plotting
│   └── io/            # Input/output handlers (YAML/TXT)
├── configs/           
│   ├── yaml/          # YAML configuration files
│   └── txt/           # TXT configuration files

../outputs/            # Simulation results (outside package)
../logs/              # Simulation logs (outside package)
```


## 👥 Authors

- **Sandy H. S. Herho** - sandy.herho@email.ucr.edu
- **Iwan P. Anwar** 
- **Faruq Khadami** 
- **Fikry P. Lugina** 
- **Rusmawan Suwarman** 
- **Dasapta E. Irawan** 

## 📄 License

This project is licensed under the WTFPL License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{nlse_solver_2025,
  title = {Simple Idealized 1D NLSE Solver},
  author = {Herho, Sandy H. S. and Anwar, Iwan P. and Khadami, Faruq and 
            Lugina, Fikry P. and Suwarman, Rusmawan and Irawan, Dasapta E.},
  year = {2025},
  version = {0.0.1},
  institution = {Samudera Sains Teknologi Ltd.},
  url = {https://github.com/sandyherho/simple_idealized_1d_nlse}
}
```
