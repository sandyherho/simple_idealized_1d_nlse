# Simple Idealized 1D NLSE Solver

[![DOI](https://zenodo.org/badge/1038236999.svg)](https://doi.org/10.5281/zenodo.16895284)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/simple-idealized-1d-nlse.svg)](https://pypi.org/project/simple-idealized-1d-nlse/)

A flexible and efficient solver for the 1D Nonlinear Schrödinger Equation (NLSE) using pseudo-spectral methods

## Overview

This package provides a high-accuracy numerical framework for solving the 1D focusing NLSE:

$$i \frac{\partial \psi}{\partial t} + \frac{1}{2} \frac{\partial^2 \psi}{\partial x^2 } + |\psi|^2 \psi = 0,$$

where $\psi(x,t)$ is the complex wave function. This equation models various physical phenomena including Bose-Einstein condensates, nonlinear optics, and water waves.

### Numerical Method

The solver employs a **Fourier pseudo-spectral method** combined with high-order time integration:

#### Spatial Discretization
Using the Fourier transform $\hat{\psi}(k,t) = \mathcal{F}[\psi(x,t)]$, the NLSE becomes:

$$\frac{\partial \hat{\psi}}{\partial t} = -\frac{i}{2}k^2\hat{\psi} + i\mathcal{F}[|\psi|^2\psi],$$

where spatial derivatives are computed spectrally:
- $\mathcal{F}[\partial^2_x \psi] = -k^2 \hat{\psi}$
- Wavenumbers: $k_j = 2\pi j/L$ for $j \in [-N/2, N/2)$

#### Anti-Aliasing Filter
To prevent aliasing from the nonlinear term, we apply an exponential filter (2/3 rule):

$$\sigma(k) = \exp\left[-36\left(\frac{|k|}{k_{max}}\right)^{36}\right], \quad k_{max} = \frac{2\pi N}{3L}$$

#### Time Integration
The filtered equation in Fourier space:

$$\frac{d\hat{\psi}_f}{dt} = -\frac{i}{2}k^2\hat{\psi}_f + i\sigma(k)\mathcal{F}[|\mathcal{F}^{-1}[\hat{\psi}_f]|^2 \mathcal{F}^{-1}[\hat{\psi}_f]]$$

is solved using the **8th-order Dormand-Prince method (DOP853)** with adaptive time-stepping, achieving relative tolerances down to $10^{-9}$.

### Conservation Laws

The solver monitors three conserved quantities to ensure numerical stability:

- **Mass (L² norm)**: $M = \int_{-\infty}^{\infty} |\psi|^2 dx$

- **Momentum**: $P = \int_{-\infty}^{\infty} \text{Im}(\psi^* \partial_x \psi) dx$

- **Energy (Hamiltonian)**: $E = \int_{-\infty}^{\infty} \left[\frac{1}{2}|\partial_x \psi|^2 - \frac{1}{2}|\psi|^4\right] dx$

### Key Features

- **Spectral accuracy**: Exponential convergence for smooth solutions
- **Adaptive time-stepping**: Automatic step size control based on error estimates
- **JIT compilation**: Optional Numba acceleration for performance-critical sections
- **Stability monitoring**: Real-time tracking of conservation laws
- **Multiple initial conditions**: Solitons, breathers, and modulation instability scenarios

## Installation

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

## Quick Start

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

## Project Structure

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


## Authors

- **Sandy H. S. Herho** - sandy.herho@email.ucr.edu
- **Iwan P. Anwar** 
- **Faruq Khadami** 
- **Edi Riawan** 
- **Rusmawan Suwarman** 
- **Dasapta E. Irawan** 

## License

This project is licensed under the WTFPL License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{nlse_solver_2025,
  title = {Simple Idealized 1D NLSE Solver},
  author = {Herho, Sandy H. S. and Anwar, Iwan P. and Khadami, Faruq and 
            Riawan, Edi and Suwarman, Rusmawan and Irawan, Dasapta E.},
  year = {2025},
  version = {0.0.3},
  url = {https://github.com/sandyherho/simple_idealized_1d_nlse}
}
