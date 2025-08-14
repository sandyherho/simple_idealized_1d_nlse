"""
Simple Idealized 1D NLSE Solver
================================

A flexible and efficient solver for the 1D Nonlinear Schrödinger Equation.

Version: 0.0.1
Date: 2025-08-13
Authors: Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Fikry P. Lugina,
         Rusmawan Suwarman, Dasapta E. Irawan
License: WTFPL
"""

__version__ = "0.0.1"
__author__ = "Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Fikry P. Lugina, Rusmawan Suwarman, Dasapta E. Irawan"
__email__ = "sandy.herho@email.ucr.edu"
__institution__ = "Samudera Sains Teknologi Ltd."
__license__ = "WTFPL"

from .core.solver import NLSESolver
from .core.initial_conditions import (
    SingleSoliton,
    TwoSoliton,
    BreatherSolution,
    GaussianNoise
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .utils.logger import SimulationLogger
from .utils.timer import Timer

__all__ = [
    "NLSESolver",
    "SingleSoliton",
    "TwoSoliton",
    "BreatherSolution",
    "GaussianNoise",
    "ConfigManager",
    "DataHandler",
    "SimulationLogger",
    "Timer",
]

def print_version():
    """Print version information."""
    print(f"Simple Idealized 1D NLSE Solver v{__version__}")
    print(f"Institution: {__institution__}")
    print(f"Authors: {__author__}")
