"""
Core Module - NLSE Solver Components
=====================================

Date: 2025-08-13
Authors: Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Edi Riawan,
         Rusmawan Suwarman, Dasapta E. Irawan
"""

from .solver import NLSESolver
from .initial_conditions import *

__all__ = ["NLSESolver", "SingleSoliton", "TwoSoliton", "BreatherSolution", "GaussianNoise"]
