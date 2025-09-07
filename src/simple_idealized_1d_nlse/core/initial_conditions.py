"""
Initial Conditions for 1D NLSE
=========================================

Date: 2025-08-13
Authors: Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami,
         Rusmawan Suwarman, Dasapta E. Irawan
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class InitialCondition(ABC):
    """Abstract base class for initial wave function ψ₀."""
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the initial wave function at given spatial points."""
        pass


def sech(x: np.ndarray) -> np.ndarray:
    """Hyperbolic secant function."""
    return 1.0 / np.cosh(x)


class SingleSoliton(InitialCondition):
    """Single soliton initial condition: ψ(x,0) = A * sech(A*(x - x₀)) * exp(i*v*x)"""
    
    def __init__(self, amplitude: float = 2.0, position: float = 0.0, velocity: float = 1.0):
        self.amplitude = amplitude
        self.position = position
        self.velocity = velocity
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        envelope = self.amplitude * sech(self.amplitude * (x - self.position))
        phase = np.exp(1j * self.velocity * x)
        return envelope * phase


class TwoSoliton(InitialCondition):
    """Two-soliton collision initial condition: ψ(x,0) = ψ₁(x,0) + ψ₂(x,0)"""
    
    def __init__(self, A1: float = 2.0, A2: float = 1.5,
                 x1: float = -10.0, x2: float = 10.0,
                 v1: float = 2.0, v2: float = -2.0):
        self.soliton1 = SingleSoliton(A1, x1, v1)
        self.soliton2 = SingleSoliton(A2, x2, v2)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.soliton1(x) + self.soliton2(x)


class BreatherSolution(InitialCondition):
    """Breather solution (Akhmediev breather approximation)"""
    
    def __init__(self, amplitude: float = 1.0, frequency: float = 0.5, modulation: float = 0.5):
        self.amplitude = amplitude
        self.frequency = frequency
        self.modulation = modulation
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        envelope = self.amplitude * np.sqrt(2) * sech(self.amplitude * x)
        phase = np.exp(1j * self.frequency * x)
        modulation = 1 + self.modulation * np.cos(2 * self.frequency * x)
        return envelope * phase * modulation


class GaussianNoise(InitialCondition):
    """Gaussian profile with small noise perturbation: ψ(x,0) = A * exp(-x²/2σ²) + noise"""
    
    def __init__(self, amplitude: float = 1.0, width: float = 5.0,
                 center: float = 0.0, noise_level: float = 0.01,
                 seed: Optional[int] = None):
        self.amplitude = amplitude
        self.width = width
        self.center = center
        self.noise_level = noise_level
        self.seed = seed
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)
        
        gaussian = self.amplitude * np.exp(-(x - self.center)**2 / (2 * self.width**2))
        noise = self.noise_level * (np.random.randn(len(x)) + 1j * np.random.randn(len(x)))
        return gaussian + noise
