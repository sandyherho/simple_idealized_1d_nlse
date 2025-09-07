"""
1D NLSE Solver Core Implementation with Enhanced Numerical Stability
==================================================================

Solves: i∂ψ/∂t + (1/2)∂²ψ/∂x² + |ψ|²ψ = 0

Date: 2025-08-13
Authors: Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami,
         Rusmawan Suwarman, Dasapta E. Irawan
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, Dict, Any
import warnings
import time
from tqdm import tqdm
import sys

# Handle colorama import
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Create dummy classes for colors
    class Fore:
        GREEN = YELLOW = RED = CYAN = MAGENTA = BLUE = ""
        RESET_ALL = ""
    class Style:
        RESET_ALL = BRIGHT = DIM = NORMAL = ""

# Handle numba import
try:
    from numba import jit
    import numba
    NUMBA_AVAILABLE = True
    # Check numba version for compatibility
    NUMBA_VERSION = tuple(map(int, numba.__version__.split('.')[:2]))
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_VERSION = (0, 0)
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class NLSESolver:
    """
    Numerically stable solver for the 1D Nonlinear Schrödinger Equation.
    
    The NLSE is given by:
        i∂ψ/∂t + (1/2)∂²ψ/∂x² + |ψ|²ψ = 0
    
    where ψ is the complex wave function.
    
    This solver uses:
    - Spectral methods (FFT) for spatial derivatives
    - Adaptive time-stepping for temporal evolution
    - Conservation law monitoring for stability
    - Anti-aliasing filtering
    
    Attributes
    ----------
    domain_length : float
        Total length of the spatial domain
    num_points : int
        Number of spatial grid points
    use_numba : bool
        Whether to use Numba JIT compilation
    use_adaptive : bool
        Use adaptive error control
    verbose : bool
        Enable verbose output with progress tracking
    logger : SimulationLogger
        Logger instance for detailed logging
    """
    
    def __init__(
        self,
        domain_length: float = 50.0,
        num_points: int = 512,
        use_numba: bool = False,
        use_adaptive: bool = True,
        use_filtering: bool = True,
        verbose: bool = True,
        logger: Optional[Any] = None
    ):
        """
        Initialize the 1D NLSE solver with enhanced stability features.
        
        Parameters
        ----------
        domain_length : float, optional
            Total length of the spatial domain (default: 50.0)
        num_points : int, optional
            Number of spatial grid points (default: 512)
        use_numba : bool, optional
            Enable Numba JIT compilation if available (default: False)
        use_adaptive : bool, optional
            Use adaptive error control (default: True)
        use_filtering : bool, optional
            Apply anti-aliasing filter (default: True)
        verbose : bool, optional
            Enable verbose output (default: True)
        logger : SimulationLogger, optional
            Logger instance for detailed logging
        """
        self.L = domain_length
        self.M = num_points
        self.dx = domain_length / num_points
        self.verbose = verbose
        self.logger = logger
        self.use_adaptive = use_adaptive
        self.use_filtering = use_filtering
        
        # Spatial grid
        self.x = np.linspace(-self.L/2, self.L/2, self.M, endpoint=False)
        
        # Wavenumber grid for spectral derivatives
        self.k = 2 * np.pi * np.fft.fftfreq(self.M, d=self.dx)
        
        # Anti-aliasing filter (2/3 rule)
        if self.use_filtering:
            k_max = 2 * np.pi * self.M / (3 * self.L)
            self.filter = np.exp(-36 * (np.abs(self.k) / k_max) ** 36)
        else:
            self.filter = np.ones_like(self.k)
        
        # Setup Numba if requested and available
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        if use_numba and not NUMBA_AVAILABLE:
            if self.verbose:
                print(f"{Fore.YELLOW}⚠ Numba requested but not available. Using pure Python.{Style.RESET_ALL}")
            if self.logger:
                self.logger.warning("Numba requested but not available. Using pure Python.")
            self.use_numba = False
        
        # Test Numba compatibility with FFT if requested
        if self.use_numba:
            try:
                # Try to create and test the Numba function
                test_func = self._create_numba_rhs()
                test_arr = np.ones(self.M, dtype=complex)
                # Try a test evaluation
                _ = test_func(0.0, test_arr, self.k)
                if self.verbose:
                    print(f"{Fore.GREEN}✓ Numba JIT compilation successful{Style.RESET_ALL}")
            except Exception as e:
                if self.verbose:
                    print(f"{Fore.YELLOW}⚠ Numba FFT compatibility issue detected. Using pure Python mode.{Style.RESET_ALL}")
                    print(f"  Reason: {str(e)[:100]}...")
                if self.logger:
                    self.logger.warning(f"Numba disabled due to: {str(e)}")
                self.use_numba = False
        
        # Select appropriate RHS function
        if self.use_numba:
            self._rhs_func = self._create_numba_rhs()
        else:
            self._rhs_func = self._nlse_rhs_pure
        
        # Print initialization info
        if self.verbose:
            self._print_solver_info()
        
        if self.logger:
            self.logger.info(f"Initialized NLSE solver: L={self.L}, M={self.M}, "
                           f"Numba={'enabled' if self.use_numba else 'disabled'}, "
                           f"Adaptive={'enabled' if self.use_adaptive else 'disabled'}, "
                           f"Filtering={'enabled' if self.use_filtering else 'disabled'}")
    
    def _print_solver_info(self):
        """Print solver configuration information."""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}NLSE Solver Configuration{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"  Domain length: {self.L:.2f}")
        print(f"  Grid points: {self.M}")
        print(f"  Grid spacing: {self.dx:.4f}")
        print(f"  Nyquist wavenumber: {np.pi/self.dx:.2f}")
        print(f"  Numba JIT: {Fore.GREEN if self.use_numba else Fore.RED}"
              f"{'Enabled' if self.use_numba else 'Disabled'}{Style.RESET_ALL}")
        print(f"  Adaptive control: {Fore.GREEN if self.use_adaptive else Fore.YELLOW}"
              f"{'Enabled' if self.use_adaptive else 'Disabled'}{Style.RESET_ALL}")
        print(f"  Anti-aliasing: {Fore.GREEN if self.use_filtering else Fore.YELLOW}"
              f"{'Enabled' if self.use_filtering else 'Disabled'}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    def _nlse_rhs_pure(self, t: float, psi_t: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Pure Python implementation of NLSE right-hand side in Fourier space.
        
        Parameters
        ----------
        t : float
            Current time
        psi_t : np.ndarray
            Wave function ψ in Fourier space
        k : np.ndarray
            Wavenumber array
        
        Returns
        -------
        np.ndarray
            RHS of the NLSE in Fourier space
        """
        # Apply filter to prevent aliasing
        psi_t_filtered = psi_t * self.filter
        
        # Transform to physical space
        psi = np.fft.ifft(psi_t_filtered)
        
        # Compute nonlinear term: |ψ|²ψ
        nonlinear = (np.abs(psi)**2) * psi
        
        # Transform nonlinear term to Fourier space
        nonlinear_t = np.fft.fft(nonlinear)
        
        # Apply filter again
        nonlinear_t_filtered = nonlinear_t * self.filter
        
        # Combine linear and nonlinear terms
        # i∂ψ/∂t = -(1/2)k²ψ - i|ψ|²ψ
        rhs = -(1j/2) * (k**2) * psi_t_filtered + 1j * nonlinear_t_filtered
        
        return rhs
    
    def _create_numba_rhs(self):
        """
        Create Numba-compiled RHS function with proper FFT handling.
        
        This version handles the FFT compatibility issue by using object mode
        for FFT operations while still getting some speedup from Numba.
        """
        filter_arr = self.filter
        
        # First, try with nopython=False (object mode) which allows FFT
        if NUMBA_AVAILABLE:
            try:
                # Use object mode to allow FFT operations
                @jit(nopython=False, forceobj=True, cache=True)
                def nlse_rhs_numba_object(t, psi_t, k):
                    """RHS with Numba in object mode (allows FFT)."""
                    psi_t_filtered = psi_t * filter_arr
                    psi = np.fft.ifft(psi_t_filtered)
                    nonlinear = (np.abs(psi)**2) * psi
                    nonlinear_t = np.fft.fft(nonlinear)
                    nonlinear_t_filtered = nonlinear_t * filter_arr
                    rhs = -(1j/2) * (k**2) * psi_t_filtered + 1j * nonlinear_t_filtered
                    return rhs
                
                return nlse_rhs_numba_object
                
            except Exception as e:
                # If even object mode fails, create a hybrid approach
                if self.verbose:
                    print(f"{Fore.YELLOW}  Using hybrid Numba approach{Style.RESET_ALL}")
                
                # Compile only the nonlinear term calculation
                @jit(nopython=True, cache=True)
                def compute_nonlinear_term(psi_real, psi_imag):
                    """Compute |ψ|²ψ efficiently with Numba."""
                    n = len(psi_real)
                    nl_real = np.zeros(n)
                    nl_imag = np.zeros(n)
                    
                    for i in range(n):
                        abs_sq = psi_real[i]**2 + psi_imag[i]**2
                        nl_real[i] = abs_sq * psi_real[i]
                        nl_imag[i] = abs_sq * psi_imag[i]
                    
                    return nl_real, nl_imag
                
                def nlse_rhs_hybrid(t, psi_t, k):
                    """Hybrid RHS: FFT in Python, nonlinear term in Numba."""
                    # Apply filter
                    psi_t_filtered = psi_t * filter_arr
                    
                    # FFT to physical space (Python/NumPy)
                    psi = np.fft.ifft(psi_t_filtered)
                    
                    # Compute nonlinear term (Numba-accelerated)
                    psi_real = np.real(psi)
                    psi_imag = np.imag(psi)
                    nl_real, nl_imag = compute_nonlinear_term(psi_real, psi_imag)
                    nonlinear = nl_real + 1j * nl_imag
                    
                    # FFT back to Fourier space (Python/NumPy)
                    nonlinear_t = np.fft.fft(nonlinear)
                    nonlinear_t_filtered = nonlinear_t * filter_arr
                    
                    # Combine terms
                    rhs = -(1j/2) * (k**2) * psi_t_filtered + 1j * nonlinear_t_filtered
                    return rhs
                
                return nlse_rhs_hybrid
        
        # Fallback to pure Python
        return self._nlse_rhs_pure
    
    def _compute_conserved_quantities(self, psi: np.ndarray) -> Dict[str, float]:
        """
        Compute conserved quantities for monitoring stability.
        
        Parameters
        ----------
        psi : np.ndarray
            Wave function in physical space
        
        Returns
        -------
        dict
            Dictionary with mass, momentum, and energy
        """
        psi_x = np.gradient(psi, self.dx)
        
        # Mass (L2 norm): ∫|ψ|² dx
        mass = np.sum(np.abs(psi)**2) * self.dx
        
        # Momentum: ∫ Im(ψ* ∂ψ/∂x) dx
        momentum = np.real(np.sum(np.conj(psi) * psi_x * 1j)) * self.dx
        
        # Energy (Hamiltonian): ∫[(1/2)|∂ψ/∂x|² - (1/2)|ψ|⁴] dx
        energy = np.sum(0.5 * np.abs(psi_x)**2 - 0.5 * np.abs(psi)**4) * self.dx
        
        return {'mass': mass, 'momentum': momentum, 'energy': energy}
    
    def solve(
        self,
        initial_condition: np.ndarray,
        t_final: float = 20.0,
        n_snapshots: int = 100,
        method: str = 'DOP853',
        rtol: float = 1e-9,
        atol: float = 1e-11,
        dense_output: bool = False,
        show_progress: bool = True,
        monitor_conservation: bool = True
    ) -> Dict[str, Any]:
        """
        Solve the NLSE with given initial condition ψ₀.
        
        Parameters
        ----------
        initial_condition : np.ndarray
            Initial wave function ψ(x, t=0)
        t_final : float, optional
            Final simulation time (default: 20.0)
        n_snapshots : int, optional
            Number of time snapshots to save (default: 100)
        method : str, optional
            ODE solver method (default: 'DOP853' for high accuracy)
        rtol : float, optional
            Relative tolerance (default: 1e-9)
        atol : float, optional
            Absolute tolerance (default: 1e-11)
        dense_output : bool, optional
            Store dense output for interpolation (default: False)
        show_progress : bool, optional
            Show progress bar (default: True)
        monitor_conservation : bool, optional
            Monitor conserved quantities (default: True)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'x': Spatial grid
            - 't': Time array
            - 'psi': Complex wave function array ψ(x,t)
            - 'psi_abs': |ψ(x,t)|
            - 'psi_real': Re(ψ)
            - 'psi_imag': Im(ψ)
            - 'solver_info': Additional solver information
            - 'timing': Performance metrics
            - 'conservation': Conservation law monitoring
        """
        # Start timing
        start_time = time.time()
        timing = {}
        
        # Time grid
        t_eval = np.linspace(0, t_final, n_snapshots)
        
        # Transform initial condition to Fourier space
        if self.verbose:
            print(f"{Fore.MAGENTA}Preparing initial wave function ψ₀...{Style.RESET_ALL}")
        
        fft_start = time.time()
        psi_t0 = np.fft.fft(initial_condition)
        timing['fft_initial'] = time.time() - fft_start
        
        # Compute initial conserved quantities (but don't print)
        if monitor_conservation:
            initial_conserved = self._compute_conserved_quantities(initial_condition)
        
        if self.logger:
            self.logger.info(f"Starting NLSE integration: T={t_final}, snapshots={n_snapshots}")
        
        # Print simulation parameters
        if self.verbose:
            print(f"\n{Fore.GREEN}Starting NLSE Integration{Style.RESET_ALL}")
            print(f"  Time span: [0, {t_final:.2f}]")
            print(f"  Snapshots: {n_snapshots}")
            print(f"  Method: {method}")
            print(f"  Tolerances: rtol={rtol:.1e}, atol={atol:.1e}")
            print(f"{Fore.CYAN}{'─'*50}{Style.RESET_ALL}\n")
        
        # Progress bar setup
        self.pbar = None
        self.last_t = 0
        
        if show_progress and self.verbose:
            self.pbar = tqdm(
                total=t_final,
                desc="Solving NLSE",
                unit="t",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]",
                colour='green'
            )
        
        # Define progress callback
        def progress_callback(t, y):
            if self.pbar is not None:
                progress = t - self.last_t
                if progress > 0:
                    self.pbar.update(progress)
                    self.last_t = t
            return 0  # Continue integration
        
        # Solve ODE in Fourier space with enhanced stability
        integration_start = time.time()
        
        try:
            # Use higher-order method with adaptive stepping
            if self.use_adaptive:
                sol = solve_ivp(
                    lambda t, y: self._rhs_func(t, y, self.k),
                    [0, t_final],
                    psi_t0,
                    t_eval=t_eval,
                    method=method,  # DOP853 is 8th order
                    rtol=rtol,
                    atol=atol,
                    dense_output=dense_output,
                    max_step=0.1,  # Limit maximum step size
                    events=progress_callback if show_progress else None
                )
            else:
                sol = solve_ivp(
                    lambda t, y: self._rhs_func(t, y, self.k),
                    [0, t_final],
                    psi_t0,
                    t_eval=t_eval,
                    method='RK45',
                    rtol=rtol,
                    atol=atol,
                    dense_output=dense_output,
                    events=progress_callback if show_progress else None
                )
        finally:
            if self.pbar is not None:
                self.pbar.close()
        
        timing['integration'] = time.time() - integration_start
        
        if not sol.success:
            warning_msg = f"Solver did not converge: {sol.message}"
            if self.verbose:
                print(f"{Fore.RED}⚠ {warning_msg}{Style.RESET_ALL}")
            if self.logger:
                self.logger.warning(warning_msg)
            warnings.warn(warning_msg)
        
        # Transform back to physical space with progress bar
        if self.verbose:
            print(f"\n{Fore.MAGENTA}Transforming solution to physical space...{Style.RESET_ALL}")
        
        ifft_start = time.time()
        psi_solution = np.zeros((n_snapshots, self.M), dtype=complex)
        
        if show_progress and self.verbose:
            snapshot_iter = tqdm(
                range(n_snapshots),
                desc="Processing snapshots",
                unit="snap",
                colour='blue'
            )
        else:
            snapshot_iter = range(n_snapshots)
        
        for i in snapshot_iter:
            psi_solution[i, :] = np.fft.ifft(sol.y[:, i])
        
        timing['ifft_transform'] = time.time() - ifft_start
        
        # Monitor conservation laws (but don't print)
        conservation_data = {}
        if monitor_conservation:
            conservation_start = time.time()
            mass_array = np.zeros(n_snapshots)
            momentum_array = np.zeros(n_snapshots)
            energy_array = np.zeros(n_snapshots)
            
            for i in range(n_snapshots):
                conserved = self._compute_conserved_quantities(psi_solution[i])
                mass_array[i] = conserved['mass']
                momentum_array[i] = conserved['momentum']
                energy_array[i] = conserved['energy']
            
            # Compute conservation errors
            mass_error = np.abs(mass_array - initial_conserved['mass']) / initial_conserved['mass']
            momentum_error = np.abs(momentum_array - initial_conserved['momentum']) / (np.abs(initial_conserved['momentum']) + 1e-10)
            energy_error = np.abs(energy_array - initial_conserved['energy']) / np.abs(initial_conserved['energy'])
            
            conservation_data = {
                'mass': mass_array,
                'momentum': momentum_array,
                'energy': energy_array,
                'mass_error': mass_error.max(),
                'momentum_error': momentum_error.max(),
                'energy_error': energy_error.max(),
                'conservation_error': max(mass_error.max(), energy_error.max())
            }
            
            timing['conservation_check'] = time.time() - conservation_start
        
        # Total time
        timing['total_time'] = time.time() - start_time
        
        # Print completion message
        if self.verbose:
            self._print_completion_summary(timing, sol, conservation_data)
        
        if self.logger:
            self.logger.info(f"NLSE integration completed in {timing['total_time']:.3f}s")
            self.logger.info(f"Performance: FFT={timing['fft_initial']:.3f}s, "
                           f"Integration={timing['integration']:.3f}s, "
                           f"IFFT={timing['ifft_transform']:.3f}s")
        
        return {
            'x': self.x,
            't': t_eval,
            'psi': psi_solution,
            'psi_abs': np.abs(psi_solution),
            'psi_real': np.real(psi_solution),
            'psi_imag': np.imag(psi_solution),
            'solver_info': {
                'success': sol.success,
                'message': sol.message if hasattr(sol, 'message') else 'Success',
                'nfev': sol.nfev,
                'method': method,
                'rtol': rtol,
                'atol': atol,
                'adaptive': self.use_adaptive,
                'filtering': self.use_filtering
            },
            'timing': timing,
            'conservation': conservation_data if monitor_conservation else None
        }
    
    def _print_completion_summary(self, timing: Dict[str, float], sol: Any, conservation: Dict[str, Any]):
        """Print simulation completion summary."""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Simulation Completed Successfully{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # Timing breakdown
        print(f"\n{Fore.YELLOW}Performance Metrics:{Style.RESET_ALL}")
        print(f"  Total time: {timing['total_time']:.3f} seconds")
        print(f"  ├─ FFT (initial): {timing['fft_initial']:.3f}s "
              f"({100*timing['fft_initial']/timing['total_time']:.1f}%)")
        print(f"  ├─ Integration: {timing['integration']:.3f}s "
              f"({100*timing['integration']/timing['total_time']:.1f}%)")
        print(f"  └─ IFFT (final): {timing['ifft_transform']:.3f}s "
              f"({100*timing['ifft_transform']/timing['total_time']:.1f}%)")
        
        # Solver statistics
        print(f"\n{Fore.YELLOW}Solver Statistics:{Style.RESET_ALL}")
        print(f"  Function evaluations: {sol.nfev}")
        print(f"  Status: {Fore.GREEN if sol.success else Fore.RED}"
              f"{'Success' if sol.success else 'Failed'}{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
