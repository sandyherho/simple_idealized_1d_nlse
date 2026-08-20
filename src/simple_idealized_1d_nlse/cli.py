import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from .core.solver import NLSESolver
from .core.initial_conditions import (
    SingleSoliton, TwoSoliton, BreatherSolution, GaussianNoise
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """Print the decorated header with ASCII art."""
    header = """
    ═══════════════════════════════════════════════════════════════════════════════
                                                                                   
                  1D NLSE SOLITON DYNAMICS SIMULATOR v.0.0.5                         
                                                                                   
         ~~~∿∿∿~~~  ψ  ~~~∿∿∿~~~  |ψ|²  ~~~∿∿∿~~~  ∂ψ/∂t  ~~~∿∿∿~~~           
                                                                                   
                        Samudera Sains Teknologi Ltd.          
                        
       Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami,
       Rusmawan Suwarman, Dasapta E. Irawan                    
           
           LICENSE: DO WHAT THE F*CK YOU WANT TO PUBLIC LICENSE                     
    ═══════════════════════════════════════════════════════════════════════════════
    """
    print(header)


def print_footer():
    """Print the decorated footer."""
    footer = """
    ═══════════════════════════════════════════════════════════════════════════════
                        Simulation Complete | ψ(x,t) Computed Successfully
                              Contact: sandyherho@itb.ac.id
    ═══════════════════════════════════════════════════════════════════════════════
    """
    print(footer)


def print_scenario_header(scenario_name: str):
    """Print decorated header for scenario."""
    header = f"""
    ─────────────────────────────────────────────────────────────────────────────
      Executing Scenario: {scenario_name:<54} 					  
      ~~~∿∿∿ Solving NLSE with Pseudo-spectral Methods ∿∿∿~~~                         
    ─────────────────────────────────────────────────────────────────────────────
    """
    print(header)


def run_scenario(config: Dict[str, Any], output_dir: str = "../outputs", verbose: bool = True) -> None:
    """Run a single simulation scenario."""
    scenario_name = config.get('scenario_name', 'simulation')
    
    if verbose:
        print_scenario_header(scenario_name)
    
    logger = SimulationLogger(
        scenario_name=scenario_name.lower().replace(' ', '_'),
        log_dir="../logs",
        verbose=verbose
    )
    
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        with timer.time_section("solver_init"):
            solver = NLSESolver(
                domain_length=config['domain_length'],
                num_points=config['num_points'],
                use_numba=config.get('use_numba', False),
                use_adaptive=config.get('use_adaptive', True),
                use_filtering=config.get('use_filtering', True),
                verbose=verbose,
                logger=logger
            )
        
        with timer.time_section("initial_condition"):
            ic_type = config.get('initial_condition', 'single_soliton')
            
            if verbose:
                print(f"    ψ₀: Initializing {ic_type.replace('_', ' ').title()}")
            
            if ic_type == 'single_soliton':
                initial = SingleSoliton(
                    amplitude=config.get('amplitude', 2.0),
                    position=config.get('position', 0.0),
                    velocity=config.get('velocity', 1.0)
                )
            elif ic_type == 'two_soliton':
                initial = TwoSoliton(
                    A1=config.get('A1', 2.0),
                    A2=config.get('A2', 1.5),
                    x1=config.get('x1', -10.0),
                    x2=config.get('x2', 10.0),
                    v1=config.get('v1', 2.0),
                    v2=config.get('v2', -2.0)
                )
            elif ic_type == 'breather':
                initial = BreatherSolution(
                    amplitude=config.get('amplitude', 1.0),
                    frequency=config.get('frequency', 0.5)
                )
            elif ic_type == 'modulation_instability':
                initial = GaussianNoise(
                    amplitude=config.get('amplitude', 1.0),
                    width=config.get('width', 5.0),
                    center=config.get('center', 0.0),
                    noise_level=config.get('noise_level', 0.01),
                    seed=config.get('seed', 42)
                )
            else:
                raise ValueError(f"Unknown initial condition: {ic_type}")
            
            psi_0 = initial(solver.x)
        
        with timer.time_section("simulation"):
            if verbose:
                print("    ∂ψ/∂t: Evolving wave function...")
            
            result = solver.solve(
                initial_condition=psi_0,
                t_final=config['t_final'],
                n_snapshots=config['n_snapshots'],
                method=config.get('method', 'DOP853'),
                rtol=config.get('rtol', 1e-9),
                atol=config.get('atol', 1e-11),
                show_progress=verbose,
                monitor_conservation=config.get('monitor_conservation', True)
            )
        
        if config.get('save_netcdf', True):
            with timer.time_section("save_netcdf"):
                filename = f"{scenario_name.lower().replace(' ', '_')}.nc"
                if verbose:
                    print(f"    💾 Saving NetCDF: {filename}")
                DataHandler.save_netcdf(
                    filename=filename,
                    x=result['x'],
                    t=result['t'],
                    psi=result['psi'],
                    metadata=config,
                    output_dir=output_dir
                )
        
        if config.get('save_animation', True):
            with timer.time_section("create_animation"):
                filename = f"{scenario_name.lower().replace(' ', '_')}.gif"
                if verbose:
                    print(f"    🎬 Creating animation: {filename}")
                Animator.create_gif(
                    x=result['x'],
                    t=result['t'],
                    psi=result['psi'],
                    filename=filename,
                    output_dir=output_dir,
                    title=scenario_name,
                    fps=config.get('fps', 20),
                    dpi=config.get('dpi', 100),
                    verbose=verbose,
                    logger=logger
                )
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        if verbose:
            print(f"    ✓ Scenario completed in {timer.get_times()['total']:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in scenario {scenario_name}: {str(e)}")
        raise
    
    finally:
        logger.finalize()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Simple Idealized 1D NLSE Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    ═══════════════════════════════════════════════════════════════════════════════
     Examples:                                                                    
        nlse-simulate single_soliton                                       
        nlse-simulate --config my_config.yaml                              
    
    ═══════════════════════════════════════════════════════════════════════════════
        """
    )
    
    parser.add_argument('scenario', nargs='?',
                       choices=['single_soliton', 'two_soliton', 'breather', 'modulation_instability'],
                       help='Predefined scenario to run')
    
    parser.add_argument('--config', '-c', type=str,
                       help='Path to configuration file (YAML or TXT)')
    
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all predefined scenarios')
    
    parser.add_argument('--output-dir', '-o', type=str, default='../outputs',
                       help='Output directory')
    
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Enable verbose output')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress all decorative output')
    
    args = parser.parse_args()
    
    # Override verbose if quiet is set
    if args.quiet:
        args.verbose = False
    
    if args.verbose:
        print_header()
    
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, args.verbose)
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs' / 'yaml'
        for config_file in sorted(configs_dir.glob('*.yaml')):
            config = ConfigManager.load(str(config_file))
            run_scenario(config, args.output_dir, args.verbose)
    elif args.scenario:
        configs_dir = Path(__file__).parent.parent.parent / 'configs' / 'yaml'
        config_file = configs_dir / f'{args.scenario}.yaml'
        if config_file.exists():
            config = ConfigManager.load(str(config_file))
            run_scenario(config, args.output_dir, args.verbose)
    else:
        if not args.verbose:
            parser.print_help()
        else:
            print_header()
            parser.print_help()
        sys.exit(0)
    
    if args.verbose:
        print_footer()


if __name__ == '__main__':
    main()
