"""Enhanced Logging System for NLSE Simulations"""

import logging
from pathlib import Path
from datetime import datetime

class SimulationLogger:
    """Enhanced logger for NLSE simulations."""
    
    def __init__(self, scenario_name: str, log_dir: str = "../logs", verbose: bool = True):
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{scenario_name}_{timestamp}.log"
        
        self.logger = self._setup_logger()
        
        self.info("="*70)
        self.info(f"NLSE Simulation Log - {scenario_name}")
        self.info(f"Institution: Samudera Sains Teknologi Ltd.")
        self.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("="*70)
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"nlse_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def log_parameters(self, params: dict):
        self.info("-" * 50)
        self.info("Simulation Parameters:")
        for key, value in params.items():
            self.info(f"  {key}: {value}")
        self.info("-" * 50)
    
    def log_timing(self, timing: dict):
        self.info("-" * 50)
        self.info("Performance Metrics:")
        self.info(f"  Total time: {timing.get('total_time', 0):.3f} seconds")
        self.info("-" * 50)
    
    def finalize(self):
        self.info("="*70)
        self.info(f"Simulation completed: {self.scenario_name}")
        self.info("="*70)
