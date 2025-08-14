"""Data Handling and Storage with Institutional Metadata"""

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class DataHandler:
    """Handle data storage and retrieval for NLSE simulations."""
    
    @staticmethod
    def save_netcdf(filename: str, x: np.ndarray, t: np.ndarray,
                   psi: np.ndarray, metadata: Optional[Dict[str, Any]] = None,
                   output_dir: str = "../outputs") -> None:
        """Save simulation results to NetCDF file with institutional metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            nc.createDimension('x', len(x))
            nc.createDimension('t', len(t))
            
            nc_x = nc.createVariable('x', 'f8', ('x',))
            nc_t = nc.createVariable('t', 'f8', ('t',))
            nc_psi_real = nc.createVariable('psi_real', 'f8', ('t', 'x'))
            nc_psi_imag = nc.createVariable('psi_imag', 'f8', ('t', 'x'))
            nc_psi_abs = nc.createVariable('psi_abs', 'f8', ('t', 'x'))
            
            nc_x[:] = x
            nc_t[:] = t
            nc_psi_real[:] = np.real(psi)
            nc_psi_imag[:] = np.imag(psi)
            nc_psi_abs[:] = np.abs(psi)
            
            # Institutional metadata
            nc.institution = "Samudera Sains Teknologi Ltd."
            nc.department = "Computational Earth and Environmental Science Division"
            nc.description = "NLSE Simulation Results"
            nc.equation = "i∂ψ/∂t + (1/2)∂²ψ/∂x² + |ψ|²ψ = 0"
            nc.created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            nc.software = "simple_idealized_1d_nlse v0.0.1"
            nc.authors = "Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Fikry P. Lugina, Rusmawan Suwarman, Dasapta E. Irawan"
            nc.contact = "sandy.herho@email.ucr.edu"
            nc.license = "WTFPL"
            
            if metadata and 'scenario_name' in metadata:
                nc.scenario = metadata['scenario_name']
            
            nc_x.units = "dimensionless"
            nc_x.long_name = "Spatial coordinate"
            
            nc_t.units = "dimensionless"
            nc_t.long_name = "Time"
            
            nc_psi_real.long_name = "Real part of wave function"
            nc_psi_real.symbol = "Re(ψ)"
            
            nc_psi_imag.long_name = "Imaginary part of wave function"
            nc_psi_imag.symbol = "Im(ψ)"
            
            nc_psi_abs.long_name = "Absolute value of wave function"
            nc_psi_abs.symbol = "|ψ|"
