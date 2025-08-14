"""Configuration Management with YAML and TXT Support"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """Manage configuration files for NLSE simulations."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Load configuration from file (YAML, JSON, or TXT)."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config = json.load(f)
        elif path.suffix == '.txt':
            config = ConfigManager._parse_txt_config(path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return config
    
    @staticmethod
    def _parse_txt_config(filepath: Path) -> Dict[str, Any]:
        """Parse TXT configuration file."""
        config = {}
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        elif value.isdigit() or (value[0] == '-' and value[1:].isdigit()):
                            value = int(value)
                        elif '.' in value or 'e' in value.lower():
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                    except (ValueError, AttributeError):
                        pass
                    
                    if '.' in key:
                        parts = key.split('.')
                        current = config
                        for part in parts[:-1]:
                            if part not in current:
                                current[part] = {}
                            current = current[part]
                        current[parts[-1]] = value
                    else:
                        config[key] = value
        
        return config
