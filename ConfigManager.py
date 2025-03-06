"""
Configuration for Kontoplan Mapping System

This module provides configuration management for the Kontoplan mapping system,
allowing settings to be loaded from config files, environment variables, or
command-line arguments.
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration values
DEFAULT_CONFIG = {
    # Paths
    'paths': {
        'kontoplan': '2023-01-31-Standardkontoplan.xlsx',
        'model': 'kontoplan_mapper_model.joblib',
        'answer_file': 'transaction_mapping.csv',
        'output_dir': 'output',
        'log_dir': 'logs',
    },
    
    # Machine learning settings
    'ml': {
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
    },
    
    # Prediction settings
    'prediction': {
        'confidence_threshold': 0.6,
        'enable_continuous_learning': True,
        'min_history_size_for_retraining': 100,
    },
    
    # Training data generation
    'training_data': {
        'num_transactions': 1000,
        'output_name': 'training_data',
    },
    
    # Logging settings
    'logging': {
        'level': 'INFO',
        'log_to_file': True,
        'log_to_console': True,
    },
}

class ConfigManager:
    """
    Configuration manager for the Kontoplan mapping system
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Parameters:
        ----------
        config_file : str, optional
            Path to the configuration file (YAML format)
        """
        self.config = DEFAULT_CONFIG.copy()
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger("ConfigManager")
        self.logger.info("Configuration loaded")
    
    def _load_from_file(self, config_file: str) -> None:
        """
        Load configuration from a YAML file
        
        Parameters:
        ----------
        config_file : str
            Path to the configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Update configuration with values from file
            if file_config:
                self._update_config(self.config, file_config)
                
        except Exception as e:
            print(f"Error loading configuration from {config_file}: {str(e)}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        # Path settings
        if 'KONTOPLAN_PATH' in os.environ:
            self.config['paths']['kontoplan'] = os.environ['KONTOPLAN_PATH']
        
        if 'KONTOPLAN_MODEL_PATH' in os.environ:
            self.config['paths']['model'] = os.environ['KONTOPLAN_MODEL_PATH']
        
        if 'KONTOPLAN_ANSWER_FILE' in os.environ:
            self.config['paths']['answer_file'] = os.environ['KONTOPLAN_ANSWER_FILE']
        
        if 'KONTOPLAN_OUTPUT_DIR' in os.environ:
            self.config['paths']['output_dir'] = os.environ['KONTOPLAN_OUTPUT_DIR']
        
        # ML settings
        if 'KONTOPLAN_CONFIDENCE_THRESHOLD' in os.environ:
            self.config['prediction']['confidence_threshold'] = float(
                os.environ['KONTOPLAN_CONFIDENCE_THRESHOLD']
            )
        
        # Logging settings
        if 'KONTOPLAN_LOG_LEVEL' in os.environ:
            self.config['logging']['level'] = os.environ['KONTOPLAN_LOG_LEVEL']
    
    def _update_config(self, config: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Recursively update configuration dictionary
        
        Parameters:
        ----------
        config : dict
            Target configuration dictionary
        updates : dict
            Updates to apply
        """
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._update_config(config[key], value)
            else:
                config[key] = value
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.config['logging']['level'], logging.INFO)
        
        handlers = []
        
        # Add console handler if enabled
        if self.config['logging']['log_to_console']:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            handlers.append(console_handler)
        
        # Add file handler if enabled
        if self.config['logging']['log_to_file']:
            # Create log directory if it doesn't exist
            log_dir = self.config['paths']['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            
            # Create log file with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"kontoplan_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            handlers.append(file_handler)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration
        
        Returns:
        -------
        dict
            Complete configuration dictionary
        """
        return self.config
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value
        
        Parameters:
        ----------
        section : str
            Configuration section
        key : str
            Configuration key
        default : any, optional
            Default value if not found
            
        Returns:
        -------
        any
            Configuration value
        """
        try:
            return self.config[section][key]
        except KeyError:
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a specific configuration value
        
        Parameters:
        ----------
        section : str
            Configuration section
        key : str
            Configuration key
        value : any
            Configuration value
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save current configuration to a YAML file
        
        Parameters:
        ----------
        file_path : str
            Path to the output file
            
        Returns:
        -------
        bool
            True if saving was successful
        """
        try:
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration to {file_path}: {str(e)}")
            return False
    
    def ensure_paths(self) -> None:
        """Ensure all configured directories exist"""
        dirs = [
            self.config['paths']['output_dir'],
            self.config['paths']['log_dir']
        ]
        
        for dir_path in dirs:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {dir_path}")
    
    @staticmethod
    def add_config_args(parser: argparse.ArgumentParser) -> None:
        """
        Add configuration-related arguments to an argument parser
        
        Parameters:
        ----------
        parser : argparse.ArgumentParser
            Argument parser to add configuration options to
        """
        group = parser.add_argument_group('Configuration')
        group.add_argument('--config', help='Path to configuration file (YAML)')
        group.add_argument('--kontoplan-path', help='Path to Standardkontoplan Excel file')
        group.add_argument('--model-path', help='Path to model file')
        group.add_argument('--answer-file', help='Path to answer file')
        group.add_argument('--output-dir', help='Output directory')
        group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                         help='Logging level')
    
    @staticmethod
    def update_from_args(config_manager, args):
        """
        Update configuration from command-line arguments
        
        Parameters:
        ----------
        config_manager : ConfigManager
            Configuration manager to update
        args : argparse.Namespace
            Parsed command-line arguments
        """
        # If config file is specified, load it
        if hasattr(args, 'config') and args.config:
            config_manager._load_from_file(args.config)
        
        # Override with specific arguments if provided
        if hasattr(args, 'kontoplan_path') and args.kontoplan_path:
            config_manager.set('paths', 'kontoplan', args.kontoplan_path)
        
        if hasattr(args, 'model_path') and args.model_path:
            config_manager.set('paths', 'model', args.model_path)
        
        if hasattr(args, 'answer_file') and args.answer_file:
            config_manager.set('paths', 'answer_file', args.answer_file)
        
        if hasattr(args, 'output_dir') and args.output_dir:
            config_manager.set('paths', 'output_dir', args.output_dir)
        
        if hasattr(args, 'log_level') and args.log_level:
            config_manager.set('logging', 'level', args.log_level)
            # Re-initialize logging with new level
            config_manager._setup_logging()


def create_default_config_file(output_path='kontoplan_config.yml'):
    """
    Create a default configuration file
    
    Parameters:
    ----------
    output_path : str
        Path to output file
        
    Returns:
    -------
    bool
        True if file was created successfully
    """
    try:
        with open(output_path, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
        
        print(f"Default configuration file created at {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating default configuration file: {str(e)}")
        return False


if __name__ == "__main__":
    # Command-line interface for configuration management
    parser = argparse.ArgumentParser(description='Kontoplan Configuration Manager')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create default config command
    create_parser = subparsers.add_parser('create-default', 
                                        help='Create a default configuration file')
    create_parser.add_argument('--output', default='kontoplan_config.yml',
                             help='Output file path (default: kontoplan_config.yml)')
    
    # View config command
    view_parser = subparsers.add_parser('view', help='View current configuration')
    view_parser.add_argument('--config', help='Path to configuration file (YAML)')
    
    # Add common configuration arguments
    ConfigManager.add_config_args(parser)
    
    args = parser.parse_args()
    
    if args.command == 'create-default':
        create_default_config_file(args.output)
    elif args.command == 'view':
        # Initialize config manager with specified file
        config = ConfigManager(args.config if hasattr(args, 'config') else None)
        
        # Update from command-line arguments
        ConfigManager.update_from_args(config, args)
        
        # Print configuration
        print(yaml.dump(config.get_config(), default_flow_style=False))
    else:
        parser.print_help()