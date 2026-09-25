import yaml
import logging

logger = logging.getLogger(__name__)

def load_yaml_config(config_path):
    """Load YAML configuration file.
    Args:
        config_path (str): Path to the YAML config file.
    Returns:
        dict: Parsed configuration dictionary.
    """
    logger.info(f'Loading configuration from {config_path}')
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.info(exc)
    return config

def save_yaml_config(config, save_path):
    """Save configuration dictionary to a YAML file.
    Args:
        config (dict): Configuration dictionary to save.
        save_path (str): Path to save the YAML config file.
    """
    logger.info(f'Saving configuration to {save_path}')
    with open(save_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        
