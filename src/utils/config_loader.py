import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import yaml

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        # Fallback to root directory if running from src/...
        # This allows running from subdirectories as well
        # Assuming the structure is project/src/utils
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root_config = os.path.join(root_dir, "config.yaml")

        if os.path.exists(root_config):
            config_path = root_config
        else:
            # Fallback for when running from project root directly
            if os.path.exists("config.yaml"):
                config_path = "config.yaml"
            else:
                 raise FileNotFoundError(f"Config file not found at {config_path} or {root_config}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_rag_config():
    config = load_config()
    return config.get('rag_data', {})
