import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import logging
from huggingface_hub import snapshot_download

from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model():
    """Download the base model specified in config."""
    config = load_config()
    
    # Get model configuration
    models_config = config.get("models", {})
    gen_config = models_config.get("generation", {})
    
    model_name = gen_config.get("model_name")
    local_path = gen_config.get("local_path")
    
    if not model_name:
        logger.error("No model_name specified in config (models.generation.model_name).")
        return

    if not local_path:
        # Default to a models directory in project root
        local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", model_name.split("/")[-1])
        logger.warning(f"No local_path specified, using default: {local_path}")

    logger.info(f"Downloading model '{model_name}' to '{local_path}'...")
    
    try:
        # Ensure directories exist
        os.makedirs(local_path, exist_ok=True)
        
        # Download from Hugging Face (supports HF_ENDPOINT for mirrors)
        snapshot_download(
            repo_id=model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,  # Download actual files
            resume_download=True
        )
        logger.info(f"Model '{model_name}' downloaded successfully to {local_path}")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")

if __name__ == "__main__":
    download_model()
