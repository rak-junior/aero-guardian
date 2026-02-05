"""
Configuration Management Module
================================
Author: AeroGuardian Member

Centralized configuration management using python-dotenv for secure
environment variable handling.

This module provides:
    - Environment variable loading from .env file
    - Configuration validation
    - Type-safe config access

Usage:
    from src.core.config import get_config, load_env
    
    # Load environment variables
    load_env()
    
    # Get configuration
    config = get_config()
    print(config.openai_api_key)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Third-party imports
try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv is required. Install with: pip install python-dotenv"
    )

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Project root is 3 levels up from this file (src/core/config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class Config:
    """
    Application configuration with type-safe access to environment variables.
    
    Attributes:
        openai_api_key: OpenAI API key for LLM calls
        openai_model: Model to use (default: gpt-4o)
        openai_temperature: Sampling temperature (default: 0.1)
        openai_max_tokens: Maximum response tokens (default: 4096)
        embedding_model: Model for text embeddings
        log_level: Logging verbosity level
        project_root: Path to project root directory
    """
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 4096
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    log_level: str = "INFO"
    project_root: Path = PROJECT_ROOT
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.openai_api_key:
            logger.error(
                "OPENAI_API_KEY not configured. LLM features will not work. "
                "Please set OPENAI_API_KEY in .env file."
            )
            raise Exception("OPENAI_API_KEY not configured. Please set OPENAI_API_KEY in .env file.")
    
    def get_data_path(self, *parts: str) -> Path:
        """
        Get absolute path to a data directory or file.
        
        Args:
            *parts: Path components relative to data/
            
        Returns:
            Absolute Path object
            
        Example:
            config.get_data_path("alfa_telemetry", "cleaned")
        """
        return self.project_root / "data" / Path(*parts)


# =============================================================================
# Module-level singleton
# =============================================================================

_config: Optional[Config] = None


def load_env(env_file: Optional[Path] = None) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Optional path to .env file. Uses project root/.env if not specified.
        
    Returns:
        True if .env file was loaded, False otherwise.
        
    Example:
        >>> load_env()
        True
    """
    target = env_file or ENV_FILE
    
    if target.exists():
        load_dotenv(target, override=True)
        logger.info(f"Loaded environment from {target}")
        return True
    else:
        logger.warning(f"No .env file found at {target}. Using system environment.")
        return False


def get_config(reload: bool = False) -> Config:
    """
    Get application configuration singleton.
    
    Lazily loads environment variables and creates Config instance on first call.
    Subsequent calls return the cached instance unless reload=True.
    
    Args:
        reload: Force reload of configuration from environment.
        
    Returns:
        Config instance with current configuration values.
        
    Example:
        >>> config = get_config()
        >>> print(config.openai_model)
        'gpt-4o'
    """
    global _config
    
    if _config is None or reload:
        # Ensure environment is loaded
        load_env()
        
        # Create config from environment variables
        _config = Config(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        
        # Configure logging based on config
        logging.basicConfig(
            level=getattr(logging, _config.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
    
    return _config


# =============================================================================
# Convenience Exports
# =============================================================================

def get_openai_api_key() -> str:
    """Get OpenAI API key from configuration."""
    return get_config().openai_api_key


def get_project_root() -> Path:
    """Get project root path."""
    return PROJECT_ROOT


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    """Test configuration loading."""
    config = get_config()
    
    print("=" * 60)
    print("AeroGuardian Configuration")
    print("=" * 60)
    print(f"Project Root: {config.project_root}")
    print(f"OpenAI Model: {config.openai_model}")
    print(f"OpenAI Temperature: {config.openai_temperature}")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"Log Level: {config.log_level}")
    print(f"API Key Set: {'Yes' if config.openai_api_key else 'No'}")
    print("=" * 60)
