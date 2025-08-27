"""
Optimized configuration manager for the Story Writer Agent.
Provides validation, caching, and type-safe configuration handling.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class ApiConfig:
    """API configuration with validation"""
    model: str = "gemini-2.5-flash"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 300
    enable_cache: bool = True
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    retry_delay: float = 2.0

    def __post_init__(self):
        """Validate configuration values"""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.timeout < 1:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if self.rate_limit_delay < 0:
            raise ValueError(f"rate_limit_delay cannot be negative, got {self.rate_limit_delay}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")


@dataclass
class StoryConfig:
    """Story generation configuration"""
    target_word_count: int = 3000
    min_word_count: int = 2000
    max_word_count: int = 5000
    auto_save: bool = True
    save_frequency: str = "after_each_stage"

    def __post_init__(self):
        """Validate story configuration"""
        if not self.min_word_count <= self.target_word_count <= self.max_word_count:
            raise ValueError(
                f"Word count range invalid: min={self.min_word_count}, "
                f"target={self.target_word_count}, max={self.max_word_count}"
            )
        valid_frequencies = ["after_each_stage", "every_minute", "manual"]
        if self.save_frequency not in valid_frequencies:
            raise ValueError(f"save_frequency must be one of {valid_frequencies}")


@dataclass
class LanguageConfig:
    """Language configuration with validation"""
    default: str = "en"
    supported: List[str] = field(default_factory=lambda: [
        "en", "zh", "zh-tw", "es", "fr", "de", "ja", "ko",
        "it", "pt", "ru", "ar", "hi"
    ])
    auto_detect: bool = False
    preserve_original_names: bool = True
    cultural_adaptation: bool = True

    def __post_init__(self):
        """Validate language configuration"""
        if self.default not in self.supported:
            raise ValueError(f"Default language '{self.default}' not in supported languages")


@dataclass
class UIConfig:
    """UI configuration"""
    show_progress: bool = True
    interactive_mode: bool = True
    color_output: bool = True
    verbose_logging: bool = False
    language: str = "en"


@dataclass
class WorkflowConfig:
    """Workflow configuration"""
    stages: List[str] = field(default_factory=lambda: [
        "concept_generation", "structure_planning", "character_development",
        "setting_creation", "draft_generation", "revision_enhancement", "final_polish"
    ])
    allow_stage_skipping: bool = False
    require_user_approval: bool = True

    def __post_init__(self):
        """Validate workflow stages"""
        required_stages = {
            "concept_generation", "structure_planning", "character_development",
            "setting_creation", "draft_generation", "revision_enhancement", "final_polish"
        }
        if not set(self.stages) == required_stages:
            missing = required_stages - set(self.stages)
            extra = set(self.stages) - required_stages
            if missing:
                raise ValueError(f"Missing required stages: {missing}")
            if extra:
                raise ValueError(f"Unknown stages: {extra}")


@dataclass
class AppConfig:
    """Complete application configuration"""
    api: ApiConfig = field(default_factory=ApiConfig)
    story: StoryConfig = field(default_factory=StoryConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)


class ConfigManager:
    """Optimized configuration manager with caching and validation"""

    _instance: Optional['ConfigManager'] = None
    _config_cache: Dict[str, AppConfig] = {}

    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern for global configuration access"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the configuration manager"""
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._config_path: Optional[Path] = None
        self._config: Optional[AppConfig] = None
        self._watchers: List[callable] = []

    @lru_cache(maxsize=32)
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration as dictionary with caching"""
        return {
            "api": {
                "model": "gemini-2.5-flash",
                "max_tokens": 4000,
                "temperature": 0.7,
                "timeout": 300,
                "enable_cache": True,
                "rate_limit_delay": 1.0,
                "max_retries": 3,
                "retry_delay": 2.0
            },
            "story": {
                "target_word_count": 3000,
                "min_word_count": 2000,
                "max_word_count": 5000,
                "auto_save": True,
                "save_frequency": "after_each_stage"
            },
            "language": {
                "default": "en",
                "supported": ["en", "zh", "zh-tw", "es", "fr", "de", "ja", "ko", "it", "pt", "ru", "ar", "hi"],
                "auto_detect": False,
                "preserve_original_names": True,
                "cultural_adaptation": True
            },
            "ui": {
                "show_progress": True,
                "interactive_mode": True,
                "color_output": True,
                "verbose_logging": False,
                "language": "en"
            },
            "workflow": {
                "stages": [
                    "concept_generation", "structure_planning", "character_development",
                    "setting_creation", "draft_generation", "revision_enhancement", "final_polish"
                ],
                "allow_stage_skipping": False,
                "require_user_approval": True
            }
        }

    def load_config(self, config_path: Union[str, Path] = "config.json") -> AppConfig:
        """Load configuration from file with validation and caching"""
        config_path = Path(config_path)

        # Check cache first
        cache_key = str(config_path.absolute())
        if cache_key in self._config_cache:
            # Check if file has been modified
            if config_path.exists():
                cached_time = getattr(self._config_cache[cache_key], '_load_time', 0)
                if config_path.stat().st_mtime <= cached_time:
                    return self._config_cache[cache_key]

        # Load configuration
        config_dict = self._get_default_config().copy()

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)

                # Deep merge configuration
                config_dict = self._deep_merge_config(config_dict, user_config)
                logger.info(f"Loaded configuration from {config_path}")

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in {config_path}: {e}. Using defaults.")
            except IOError as e:
                logger.warning(f"Error reading {config_path}: {e}. Using defaults.")
        else:
            logger.info(f"Config file {config_path} not found, using defaults.")

        # Create and validate configuration objects
        try:
            config = AppConfig(
                api=ApiConfig(**config_dict["api"]),
                story=StoryConfig(**config_dict["story"]),
                language=LanguageConfig(**config_dict["language"]),
                ui=UIConfig(**config_dict["ui"]),
                workflow=WorkflowConfig(**config_dict["workflow"])
            )

            # Add load time for cache invalidation
            config._load_time = config_path.stat().st_mtime if config_path.exists() else 0

            # Cache the configuration
            self._config_cache[cache_key] = config
            self._config_path = config_path
            self._config = config

            # Notify watchers
            self._notify_watchers(config)

            return config

        except (ValueError, TypeError) as e:
            logger.error(f"Configuration validation error: {e}")
            # Fall back to default configuration
            config = AppConfig()
            config._load_time = 0
            return config

    def _deep_merge_config(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge user configuration with defaults"""
        result = default.copy()

        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_config(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(self, config: Optional[AppConfig] = None,
                   config_path: Optional[Union[str, Path]] = None) -> bool:
        """Save configuration to file with backup"""
        if config is None:
            config = self._config
        if config is None:
            logger.error("No configuration to save")
            return False

        if config_path is None:
            config_path = self._config_path
        if config_path is None:
            config_path = Path("config.json")
        else:
            config_path = Path(config_path)

        # Create backup if file exists
        if config_path.exists():
            backup_path = config_path.with_suffix(f"{config_path.suffix}.backup")
            try:
                backup_path.write_bytes(config_path.read_bytes())
            except IOError as e:
                logger.warning(f"Failed to create backup: {e}")

        # Convert config to dictionary
        config_dict = {
            "api": {
                "model": config.api.model,
                "max_tokens": config.api.max_tokens,
                "temperature": config.api.temperature,
                "timeout": config.api.timeout,
                "enable_cache": config.api.enable_cache,
                "rate_limit_delay": config.api.rate_limit_delay,
                "max_retries": config.api.max_retries,
                "retry_delay": config.api.retry_delay
            },
            "story": {
                "target_word_count": config.story.target_word_count,
                "min_word_count": config.story.min_word_count,
                "max_word_count": config.story.max_word_count,
                "auto_save": config.story.auto_save,
                "save_frequency": config.story.save_frequency
            },
            "language": {
                "default": config.language.default,
                "supported": config.language.supported,
                "auto_detect": config.language.auto_detect,
                "preserve_original_names": config.language.preserve_original_names,
                "cultural_adaptation": config.language.cultural_adaptation
            },
            "ui": {
                "show_progress": config.ui.show_progress,
                "interactive_mode": config.ui.interactive_mode,
                "color_output": config.ui.color_output,
                "verbose_logging": config.ui.verbose_logging,
                "language": config.ui.language
            },
            "workflow": {
                "stages": config.workflow.stages,
                "allow_stage_skipping": config.workflow.allow_stage_skipping,
                "require_user_approval": config.workflow.require_user_approval
            }
        }

        try:
            # Write with atomic operation
            temp_path = config_path.with_suffix(".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_path.replace(config_path)

            logger.info(f"Configuration saved to {config_path}")
            return True

        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def get_config(self) -> AppConfig:
        """Get current configuration, loading if necessary"""
        if self._config is None:
            return self.load_config()
        return self._config

    def update_config(self, **kwargs) -> None:
        """Update configuration values dynamically"""
        if self._config is None:
            self._config = self.load_config()

        # Update values
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

        # Notify watchers
        self._notify_watchers(self._config)

    def watch_config(self, callback: callable) -> None:
        """Register a callback for configuration changes"""
        self._watchers.append(callback)

    def _notify_watchers(self, config: AppConfig) -> None:
        """Notify all registered watchers of configuration changes"""
        for callback in self._watchers:
            try:
                callback(config)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")

    def validate_config(self, config_dict: Dict[str, Any]) -> List[str]:
        """Validate configuration dictionary and return list of errors"""
        errors = []

        try:
            # Try to create configuration objects to validate
            ApiConfig(**config_dict.get("api", {}))
        except (ValueError, TypeError) as e:
            errors.append(f"API config error: {e}")

        try:
            StoryConfig(**config_dict.get("story", {}))
        except (ValueError, TypeError) as e:
            errors.append(f"Story config error: {e}")

        try:
            LanguageConfig(**config_dict.get("language", {}))
        except (ValueError, TypeError) as e:
            errors.append(f"Language config error: {e}")

        try:
            UIConfig(**config_dict.get("ui", {}))
        except (ValueError, TypeError) as e:
            errors.append(f"UI config error: {e}")

        try:
            WorkflowConfig(**config_dict.get("workflow", {}))
        except (ValueError, TypeError) as e:
            errors.append(f"Workflow config error: {e}")

        return errors

    def clear_cache(self) -> None:
        """Clear configuration cache"""
        self._config_cache.clear()
        # Clear LRU cache
        self._get_default_config.cache_clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics for debugging"""
        return {
            "cached_configs": len(self._config_cache),
            "cache_keys": list(self._config_cache.keys()),
            "default_config_cache": self._get_default_config.cache_info()._asdict(),
            "watchers_count": len(self._watchers)
        }


# Global configuration manager instance
config_manager = ConfigManager()


def get_config(config_path: Union[str, Path] = "config.json") -> AppConfig:
    """Convenience function to get configuration"""
    return config_manager.load_config(config_path)


def save_config(config: AppConfig, config_path: Union[str, Path] = "config.json") -> bool:
    """Convenience function to save configuration"""
    return config_manager.save_config(config, config_path)
