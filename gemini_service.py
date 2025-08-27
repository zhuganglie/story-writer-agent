
import asyncio
from typing import Dict, Any, Optional
import google.generativeai as genai
from rich.console import Console
import time
import hashlib
import json
from functools import lru_cache

class GeminiService:
    """Service for interacting with the Google Gemini API with enhanced error handling and caching"""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """Initialize the Gemini service"""
        self.console = Console()
        self.config = config
        self.cache = {}
        self.rate_limit_delay = config.get("api", {}).get("rate_limit_delay", 1.0)
        self.last_api_call = 0
        self.max_retries = config.get("api", {}).get("max_retries", 3)
        self.retry_delay = config.get("api", {}).get("retry_delay", 2.0)

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.config["api"]["model"])
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini API: {e}")

    def _get_cache_key(self, prompt: str, max_tokens: int) -> str:
        """Generate a cache key for the prompt"""
        content = f"{prompt}:{max_tokens}:{self.config['api']['temperature']}"
        return hashlib.md5(content.encode()).hexdigest()

    def _should_use_cache(self) -> bool:
        """Check if caching is enabled"""
        return self.config.get("api", {}).get("enable_cache", True)

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available"""
        if self._should_use_cache() and cache_key in self.cache:
            return self.cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache the response"""
        if self._should_use_cache():
            self.cache[cache_key] = response

    async def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            await asyncio.sleep(sleep_time)
        self.last_api_call = time.time()

    async def generate_content(self, prompt: str, max_tokens: int = 4000) -> str:
        """Generate content using Gemini API with retry logic and caching"""
        cache_key = self._get_cache_key(prompt, max_tokens)

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response is not None:
            if self.config.get("ui", {}).get("verbose_logging", False):
                self.console.print(f"Cache hit for prompt: {prompt[:50]}...")
            return cached_response

        for attempt in range(self.max_retries):
            try:
                # Enforce rate limiting
                await self._enforce_rate_limit()

                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=self.config["api"]["temperature"]
                    )
                )

                if response and response.parts:
                    result = response.text
                    # Cache successful response
                    self._cache_response(cache_key, result)
                    return result
                else:
                    self.console.print(f"Warning: API returned empty response for prompt: {prompt[:100]}...")
                    return ""

            except asyncio.TimeoutError:
                self.console.print(f"API request timed out (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    self.console.print("Max retries reached due to timeout")
                    return ""

            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "rate limit" in error_msg:
                    self.console.print(f"Rate limit exceeded (attempt {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue

                self.console.print(f"Error generating content (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return ""

        return ""
