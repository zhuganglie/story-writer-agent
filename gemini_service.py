"""
Optimized service for interacting with the Google Gemini API.
Enhanced with improved error handling, caching, and performance optimizations.
"""

import asyncio
import hashlib
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json

import google.generativeai as genai
from rich.console import Console

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy options"""
    EXPONENTIAL_BACKOFF = "exponential"
    LINEAR_BACKOFF = "linear"
    FIXED_DELAY = "fixed"


@dataclass
class GenerationRequest:
    """Request object for content generation"""
    prompt: str
    max_tokens: int = 4000
    temperature: float = 0.7
    cache_key: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.cache_key is None:
            self.cache_key = self._generate_cache_key()

    def _generate_cache_key(self) -> str:
        """Generate cache key from request parameters"""
        content = f"{self.prompt}:{self.max_tokens}:{self.temperature}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class GenerationResponse:
    """Response object for content generation"""
    content: str
    tokens_used: int = 0
    cached: bool = False
    generation_time: float = 0.0
    model_used: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CacheEntry:
    """Cache entry with TTL and metadata"""
    def __init__(self, content: str, ttl: int = 3600, metadata: Dict[str, Any] = None):
        self.content = content
        self.created_at = time.time()
        self.ttl = ttl
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.created_at > self.ttl

    def access(self) -> str:
        """Access the cached content and update statistics"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.content


class GeminiServiceError(Exception):
    """Base exception for Gemini service errors"""
    pass


class RateLimitError(GeminiServiceError):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceededError(GeminiServiceError):
    """Raised when API quota is exceeded"""
    pass


class InvalidRequestError(GeminiServiceError):
    """Raised for invalid requests"""
    pass


class GeminiService:
    """Optimized service for interacting with the Google Gemini API"""

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """Initialize the Gemini service with enhanced configuration"""
        self.console = Console()
        self.config = config
        self._api_config = config.get("api", {})

        # Cache configuration
        self._cache: Dict[str, CacheEntry] = {}
        self._max_cache_size = self._api_config.get("max_cache_size", 1000)
        self._cache_ttl = self._api_config.get("cache_ttl", 3600)
        self._enable_cache = self._api_config.get("enable_cache", True)

        # Rate limiting
        self._rate_limit_delay = self._api_config.get("rate_limit_delay", 1.0)
        self._last_api_call = 0
        self._request_queue: List[float] = []
        self._max_requests_per_minute = self._api_config.get("max_requests_per_minute", 60)

        # Retry configuration
        self._max_retries = self._api_config.get("max_retries", 3)
        self._retry_delay = self._api_config.get("retry_delay", 2.0)
        self._retry_strategy = RetryStrategy(self._api_config.get("retry_strategy", "exponential"))

        # Statistics
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "retries": 0,
            "total_tokens": 0,
            "total_generation_time": 0.0
        }

        # Initialize Gemini API
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self._api_config["model"])
            logger.info(f"Initialized Gemini API with model: {self._api_config['model']}")
        except Exception as e:
            raise GeminiServiceError(f"Failed to initialize Gemini API: {e}") from e

    def _should_use_cache(self) -> bool:
        """Check if caching is enabled and available"""
        return self._enable_cache and len(self._cache) < self._max_cache_size

    def _get_cache_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Get cache entry if valid and not expired"""
        if not self._enable_cache or cache_key not in self._cache:
            return None

        entry = self._cache[cache_key]
        if entry.is_expired():
            del self._cache[cache_key]
            return None

        return entry

    def _cache_response(self, cache_key: str, content: str, metadata: Dict[str, Any] = None):
        """Cache response with LRU eviction if needed"""
        if not self._should_use_cache():
            return

        # Evict oldest entries if cache is full
        if len(self._cache) >= self._max_cache_size:
            self._evict_oldest_entries()

        self._cache[cache_key] = CacheEntry(
            content=content,
            ttl=self._cache_ttl,
            metadata=metadata or {}
        )

    def _evict_oldest_entries(self, count: int = 10):
        """Evict oldest cache entries to make room"""
        if len(self._cache) <= count:
            return

        # Sort by last accessed time and remove oldest
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed
        )

        for cache_key, _ in sorted_entries[:count]:
            del self._cache[cache_key]

        logger.debug(f"Evicted {count} cache entries")

    async def _enforce_rate_limit(self):
        """Enforce rate limiting with sliding window"""
        current_time = time.time()

        # Clean old requests from queue
        cutoff_time = current_time - 60  # 1 minute window
        self._request_queue = [req_time for req_time in self._request_queue if req_time > cutoff_time]

        # Check if we can make a request
        if len(self._request_queue) >= self._max_requests_per_minute:
            sleep_time = 60 - (current_time - self._request_queue[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                current_time = time.time()

        # Enforce minimum delay between requests
        time_since_last = current_time - self._last_api_call
        if time_since_last < self._rate_limit_delay:
            sleep_time = self._rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
            current_time = time.time()

        # Record this request
        self._request_queue.append(current_time)
        self._last_api_call = current_time

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy"""
        base_delay = self._retry_delay

        if self._retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay * (2 ** attempt)
        elif self._retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return base_delay * (attempt + 1)
        else:  # FIXED_DELAY
            return base_delay

    def _classify_error(self, error: Exception) -> Exception:
        """Classify and convert generic errors to specific types"""
        error_msg = str(error).lower()

        if "quota" in error_msg or "limit" in error_msg:
            if "rate" in error_msg:
                retry_after = self._extract_retry_after(str(error))
                return RateLimitError(str(error), retry_after)
            else:
                return QuotaExceededError(str(error))
        elif "invalid" in error_msg or "bad request" in error_msg:
            return InvalidRequestError(str(error))
        else:
            return GeminiServiceError(str(error))

    def _extract_retry_after(self, error_message: str) -> Optional[int]:
        """Extract retry-after value from error message"""
        import re
        match = re.search(r'retry.*?(\d+)', error_message, re.IGNORECASE)
        return int(match.group(1)) if match else None

    async def generate_content(self,
                             prompt: str,
                             max_tokens: int = 4000,
                             temperature: Optional[float] = None,
                             **kwargs) -> GenerationResponse:
        """Generate content with comprehensive error handling and optimization"""

        # Create request object
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else self._api_config["temperature"],
            metadata=kwargs
        )

        # Update statistics
        self._stats["total_requests"] += 1

        # Check cache first
        if self._enable_cache:
            cache_entry = self._get_cache_entry(request.cache_key)
            if cache_entry:
                self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return GenerationResponse(
                    content=cache_entry.access(),
                    cached=True,
                    metadata=cache_entry.metadata
                )

        self._stats["cache_misses"] += 1

        # Attempt generation with retries
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                start_time = time.time()

                # Enforce rate limiting
                await self._enforce_rate_limit()

                # Make API request
                response = await self._make_api_request(request)
                generation_time = time.time() - start_time

                # Update statistics
                self._stats["total_generation_time"] += generation_time
                estimated_tokens = len(response.split()) * 1.3  # Rough estimation
                self._stats["total_tokens"] += int(estimated_tokens)

                # Cache successful response
                if self._enable_cache:
                    self._cache_response(
                        request.cache_key,
                        response,
                        {"tokens": estimated_tokens, "generation_time": generation_time}
                    )

                logger.debug(f"Generated {len(response)} characters in {generation_time:.2f}s")

                return GenerationResponse(
                    content=response,
                    tokens_used=int(estimated_tokens),
                    cached=False,
                    generation_time=generation_time,
                    model_used=self._api_config["model"],
                    metadata={"attempt": attempt}
                )

            except Exception as e:
                last_error = self._classify_error(e)
                self._stats["errors"] += 1

                # Don't retry for certain errors
                if isinstance(last_error, (QuotaExceededError, InvalidRequestError)):
                    logger.error(f"Non-retryable error: {last_error}")
                    break

                if attempt < self._max_retries:
                    self._stats["retries"] += 1
                    retry_delay = self._calculate_retry_delay(attempt)

                    # Use retry-after header if available
                    if isinstance(last_error, RateLimitError) and last_error.retry_after:
                        retry_delay = max(retry_delay, last_error.retry_after)

                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self._max_retries + 1}): "
                        f"{last_error}. Retrying in {retry_delay:.2f}s"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached. Final error: {last_error}")

        # All retries exhausted
        raise last_error or GeminiServiceError("Unknown error occurred")

    async def _make_api_request(self, request: GenerationRequest) -> str:
        """Make the actual API request with timeout handling"""
        timeout = self._api_config.get("timeout", 300)

        try:
            response = await asyncio.wait_for(
                self._generate_content_internal(request),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            raise GeminiServiceError(f"Request timed out after {timeout} seconds")

    async def _generate_content_internal(self, request: GenerationRequest) -> str:
        """Internal method to generate content"""
        try:
            # Run the synchronous generation in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._sync_generate_content,
                request
            )
            return response
        except Exception as e:
            raise self._classify_error(e)

    def _sync_generate_content(self, request: GenerationRequest) -> str:
        """Synchronous content generation"""
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=request.max_tokens,
            temperature=request.temperature
        )

        response = self.model.generate_content(
            request.prompt,
            generation_config=generation_config
        )

        if not response or not response.parts:
            raise GeminiServiceError("API returned empty response")

        return response.text

    async def generate_batch(self,
                           requests: List[Dict[str, Any]],
                           max_concurrent: int = 5) -> List[GenerationResponse]:
        """Generate multiple content requests concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _generate_single(req_data: Dict[str, Any]) -> GenerationResponse:
            async with semaphore:
                return await self.generate_content(**req_data)

        tasks = [_generate_single(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch request {i} failed: {response}")
                results.append(GenerationResponse(
                    content="",
                    metadata={"error": str(response), "request_index": i}
                ))
            else:
                results.append(response)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        cache_stats = {
            "cache_size": len(self._cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] / (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0 else 0
            ),
            "total_cache_entries": len(self._cache),
            "cache_memory_usage": sum(len(entry.content) for entry in self._cache.values())
        }

        performance_stats = {
            "average_generation_time": (
                self._stats["total_generation_time"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "requests_per_second": len(self._request_queue),  # Rough estimate for last minute
            "error_rate": (
                self._stats["errors"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            )
        }

        return {
            **self._stats,
            **cache_stats,
            **performance_stats,
            "config": {
                "model": self._api_config["model"],
                "max_retries": self._max_retries,
                "cache_enabled": self._enable_cache,
                "retry_strategy": self._retry_strategy.value
            }
        }

    def clear_cache(self):
        """Clear the content cache"""
        cleared_count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {cleared_count} cache entries")

    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the service"""
        try:
            test_response = await self.generate_content(
                "Test prompt for health check",
                max_tokens=10
            )

            return {
                "status": "healthy",
                "response_received": len(test_response.content) > 0,
                "cache_operational": self._enable_cache,
                "statistics": self.get_statistics()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "statistics": self.get_statistics()
            }
