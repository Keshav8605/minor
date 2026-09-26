"""
State management for meme analysis caching and image identity.

Provides SHA-256 content-based image identity and mode-aware caching
to prevent redundant VLM inference for previously analyzed images.

Cache key: (image_identity_hash, analysis_mode)
This ensures General and Cultural-Aware results are never mixed.
"""

import hashlib
import copy
import logging

logger = logging.getLogger("state_manager")

# Cache: (image_identity_hash, analysis_mode) → complete analysis result dict
MEME_CACHE = {}


def get_image_identity(path):
    """
    Compute a SHA-256 content hash for the image at the given path.

    Content-based identity ensures the same image is recognized
    even if its temporary upload path changes between sessions.

    Args:
        path: Filesystem path to the image file.

    Returns:
        str: Hex-encoded SHA-256 hash of the file contents.
    """
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()


def _cache_key(image_identity, analysis_mode):
    """Build composite cache key from image identity and analysis mode."""
    return (image_identity, analysis_mode)


def get_cached_result(image_identity, analysis_mode):
    """
    Retrieve a cached analysis result for the given image and mode.

    Returns a deep copy of the cached result to prevent mutation
    of the canonical cached data.

    Args:
        image_identity: SHA-256 hex hash of the image content.
        analysis_mode: Analysis mode string (e.g., "general", "cultural").

    Returns:
        dict or None: Deep copy of the cached result, or None on cache miss.
    """
    key = _cache_key(image_identity, analysis_mode)
    result = MEME_CACHE.get(key)
    if result is not None:
        logger.info("[CACHE] HIT image=%s... mode=%s", image_identity[:12], analysis_mode)
        return copy.deepcopy(result)
    logger.info("[CACHE] MISS image=%s... mode=%s", image_identity[:12], analysis_mode)
    return None


def store_cached_result(image_identity, analysis_mode, result):
    """
    Store a complete analysis result in the cache.

    A deep copy is stored to ensure the cached data cannot be
    mutated by subsequent code that modifies the original dict.

    Args:
        image_identity: SHA-256 hex hash of the image content.
        analysis_mode: Analysis mode string (e.g., "general", "cultural").
        result: Complete analysis result dictionary.
    """
    key = _cache_key(image_identity, analysis_mode)
    MEME_CACHE[key] = copy.deepcopy(result)
    logger.info("[CACHE] STORED image=%s... mode=%s", image_identity[:12], analysis_mode)


def clear_cache():
    """Clear all cached analysis results."""
    count = len(MEME_CACHE)
    MEME_CACHE.clear()
    logger.info("[CACHE] Cleared %d cached results", count)
