"""
Geocoder Service
================
Author: AeroGuardian Member
Date: 2026-01-18

Converts city/state locations to GPS coordinates using OpenStreetMap Nominatim.
No API key required. Rate limited to 1 request/second per Nominatim policy.
"""

import logging
import time
import requests
from typing import Tuple, Dict

logger = logging.getLogger("AeroGuardian.Geocoder")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "AeroGuardian/1.0 (UAV Safety Research)"

# Cache for already geocoded locations
_geocode_cache: Dict[str, Tuple[float, float]] = {}

# Last request timestamp for rate limiting
_last_request_time = 0.0


def geocode(city: str, state: str) -> Tuple[float, float]:
    """
    Convert city/state to GPS coordinates.
    
    Args:
        city: City name (e.g., "MINNEAPOLIS")
        state: State name (e.g., "MINNESOTA")
    
    Returns:
        Tuple of (latitude, longitude)
    
    Raises:
        ValueError: If location cannot be geocoded
    """
    global _last_request_time
    
    # Normalize input
    location_key = f"{city.strip()}, {state.strip()}, USA".upper()
    
    # Check cache
    if location_key in _geocode_cache:
        return _geocode_cache[location_key]
    
    # Rate limit (1 req/sec)
    now = time.time()
    if now - _last_request_time < 1.0:
        time.sleep(1.0 - (now - _last_request_time))
    
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={
                "q": location_key,
                "format": "json",
                "limit": 1,
                "countrycodes": "us",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        _last_request_time = time.time()
        
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError(f"No geocoding results for: {location_key}")
        
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        
        # Cache result
        _geocode_cache[location_key] = (lat, lon)
        
        logger.info(f"Geocoded '{location_key}' -> ({lat:.4f}, {lon:.4f})")
        return lat, lon
        
    except requests.RequestException as e:
        logger.error(f"Geocoding request failed: {e}")
        raise ValueError(f"Geocoding failed for {location_key}: {e}")


def geocode_incident(incident: Dict) -> Tuple[float, float]:
    """
    Geocode an FAA incident dict.
    
    Falls back to PX4 default location if geocoding fails.
    """
    city = incident.get("city", "")
    state = incident.get("state", "")
    
    if not city or not state:
        logger.warning("Incident missing city/state, using default location")
        return (47.397742, 8.545594)  # PX4 Default (Zurich)
    
    try:
        return geocode(city, state)
    except ValueError as e:
        logger.warning(f"Geocoding failed: {e}, using default location")
        return (47.397742, 8.545594)


def get_cache_size() -> int:
    """Get number of cached locations."""
    return len(_geocode_cache)
