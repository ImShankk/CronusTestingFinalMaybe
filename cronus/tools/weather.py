"""Weather via Open-Meteo (no API key required).

Migrated from the original ``tool.py`` and extended with a forecast, WMO code
descriptions, and honest failure messages.
"""

from __future__ import annotations

from typing import Any

import requests

from ..logging_setup import get_logger
from .base import RiskLevel, Tool, ToolContext, ToolResult, object_schema

log = get_logger("tools.weather")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 10

# WMO weather interpretation codes, condensed to what people actually say.
WMO_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers", 87: "hail showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "severe thunderstorms with hail",
}


def describe_code(code: Any) -> str:
    try:
        return WMO_CODES.get(int(code), "unsettled")
    except (TypeError, ValueError):
        return "unsettled"


def _geocode(city: str) -> dict[str, Any] | None:
    response = requests.get(
        GEOCODE_URL, params={"name": city, "count": 1}, timeout=_TIMEOUT
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    return results[0] if results else None


def resolve_location(city: str | None, context: ToolContext | None) -> str | None:
    """Where to look, preferring what the user actually said.

    Order: the city in the request, then anything the user has told Cronus,
    then the configured location. Returns None when nothing is known, so the
    caller can ask instead of picking somewhere at random.
    """
    if city and city.strip():
        return city.strip()
    if context is None:
        return None
    if context.profile is not None:
        stated = context.profile.get("location") or context.profile.get("default_city")
        if stated:
            return stated
    return context.config.location


def get_weather(
    city: str | None = None, days: int = 1, context: ToolContext | None = None
) -> ToolResult:
    """Current conditions, plus a short forecast when asked for."""
    city = resolve_location(city, context)
    if not city:
        return ToolResult.failure(
            "No location is configured and none was given, so there is nowhere "
            "to check. Ask the user which place they mean -- do not guess a city."
        )
    try:
        place = _geocode(city)
    except requests.RequestException as exc:
        log.error("geocoding failed for %r: %s", city, exc)
        return ToolResult.failure(
            "The weather service is unreachable right now, so I have no data for "
            f"{city}."
        )

    if place is None:
        return ToolResult.failure(
            f"No place called {city!r} was found. Ask the user to be more specific."
        )

    label = ", ".join(
        str(part)
        for part in (place.get("name"), place.get("admin1"), place.get("country"))
        if part
    )
    params: dict[str, Any] = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,wind_speed_10m,weather_code",
        "timezone": "auto",
    }
    days = max(1, min(int(days), 7))
    if days > 1:
        params["daily"] = "temperature_2m_max,temperature_2m_min,weather_code"
        params["forecast_days"] = days

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        log.error("forecast failed for %r: %s", city, exc)
        return ToolResult.failure(
            f"I reached the map service but not the forecast for {label}."
        )

    current = payload.get("current") or {}
    temperature = current.get("temperature_2m")
    feels_like = current.get("apparent_temperature")
    wind = current.get("wind_speed_10m")
    condition = describe_code(current.get("weather_code"))

    lines = [
        f"{label} right now: {_round(temperature)} degrees Celsius, {condition}, "
        f"wind {_round(wind)} km/h."
    ]
    if feels_like is not None and temperature is not None:
        if abs(float(feels_like) - float(temperature)) >= 3:
            lines.append(f"It feels like {_round(feels_like)} degrees.")

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    for index, date in enumerate(dates[:days]):
        lines.append(
            f"{date}: high {_round(daily['temperature_2m_max'][index])}, "
            f"low {_round(daily['temperature_2m_min'][index])}, "
            f"{describe_code(daily['weather_code'][index])}."
        )

    return ToolResult(
        content="\n".join(lines),
        display=f"{label}: {_round(temperature)}C, {condition}",
        data={"location": label, "temperature_c": temperature, "condition": condition},
    )


def _round(value: Any) -> Any:
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return value


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description=(
                "Current weather and short forecast for a city. Use this for any "
                "question about temperature, rain, snow, or conditions right now "
                "or in the next seven days. It cannot say what a place is "
                "typically like in a season or a month -- for that, answer from "
                "what you know and say you are describing the usual climate "
                "rather than a forecast."
            ),
            parameters=object_schema(
                {
                    "city": {
                        "type": "string",
                        "description": (
                            "City name, optionally with region or country. Leave "
                            "this out to use the user's own location. Never "
                            "invent a city."
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "description": "Days of forecast to include, 1 to 7. 1 means now only.",
                        "minimum": 1,
                        "maximum": 7,
                        "default": 1,
                    },
                },
            ),
            handler=get_weather,
            risk=RiskLevel.SAFE,
            category="information",
            timeout=20.0,
        )
    ]
