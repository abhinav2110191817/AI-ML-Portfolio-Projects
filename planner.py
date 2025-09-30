import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")

HEADERS_OPENROUTER = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
}
NOMINATIM_HEADERS = {"User-Agent": "travel-planner/1.0"}

def geocode_place(place):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers=NOMINATIM_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    return {
        "lat": float(data[0]["lat"]),
        "lon": float(data[0]["lon"]),
        "display_name": data[0]["display_name"]
    }

def fetch_weather(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    days = {}
    for i, d in enumerate(data["daily"]["time"]):
        days[d] = {
            "temp_max": data["daily"]["temperature_2m_max"][i],
            "temp_min": data["daily"]["temperature_2m_min"][i],
            "precipitation": data["daily"]["precipitation_sum"][i],
            "weathercode": data["daily"]["weathercode"][i],
        }
    return {"daily": days}

def build_mcp_context(geodata, arrival, departure, weather):
    return {
        "place": geodata,
        "arrival": arrival,
        "departure": departure,
        "weather": weather,
        "notes": "Create an itinerary that adapts to daily weather conditions."
    }

def call_openrouter_with_context(context, user_request):
    if not OPENROUTER_KEY:
        raise Exception("Missing OpenRouter API key")
    url = "https://openrouter.ai/api/v1/chat/completions"
    system_msg = (
        "You are a precise travel planning assistant. Use the provided JSON context to create a "
        "day-by-day itinerary that considers the weather and arrival/departure details."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "<<<JSON_CONTEXT>>>\n" + json.dumps(context, indent=2)},
        {"role": "user", "content": user_request},
    ]
    payload = {"model": MODEL, "messages": messages, "temperature": 0.4}
    r = requests.post(url, headers=HEADERS_OPENROUTER, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def plan_trip(destination, start_date, end_date, arrival_info, departure_info):
    geodata = geocode_place(destination)
    if not geodata:
        raise Exception(f"Could not find {destination}")
    weather = fetch_weather(geodata["lat"], geodata["lon"], start_date, end_date)
    context = build_mcp_context(geodata, arrival_info, departure_info, weather)
    user_request = (
        f"Plan a trip to {destination} from {start_date} to {end_date}, "
        "with arrival/departure details included and daily weather-based suggestions."
    )
    return call_openrouter_with_context(context, user_request)
