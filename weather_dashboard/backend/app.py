# Weather Dashboard - Backend API
# FastAPI service for fetching and serving weather data

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import httpx
import asyncio
import os
from dotenv import load_dotenv
import logging
import json
from pathlib import Path

# -------------------------
# SETUP
# -------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="Weather Dashboard API",
    description="Fetch and serve real-time weather data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# CONFIGURATION
# -------------------------
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "demo")
OPENWEATHERMAP_BASE_URL = "https://api.openweathermap.org/data/2.5"
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./weather_cache"))
CACHE_DURATION = int(os.getenv("CACHE_DURATION_MINUTES", 10))

# Create cache directory
CACHE_DIR.mkdir(exist_ok=True)

# -------------------------
# DATA MODELS
# -------------------------
class Coordinates(BaseModel):
    """Geographic coordinates"""
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")

class Temperature(BaseModel):
    """Temperature data"""
    current: float = Field(..., description="Current temperature in Celsius")
    feels_like: float = Field(..., description="Feels like temperature")
    min: float = Field(..., description="Minimum temperature")
    max: float = Field(..., description="Maximum temperature")

class Weather(BaseModel):
    """Weather condition"""
    main: str = Field(..., description="Main weather condition (e.g., Clear, Clouds)")
    description: str = Field(..., description="Detailed description")
    icon: str = Field(..., description="Weather icon code")

class Wind(BaseModel):
    """Wind data"""
    speed: float = Field(..., description="Wind speed in m/s")
    deg: Optional[int] = Field(None, description="Wind direction in degrees")
    gust: Optional[float] = Field(None, description="Wind gust speed")

class WeatherData(BaseModel):
    """Complete weather data for a location"""
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country code")
    coordinates: Coordinates = Field(..., description="Geographic coordinates")
    temperature: Temperature = Field(..., description="Temperature data")
    weather: Weather = Field(..., description="Current weather condition")
    humidity: int = Field(..., description="Humidity percentage")
    pressure: int = Field(..., description="Atmospheric pressure in hPa")
    visibility: int = Field(..., description="Visibility in meters")
    wind: Wind = Field(..., description="Wind data")
    cloudiness: int = Field(..., description="Cloudiness percentage")
    timezone: int = Field(..., description="Timezone offset in seconds")
    sunrise: datetime = Field(..., description="Sunrise time")
    sunset: datetime = Field(..., description="Sunset time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data fetch time")

class ForecastItem(BaseModel):
    """Single forecast item"""
    timestamp: datetime = Field(..., description="Forecast time")
    temperature: float = Field(..., description="Forecasted temperature")
    weather: Weather = Field(..., description="Weather condition")
    humidity: int = Field(..., description="Humidity")
    wind_speed: float = Field(..., description="Wind speed")
    precipitation: float = Field(default=0, description="Precipitation in mm")

class Forecast(BaseModel):
    """Weather forecast data"""
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country code")
    items: List[ForecastItem] = Field(..., description="Forecast items")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AirQuality(BaseModel):
    """Air quality data"""
    aqi: int = Field(..., description="Air Quality Index (1-5)")
    co: float = Field(..., description="Carbon monoxide level")
    no2: float = Field(..., description="Nitrogen dioxide level")
    o3: float = Field(..., description="Ozone level")
    pm2_5: float = Field(..., description="PM2.5 level")
    pm10: float = Field(..., description="PM10 level")
    so2: float = Field(..., description="Sulfur dioxide level")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# -------------------------
# CACHE SYSTEM
# -------------------------
class WeatherCache:
    """Simple file-based cache for weather data"""
    
    @staticmethod
    def get_cache_path(cache_key: str) -> Path:
        """Get cache file path"""
        return CACHE_DIR / f"{cache_key}.json"
    
    @staticmethod
    def is_cache_valid(cache_key: str) -> bool:
        """Check if cache is still valid"""
        cache_path = WeatherCache.get_cache_path(cache_key)
        if not cache_path.exists():
            return False
        
        file_age = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds()
        return file_age < (CACHE_DURATION_MINUTES * 60)
    
    @staticmethod
    def get(cache_key: str) -> Optional[Dict]:
        """Retrieve from cache"""
        if not WeatherCache.is_cache_valid(cache_key):
            return None
        
        try:
            cache_path = WeatherCache.get_cache_path(cache_key)
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None
    
    @staticmethod
    def set(cache_key: str, data: Dict) -> None:
        """Store in cache"""
        try:
            cache_path = WeatherCache.get_cache_path(cache_key)
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            logger.info(f"Cache updated: {cache_key}")
        except Exception as e:
            logger.error(f"Cache write error: {e}")

# -------------------------
# WEATHER API CLIENT
# -------------------------
class WeatherAPIClient:
    """Client for OpenWeatherMap API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = OPENWEATHERMAP_BASE_URL
        self.timeout = 10
    
    async def get_current_weather(self, lat: float, lon: float, units: str = "metric") -> Optional[WeatherData]:
        """Fetch current weather by coordinates"""
        cache_key = f"weather_{lat}_{lon}"
        
        # Check cache
        cached = WeatherCache.get(cache_key)
        if cached:
            logger.info(f"Cache hit: {cache_key}")
            return WeatherData(**cached)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/weather",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "appid": self.api_key,
                        "units": units
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                weather_data = self._parse_weather_response(data)
                
                # Cache result
                WeatherCache.set(cache_key, weather_data.dict())
                
                return weather_data
        
        except httpx.HTTPError as e:
            logger.error(f"API error: {e}")
            return None
    
    async def get_weather_by_city(self, city: str, units: str = "metric") -> Optional[WeatherData]:
        """Fetch weather by city name"""
        cache_key = f"weather_city_{city}"
        
        # Check cache
        cached = WeatherCache.get(cache_key)
        if cached:
            logger.info(f"Cache hit: {cache_key}")
            return WeatherData(**cached)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/weather",
                    params={
                        "q": city,
                        "appid": self.api_key,
                        "units": units
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                weather_data = self._parse_weather_response(data)
                
                # Cache result
                WeatherCache.set(cache_key, weather_data.dict())
                
                return weather_data
        
        except httpx.HTTPError as e:
            logger.error(f"API error: {e}")
            return None
    
    async def get_forecast(self, lat: float, lon: float, units: str = "metric", count: int = 40) -> Optional[Forecast]:
        """Fetch 5-day forecast by coordinates"""
        cache_key = f"forecast_{lat}_{lon}"
        
        # Check cache
        cached = WeatherCache.get(cache_key)
        if cached:
            logger.info(f"Cache hit: {cache_key}")
            return Forecast(**cached)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/forecast",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "appid": self.api_key,
                        "units": units,
                        "cnt": count
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                forecast = self._parse_forecast_response(data)
                
                # Cache result
                WeatherCache.set(cache_key, forecast.dict())
                
                return forecast
        
        except httpx.HTTPError as e:
            logger.error(f"API error: {e}")
            return None
    
    async def get_air_quality(self, lat: float, lon: float) -> Optional[AirQuality]:
        """Fetch air quality data by coordinates"""
        cache_key = f"aqi_{lat}_{lon}"
        
        # Check cache
        cached = WeatherCache.get(cache_key)
        if cached:
            logger.info(f"Cache hit: {cache_key}")
            return AirQuality(**cached)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/air_pollution",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "appid": self.api_key
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                aqi = self._parse_air_quality_response(data)
                
                # Cache result
                WeatherCache.set(cache_key, aqi.dict())
                
                return aqi
        
        except httpx.HTTPError as e:
            logger.error(f"API error: {e}")
            return None
    
    @staticmethod
    def _parse_weather_response(data: Dict) -> WeatherData:
        """Parse OpenWeatherMap response"""
        return WeatherData(
            city=data['name'],
            country=data['sys']['country'],
            coordinates=Coordinates(
                lat=data['coord']['lat'],
                lon=data['coord']['lon']
            ),
            temperature=Temperature(
                current=data['main']['temp'],
                feels_like=data['main']['feels_like'],
                min=data['main']['temp_min'],
                max=data['main']['temp_max']
            ),
            weather=Weather(
                main=data['weather'][0]['main'],
                description=data['weather'][0]['description'],
                icon=data['weather'][0]['icon']
            ),
            humidity=data['main']['humidity'],
            pressure=data['main']['pressure'],
            visibility=data.get('visibility', 0),
            wind=Wind(
                speed=data['wind']['speed'],
                deg=data['wind'].get('deg'),
                gust=data['wind'].get('gust')
            ),
            cloudiness=data['clouds']['all'],
            timezone=data['timezone'],
            sunrise=datetime.fromtimestamp(data['sys']['sunrise']),
            sunset=datetime.fromtimestamp(data['sys']['sunset'])
        )
    
    @staticmethod
    def _parse_forecast_response(data: Dict) -> Forecast:
        """Parse forecast response"""
        items = []
        for item in data['list']:
            items.append(ForecastItem(
                timestamp=datetime.fromtimestamp(item['dt']),
                temperature=item['main']['temp'],
                weather=Weather(
                    main=item['weather'][0]['main'],
                    description=item['weather'][0]['description'],
                    icon=item['weather'][0]['icon']
                ),
                humidity=item['main']['humidity'],
                wind_speed=item['wind']['speed'],
                precipitation=item.get('rain', {}).get('3h', 0)
            ))
        
        return Forecast(
            city=data['city']['name'],
            country=data['city']['country'],
            items=items
        )
    
    @staticmethod
    def _parse_air_quality_response(data: Dict) -> AirQuality:
        """Parse air quality response"""
        components = data['list'][0]['components']
        return AirQuality(
            aqi=data['list'][0]['main']['aqi'],
            co=components.get('co', 0),
            no2=components.get('no2', 0),
            o3=components.get('o3', 0),
            pm2_5=components.get('pm2_5', 0),
            pm10=components.get('pm10', 0),
            so2=components.get('so2', 0)
        )

# -------------------------
# INITIALIZATION
# -------------------------
client = WeatherAPIClient(OPENWEATHERMAP_API_KEY)

# -------------------------
# ENDPOINTS
# -------------------------
@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Weather Dashboard API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/weather/current", response_model=WeatherData, tags=["Current Weather"])
async def get_current_weather(
    lat: float = Field(..., description="Latitude"),
    lon: float = Field(..., description="Longitude"),
    units: str = Field("metric", description="Units: metric, imperial, kelvin")
):
    """Get current weather by coordinates"""
    logger.info(f"Fetching weather for: lat={lat}, lon={lon}")
    
    weather = await client.get_current_weather(lat, lon, units)
    if not weather:
        raise HTTPException(status_code=502, detail="Failed to fetch weather data")
    
    return weather

@app.get("/weather/city", response_model=WeatherData, tags=["Current Weather"])
async def get_weather_by_city(
    city: str = Field(..., description="City name"),
    units: str = Field("metric", description="Units: metric, imperial, kelvin")
):
    """Get current weather by city name"""
    logger.info(f"Fetching weather for city: {city}")
    
    weather = await client.get_weather_by_city(city, units)
    if not weather:
        raise HTTPException(status_code=502, detail=f"City '{city}' not found")
    
    return weather

@app.post("/weather/cities", response_model=List[WeatherData], tags=["Current Weather"])
async def get_weather_for_multiple_cities(
    cities: List[str] = Field(..., description="List of city names")
):
    """Get current weather for multiple cities"""
    logger.info(f"Fetching weather for {len(cities)} cities")
    
    tasks = [client.get_weather_by_city(city) for city in cities]
    results = await asyncio.gather(*tasks)
    
    weather_list = [w for w in results if w is not None]
    
    if not weather_list:
        raise HTTPException(status_code=502, detail="Failed to fetch weather data for any city")
    
    return weather_list

@app.get("/forecast", response_model=Forecast, tags=["Forecast"])
async def get_forecast(
    lat: float = Field(..., description="Latitude"),
    lon: float = Field(..., description="Longitude"),
    units: str = Field("metric", description="Units: metric, imperial, kelvin")
):
    """Get 5-day weather forecast by coordinates"""
    logger.info(f"Fetching forecast for: lat={lat}, lon={lon}")
    
    forecast = await client.get_forecast(lat, lon, units)
    if not forecast:
        raise HTTPException(status_code=502, detail="Failed to fetch forecast data")
    
    return forecast

@app.get("/air-quality", response_model=AirQuality, tags=["Air Quality"])
async def get_air_quality(
    lat: float = Field(..., description="Latitude"),
    lon: float = Field(..., description="Longitude")
):
    """Get air quality data by coordinates"""
    logger.info(f"Fetching air quality for: lat={lat}, lon={lon}")
    
    aqi = await client.get_air_quality(lat, lon)
    if not aqi:
        raise HTTPException(status_code=502, detail="Failed to fetch air quality data")
    
    return aqi

@app.get("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Clear all cached weather data"""
    import shutil
    try:
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        logger.info("Cache cleared")
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("🌤️ Starting Weather Dashboard API")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
