# 🌤️ Weather Dashboard

A real-time weather dashboard that fetches data from OpenWeatherMap API and displays current weather, 5-day forecasts, and air quality information.

## Features

✅ **Current Weather** - Real-time temperature, humidity, wind speed, and more
✅ **5-Day Forecast** - Hourly weather predictions
✅ **Air Quality Index** - PM2.5, PM10, NO₂ and other pollutants
✅ **Multiple Cities** - Compare weather across different locations
✅ **Geolocation** - Get weather for your current location
✅ **Caching** - Smart caching to reduce API calls
✅ **Responsive Design** - Works on desktop, tablet, and mobile
✅ **Real-time Updates** - Live weather data from OpenWeatherMap

## Architecture

```
weather_dashboard/
├── backend/
│   └── app.py              # FastAPI backend service
├── frontend/
│   ├── index.html          # Main HTML page
│   ├── styles.css          # Styling
│   └── script.js           # JavaScript logic
├── .env.example            # Environment template
└── README.md               # This file
```

## Prerequisites

- Python 3.9+
- OpenWeatherMap API Key (free at https://openweathermap.org/api)
- Modern web browser
- pip

## Installation

### 1. Get OpenWeatherMap API Key

1. Visit https://openweathermap.org/api
2. Sign up for a free account
3. Create a new API key
4. Copy your API key

### 2. Setup Backend

```bash
# Navigate to weather dashboard directory
cd weather_dashboard

# Install dependencies
pip install fastapi uvicorn httpx python-dotenv pydantic

# Create .env file
cp .env.example .env

# Edit .env and add your API key
echo "OPENWEATHERMAP_API_KEY=your_api_key_here" >> .env

# Run backend
python backend/app.py
```

The backend will start on `http://localhost:8001`

### 3. Setup Frontend

```bash
# In another terminal, navigate to frontend directory
cd weather_dashboard/frontend

# Start a local web server (Python 3)
python -m http.server 8000

# Or use any other static server (npm, node, etc.)
```

Access the dashboard at `http://localhost:8000`

## API Endpoints

### Current Weather

**By Coordinates**
```bash
GET /weather/current?lat=51.5074&lon=-0.1278&units=metric
```

**By City Name**
```bash
GET /weather/city?city=London&units=metric
```

### Forecast

```bash
GET /forecast?lat=51.5074&lon=-0.1278&units=metric
```

### Air Quality

```bash
GET /air-quality?lat=51.5074&lon=-0.1278
```

### Multiple Cities

```bash
POST /weather/cities
Content-Type: application/json

{"cities": ["London", "Paris", "Tokyo"]}
```

## Usage

### Search by City
1. Enter a city name in the search box
2. Click "Search" or press Enter
3. View current weather, forecast, and air quality

### Use Your Location
1. Click "📍 Use My Location"
2. Allow browser geolocation access
3. Weather data for your location appears automatically

### Compare Multiple Cities
1. Enter city names separated by commas
2. Click "📊 Compare"
3. View weather cards side-by-side

## Configuration

Edit `.env` file:

```env
# API Key
OPENWEATHERMAP_API_KEY=your_key_here

# Cache settings
CACHE_DIR=./weather_cache
CACHE_DURATION_MINUTES=10  # Cache duration in minutes

# Server settings
HOST=0.0.0.0
PORT=8001
```

## Data Models

### WeatherData
```json
{
  "city": "London",
  "country": "GB",
  "coordinates": {"lat": 51.5074, "lon": -0.1278},
  "temperature": {
    "current": 15.2,
    "feels_like": 14.8,
    "min": 12.5,
    "max": 18.3
  },
  "weather": {
    "main": "Clouds",
    "description": "overcast clouds",
    "icon": "04d"
  },
  "humidity": 72,
  "pressure": 1013,
  "visibility": 10000,
  "wind": {"speed": 5.2, "deg": 230},
  "cloudiness": 90,
  "sunrise": "2024-01-15T07:30:00",
  "sunset": "2024-01-15T16:45:00"
}
```

### Forecast Item
```json
{
  "timestamp": "2024-01-15T12:00:00",
  "temperature": 16.5,
  "weather": {"main": "Clouds", "description": "scattered clouds"},
  "humidity": 68,
  "wind_speed": 4.8,
  "precipitation": 0
}
```

## Caching

The backend implements smart caching to reduce API calls:

- **Cache Duration**: Configurable (default: 10 minutes)
- **Storage**: File-based JSON cache in `weather_cache/` directory
- **Auto-expiry**: Cached data expires after configured duration
- **Clear Cache**: `GET /cache/clear` endpoint

## Units

Supported units for temperature and wind speed:
- `metric` (Celsius, m/s) - **Default**
- `imperial` (Fahrenheit, mph)
- `kelvin` (Kelvin, m/s)

## Air Quality Index (AQI)

AQI Levels:
- **1**: Good (0-50 µg/m³)
- **2**: Fair (51-100 µg/m³)
- **3**: Moderate (101-150 µg/m³)
- **4**: Poor (151-200 µg/m³)
- **5**: Very Poor (>200 µg/m³)

## Weather Icons

Icons come from OpenWeatherMap and include:
- Clear sky ☀️
- Clouds ☁️
- Rain 🌧️
- Thunderstorm ⛈️
- Snow ❄️
- Mist 🌫️

## Troubleshooting

### Issue: CORS Error
**Solution**: The backend has CORS enabled for all origins by default. If you get CORS errors:
1. Check that backend is running on port 8001
2. Verify API endpoint URLs in script.js
3. Check browser console for specific errors

### Issue: "City not found"
**Solution**:
1. Check city name spelling
2. Try using English city names
3. Use coordinates instead (lat/lon)

### Issue: No air quality data
**Solution**: Some cities may not have air quality data. Try major cities like London, Paris, Tokyo.

### Issue: Cache not clearing
**Solution**: Manual clear:
```bash
rm -rf weather_cache/
mkdir weather_cache/
```

## API Rate Limits

Free tier OpenWeatherMap API:
- **Calls per minute**: 60
- **Calls per month**: 1,000,000

Caching helps reduce rate limit issues.

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY weather_dashboard ./

CMD ["python", "backend/app.py"]
```

### Heroku

```bash
heroku create weather-dashboard
heroku config:set OPENWEATHERMAP_API_KEY=your_key
git push heroku main
```

### Docker Compose

```yaml
version: '3.8'
services:
  backend:
    build: ./weather_dashboard
    ports:
      - "8001:8001"
    environment:
      OPENWEATHERMAP_API_KEY: ${OPENWEATHERMAP_API_KEY}
  
  frontend:
    image: nginx:alpine
    volumes:
      - ./weather_dashboard/frontend:/usr/share/nginx/html
    ports:
      - "8000:80"
```

## Performance Tips

1. **Increase cache duration** for less active monitoring
2. **Limit forecast items** to reduce data transfer
3. **Use coordinates** instead of city names for faster lookups
4. **Implement pagination** for multiple cities comparison
5. **Lazy load** forecast and AQI sections

## Security

1. **Never commit `.env`** file with API keys
2. **Use environment variables** for sensitive data
3. **Validate all inputs** (already done with Pydantic)
4. **Rate limit** in production (use nginx or API gateway)
5. **HTTPS only** in production

## Future Enhancements

- [ ] Historical weather data
- [ ] Weather alerts and notifications
- [ ] Custom saved locations
- [ ] Weather graphs and charts
- [ ] Map integration
- [ ] Dark mode
- [ ] Mobile app
- [ ] Advanced filters

## Dependencies

**Backend:**
- fastapi - Web framework
- uvicorn - ASGI server
- httpx - Async HTTP client
- pydantic - Data validation
- python-dotenv - Environment variables

**Frontend:**
- Vanilla JavaScript (no dependencies)
- HTML5
- CSS3

## License

Part of Neon Artificial Cognitive Development Initiative

## Support

For issues:
1. Check the Troubleshooting section
2. Review OpenWeatherMap API documentation
3. Check browser console for errors
4. Verify .env configuration

---

**Weather Dashboard is now online! 🌤️**
