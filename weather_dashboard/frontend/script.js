// Weather Dashboard Frontend
const API_BASE_URL = 'http://localhost:8001';

const elements = {
    cityInput: document.getElementById('cityInput'),
    searchBtn: document.getElementById('searchBtn'),
    locationBtn: document.getElementById('locationBtn'),
    loading: document.getElementById('loading'),
    error: document.getElementById('error'),
    weatherSection: document.getElementById('weatherSection'),
    forecastSection: document.getElementById('forecastSection'),
    aqiSection: document.getElementById('aqiSection'),
    citiesInput: document.getElementById('citiesInput'),
    compareBtn: document.getElementById('compareBtn'),
    citiesContainer: document.getElementById('citiesContainer')
};

// Event Listeners
elements.searchBtn.addEventListener('click', () => searchByCity());
elements.locationBtn.addEventListener('click', () => useGeolocation());
elements.compareBtn.addEventListener('click', () => compareMultipleCities());
elements.cityInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') searchByCity();
});

// Search by city
async function searchByCity() {
    const city = elements.cityInput.value.trim();
    if (!city) {
        showError('Please enter a city name');
        return;
    }
    
    await fetchWeatherByCity(city);
    await fetchForecastByCity(city);
    await fetchAirQualityByCity(city);
}

// Use geolocation
function useGeolocation() {
    if (!navigator.geolocation) {
        showError('Geolocation is not supported by your browser');
        return;
    }
    
    showLoading(true);
    navigator.geolocation.getCurrentPosition(
        async (position) => {
            const { latitude, longitude } = position.coords;
            await fetchWeatherByCoords(latitude, longitude);
            await fetchForecastByCoords(latitude, longitude);
            await fetchAirQualityByCoords(latitude, longitude);
        },
        (error) => {
            showError(`Geolocation error: ${error.message}`);
            showLoading(false);
        }
    );
}

// Fetch current weather by city
async function fetchWeatherByCity(city) {
    try {
        showLoading(true);
        hideError();
        
        const response = await fetch(`${API_BASE_URL}/weather/city?city=${city}&units=metric`);
        if (!response.ok) throw new Error('City not found');
        
        const data = await response.json();
        displayWeather(data);
    } catch (error) {
        showError(`Error fetching weather: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Fetch current weather by coordinates
async function fetchWeatherByCoords(lat, lon) {
    try {
        showLoading(true);
        hideError();
        
        const response = await fetch(`${API_BASE_URL}/weather/current?lat=${lat}&lon=${lon}&units=metric`);
        if (!response.ok) throw new Error('Failed to fetch weather');
        
        const data = await response.json();
        displayWeather(data);
        elements.cityInput.value = `${data.city}, ${data.country}`;
    } catch (error) {
        showError(`Error fetching weather: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Fetch forecast
async function fetchForecastByCity(city) {
    try {
        const response = await fetch(`${API_BASE_URL}/weather/city?city=${city}&units=metric`);
        const weatherData = await response.json();
        
        const forecastResponse = await fetch(
            `${API_BASE_URL}/forecast?lat=${weatherData.coordinates.lat}&lon=${weatherData.coordinates.lon}&units=metric`
        );
        if (!forecastResponse.ok) throw new Error('Failed to fetch forecast');
        
        const forecast = await forecastResponse.json();
        displayForecast(forecast);
    } catch (error) {
        console.error('Error fetching forecast:', error);
    }
}

async function fetchForecastByCoords(lat, lon) {
    try {
        const response = await fetch(
            `${API_BASE_URL}/forecast?lat=${lat}&lon=${lon}&units=metric`
        );
        if (!response.ok) throw new Error('Failed to fetch forecast');
        
        const forecast = await response.json();
        displayForecast(forecast);
    } catch (error) {
        console.error('Error fetching forecast:', error);
    }
}

// Fetch air quality
async function fetchAirQualityByCity(city) {
    try {
        const response = await fetch(`${API_BASE_URL}/weather/city?city=${city}&units=metric`);
        const weatherData = await response.json();
        
        const aqiResponse = await fetch(
            `${API_BASE_URL}/air-quality?lat=${weatherData.coordinates.lat}&lon=${weatherData.coordinates.lon}`
        );
        if (!aqiResponse.ok) throw new Error('Failed to fetch air quality');
        
        const aqi = await aqiResponse.json();
        displayAirQuality(aqi);
    } catch (error) {
        console.error('Error fetching air quality:', error);
    }
}

async function fetchAirQualityByCoords(lat, lon) {
    try {
        const response = await fetch(
            `${API_BASE_URL}/air-quality?lat=${lat}&lon=${lon}`
        );
        if (!response.ok) throw new Error('Failed to fetch air quality');
        
        const aqi = await response.json();
        displayAirQuality(aqi);
    } catch (error) {
        console.error('Error fetching air quality:', error);
    }
}

// Display weather
function displayWeather(data) {
    const iconUrl = `https://openweathermap.org/img/wn/${data.weather.icon}@4x.png`;
    const sunrise = new Date(data.sunrise).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const sunset = new Date(data.sunset).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const lastUpdate = new Date(data.timestamp).toLocaleTimeString();
    
    document.getElementById('cityName').textContent = `${data.city}, ${data.country}`;
    document.getElementById('lastUpdate').textContent = `Last updated: ${lastUpdate}`;
    document.getElementById('weatherIcon').src = iconUrl;
    document.getElementById('temp').textContent = Math.round(data.temperature.current);
    document.getElementById('weatherDesc').textContent = data.weather.description;
    document.getElementById('feelsLike').textContent = `Feels like ${Math.round(data.temperature.feels_like)}°C`;
    document.getElementById('humidity').textContent = `${data.humidity}%`;
    document.getElementById('windSpeed').textContent = `${data.wind.speed.toFixed(1)} m/s`;
    document.getElementById('pressure').textContent = `${data.pressure} hPa`;
    document.getElementById('visibility').textContent = `${(data.visibility / 1000).toFixed(1)} km`;
    document.getElementById('cloudiness').textContent = `${data.cloudiness}%`;
    document.getElementById('sunrise').textContent = sunrise;
    document.getElementById('minTemp').textContent = `${Math.round(data.temperature.min)}°C`;
    document.getElementById('maxTemp').textContent = `${Math.round(data.temperature.max)}°C`;
    
    elements.weatherSection.style.display = 'block';
}

// Display forecast
function displayForecast(forecast) {
    const container = document.getElementById('forecastContainer');
    container.innerHTML = '';
    
    forecast.items.forEach(item => {
        const date = new Date(item.timestamp);
        const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const iconUrl = `https://openweathermap.org/img/wn/${item.weather.icon}@2x.png`;
        
        const card = document.createElement('div');
        card.className = 'forecast-item';
        card.innerHTML = `
            <div class="time">${time}</div>
            <img src="${iconUrl}" alt="${item.weather.description}" class="icon">
            <div class="temp">${Math.round(item.temperature)}°C</div>
            <div class="desc">${item.weather.description}</div>
            <div style="font-size: 0.8em; color: #999; margin-top: 5px;">
                💧 ${item.humidity}% | 💨 ${item.wind_speed.toFixed(1)} m/s
            </div>
        `;
        container.appendChild(card);
    });
    
    elements.forecastSection.style.display = 'block';
}

// Display air quality
function displayAirQuality(data) {
    const aqiLevels = ['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor'];
    const aqiLevel = aqiLevels[data.aqi - 1] || 'Unknown';
    
    document.getElementById('aqi').textContent = `${aqiLevel} (${data.aqi})`;
    document.getElementById('pm2_5').textContent = `${data.pm2_5.toFixed(2)} µg/m³`;
    document.getElementById('pm10').textContent = `${data.pm10.toFixed(2)} µg/m³`;
    document.getElementById('no2').textContent = `${data.no2.toFixed(2)} µg/m³`;
    
    elements.aqiSection.style.display = 'block';
}

// Compare multiple cities
async function compareMultipleCities() {
    const citiesInput = elements.citiesInput.value.trim();
    if (!citiesInput) {
        showError('Please enter city names separated by commas');
        return;
    }
    
    const cities = citiesInput.split(',').map(c => c.trim());
    
    try {
        showLoading(true);
        hideError();
        
        const response = await fetch(`${API_BASE_URL}/weather/cities`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cities })
        });
        
        if (!response.ok) throw new Error('Failed to fetch weather for cities');
        
        const weatherList = await response.json();
        displayMultipleCities(weatherList);
    } catch (error) {
        showError(`Error comparing cities: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Display multiple cities
function displayMultipleCities(weatherList) {
    elements.citiesContainer.innerHTML = '';
    
    weatherList.forEach(weather => {
        const card = document.createElement('div');
        card.className = 'city-card';
        card.innerHTML = `
            <div class="city-card-header">📍 ${weather.city}, ${weather.country}</div>
            <div class="city-card-temp">${Math.round(weather.temperature.current)}°C</div>
            <div class="city-card-desc">${weather.weather.description}</div>
            <div class="city-card-info">
                💧 ${weather.humidity}% | 💨 ${weather.wind.speed.toFixed(1)} m/s
            </div>
            <div class="city-card-info" style="margin-top: 5px;">
                🌡️ ${Math.round(weather.temperature.min)}°C - ${Math.round(weather.temperature.max)}°C
            </div>
        `;
        elements.citiesContainer.appendChild(card);
    });
}

// UI Helpers
function showLoading(show) {
    elements.loading.style.display = show ? 'flex' : 'none';
}

function showError(message) {
    elements.error.textContent = message;
    elements.error.style.display = 'block';
}

function hideError() {
    elements.error.style.display = 'none';
}

// Load default city on page load
window.addEventListener('load', () => {
    elements.cityInput.value = 'London';
    searchByCity();
});
