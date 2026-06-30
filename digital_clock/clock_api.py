# Digital Clock - Multi-Timezone Display
# Real-time clock showing current time across multiple time zones

from datetime import datetime
import pytz
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json

# -------------------------
# SETUP
# -------------------------
app = FastAPI(
    title="Digital Clock API",
    description="Multi-timezone real-time clock display",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# DATA MODELS
# -------------------------
class TimeZoneInfo(BaseModel):
    """Timezone information"""
    timezone: str = Field(..., description="Timezone name (e.g., 'America/New_York')")
    display_name: str = Field(..., description="Display name (e.g., 'Eastern Time')")
    abbreviation: str = Field(..., description="Timezone abbreviation (e.g., 'EST')")
    utc_offset: str = Field(..., description="UTC offset (e.g., '-05:00')")

class ClockTime(BaseModel):
    """Time in specific timezone"""
    timezone: str = Field(..., description="Timezone name")
    display_name: str = Field(..., description="Display name")
    current_time: str = Field(..., description="Current time in HH:MM:SS format")
    current_date: str = Field(..., description="Current date in YYYY-MM-DD format")
    iso_format: str = Field(..., description="ISO format timestamp")
    unix_timestamp: float = Field(..., description="Unix timestamp")
    utc_offset: str = Field(..., description="UTC offset")
    is_dst: bool = Field(..., description="Is daylight saving time active")

class MultiClockResponse(BaseModel):
    """Multiple timezone clock display"""
    utc_time: str = Field(..., description="Current UTC time")
    clocks: List[ClockTime] = Field(..., description="Clocks for each timezone")
    timestamp: float = Field(..., description="Response timestamp")

class AlarmRequest(BaseModel):
    """Alarm request"""
    timezone: str = Field(..., description="Timezone for alarm")
    time: str = Field(..., description="Time in HH:MM format")
    label: str = Field(default="Alarm", description="Alarm label")
    enabled: bool = Field(default=True, description="Is alarm enabled")

# -------------------------
# TIMEZONE MANAGER
# -------------------------
class TimezoneManager:
    """Manages timezone operations"""
    
    # Popular timezones
    POPULAR_TIMEZONES = {
        "America/New_York": "Eastern Time (ET)",
        "America/Chicago": "Central Time (CT)",
        "America/Denver": "Mountain Time (MT)",
        "America/Los_Angeles": "Pacific Time (PT)",
        "Europe/London": "Greenwich Mean Time (GMT)",
        "Europe/Paris": "Central European Time (CET)",
        "Europe/Moscow": "Moscow Standard Time (MSK)",
        "Asia/Dubai": "Gulf Standard Time (GST)",
        "Asia/Kolkata": "Indian Standard Time (IST)",
        "Asia/Bangkok": "Indochina Time (ICT)",
        "Asia/Singapore": "Singapore Time (SGT)",
        "Asia/Hong_Kong": "Hong Kong Time (HKT)",
        "Asia/Tokyo": "Japan Standard Time (JST)",
        "Asia/Shanghai": "China Standard Time (CST)",
        "Australia/Sydney": "Australian Eastern Time (AEDT)",
        "Pacific/Auckland": "New Zealand Standard Time (NZST)",
        "UTC": "Coordinated Universal Time (UTC)",
    }
    
    @staticmethod
    def get_time_in_timezone(tz_name: str) -> ClockTime:
        """Get current time in timezone"""
        try:
            tz = pytz.timezone(tz_name)
            now = datetime.now(tz)
            
            # Get offset
            offset = now.strftime("%z")
            offset_formatted = f"{offset[:3]}:{offset[3:]}"
            
            # Check DST
            is_dst = bool(now.dst())
            
            display_name = TimezoneManager.POPULAR_TIMEZONES.get(
                tz_name,
                tz_name
            )
            
            return ClockTime(
                timezone=tz_name,
                display_name=display_name,
                current_time=now.strftime("%H:%M:%S"),
                current_date=now.strftime("%Y-%m-%d"),
                iso_format=now.isoformat(),
                unix_timestamp=now.timestamp(),
                utc_offset=offset_formatted,
                is_dst=is_dst
            )
        except Exception as e:
            raise ValueError(f"Invalid timezone: {tz_name}")
    
    @staticmethod
    def get_all_timezones() -> Dict[str, str]:
        """Get all available timezones"""
        return TimezoneManager.POPULAR_TIMEZONES
    
    @staticmethod
    def get_timezone_info(tz_name: str) -> TimeZoneInfo:
        """Get timezone information"""
        try:
            tz = pytz.timezone(tz_name)
            now = datetime.now(tz)
            
            offset = now.strftime("%z")
            offset_formatted = f"{offset[:3]}:{offset[3:]}"
            
            display_name = TimezoneManager.POPULAR_TIMEZONES.get(
                tz_name,
                tz_name
            )
            
            return TimeZoneInfo(
                timezone=tz_name,
                display_name=display_name,
                abbreviation=now.strftime("%Z"),
                utc_offset=offset_formatted
            )
        except Exception as e:
            raise ValueError(f"Invalid timezone: {tz_name}")
    
    @staticmethod
    def convert_time(from_tz: str, to_tz: str, time_str: str = None) -> Dict:
        """Convert time between timezones"""
        try:
            from_timezone = pytz.timezone(from_tz)
            to_timezone = pytz.timezone(to_tz)
            
            # If time not provided, use current time
            if time_str:
                # Parse time (HH:MM or HH:MM:SS)
                parts = time_str.split(":")
                if len(parts) == 2:
                    hour, minute = int(parts[0]), int(parts[1])
                    second = 0
                elif len(parts) == 3:
                    hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])
                else:
                    raise ValueError("Invalid time format")
                
                now = datetime.now(from_timezone)
                time_obj = from_timezone.localize(
                    datetime(now.year, now.month, now.day, hour, minute, second)
                )
            else:
                time_obj = datetime.now(from_timezone)
            
            converted = time_obj.astimezone(to_timezone)
            
            return {
                "original_time": time_obj.strftime("%H:%M:%S"),
                "original_timezone": from_tz,
                "converted_time": converted.strftime("%H:%M:%S"),
                "converted_timezone": to_tz,
                "original_offset": time_obj.strftime("%z"),
                "converted_offset": converted.strftime("%z"),
                "time_difference": {
                    "hours": (converted.hour - time_obj.hour) % 24,
                    "minutes": converted.minute - time_obj.minute
                }
            }
        except Exception as e:
            raise ValueError(f"Conversion error: {str(e)}")

# -------------------------
# ENDPOINTS - HEALTH
# -------------------------
@app.get("/", tags=["Health"])
async def root():
    """Health check"""
    return {
        "status": "Digital Clock API Online",
        "version": "1.0.0",
        "features": [
            "Multi-timezone display",
            "Real-time clock",
            "Timezone conversion",
            "Alarm management",
            "DST detection",
            "UTC offset tracking"
        ]
    }

# -------------------------
# ENDPOINTS - CLOCK
# -------------------------
@app.get("/time", response_model=ClockTime, tags=["Clock"])
async def get_time(timezone: str = "UTC"):
    """Get current time in timezone"""
    try:
        return TimezoneManager.get_time_in_timezone(timezone)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/time/multi", response_model=MultiClockResponse, tags=["Clock"])
async def get_multi_clock():
    """Get current time in multiple popular timezones"""
    try:
        utc_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
        clocks = [
            TimezoneManager.get_time_in_timezone(tz)
            for tz in TimezoneManager.POPULAR_TIMEZONES.keys()
        ]
        
        return MultiClockResponse(
            utc_time=utc_time,
            clocks=clocks,
            timestamp=datetime.now().timestamp()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/time/custom", response_model=MultiClockResponse, tags=["Clock"])
async def get_custom_clocks(timezones: str):
    """Get time in custom timezones
    
    Example: /time/custom?timezones=UTC,America/New_York,Asia/Tokyo
    """
    try:
        tz_list = [tz.strip() for tz in timezones.split(",")]
        utc_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
        clocks = [
            TimezoneManager.get_time_in_timezone(tz)
            for tz in tz_list
        ]
        
        return MultiClockResponse(
            utc_time=utc_time,
            clocks=clocks,
            timestamp=datetime.now().timestamp()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------
# ENDPOINTS - TIMEZONE INFO
# -------------------------
@app.get("/timezones", tags=["Timezones"])
async def list_timezones():
    """List all available timezones"""
    return {
        "timezones": TimezoneManager.get_all_timezones(),
        "count": len(TimezoneManager.get_all_timezones())
    }

@app.get("/timezone/{timezone}", response_model=TimeZoneInfo, tags=["Timezones"])
async def get_timezone_info(timezone: str):
    """Get timezone information"""
    try:
        return TimezoneManager.get_timezone_info(timezone)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------
# ENDPOINTS - CONVERSION
# -------------------------
@app.get("/convert", tags=["Conversion"])
async def convert_timezone(from_tz: str, to_tz: str, time: Optional[str] = None):
    """Convert time between timezones
    
    Example: /convert?from_tz=America/New_York&to_tz=Asia/Tokyo&time=14:30
    """
    try:
        return TimezoneManager.convert_time(from_tz, to_tz, time)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------
# ENDPOINTS - UTILITIES
# -------------------------
@app.get("/utc", tags=["Utilities"])
async def get_utc_time():
    """Get current UTC time"""
    utc_now = datetime.now(pytz.UTC)
    return {
        "utc_time": utc_now.strftime("%H:%M:%S"),
        "utc_date": utc_now.strftime("%Y-%m-%d"),
        "iso_format": utc_now.isoformat(),
        "unix_timestamp": utc_now.timestamp()
    }

@app.get("/timestamp/{timestamp}", tags=["Utilities"])
async def convert_timestamp(timestamp: float, timezone: str = "UTC"):
    """Convert Unix timestamp to datetime"""
    try:
        dt = datetime.fromtimestamp(timestamp, tz=pytz.timezone(timezone))
        return {
            "timestamp": timestamp,
            "timezone": timezone,
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "iso_format": dt.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/now-all", tags=["Utilities"])
async def get_all_info():
    """Get comprehensive time information"""
    utc_now = datetime.now(pytz.UTC)
    
    return {
        "utc_time": utc_now.strftime("%Y-%m-%d %H:%M:%S"),
        "unix_timestamp": utc_now.timestamp(),
        "iso_format": utc_now.isoformat(),
        "day_of_week": utc_now.strftime("%A"),
        "week_number": utc_now.isocalendar()[1],
        "day_of_year": utc_now.timetuple().tm_yday
    }

# -------------------------
# STATIC HTML FRONTEND
# -------------------------
@app.get("/clock", response_class=HTMLResponse, tags=["Frontend"])
async def clock_frontend():
    """Interactive clock frontend"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Digital Clock - Multi-Timezone</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Courier New', monospace;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 800px;
                width: 100%;
            }
            
            h1 {
                text-align: center;
                color: #667eea;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            
            .clocks-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .clock {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                padding: 20px;
                color: white;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.02); }
            }
            
            .clock.active {
                animation: pulse 1s infinite;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }
            
            .clock-timezone {
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 10px;
            }
            
            .clock-time {
                font-size: 2.5em;
                font-weight: bold;
                letter-spacing: 2px;
                margin: 10px 0;
            }
            
            .clock-date {
                font-size: 0.85em;
                opacity: 0.8;
                margin-top: 10px;
            }
            
            .clock-offset {
                font-size: 0.75em;
                opacity: 0.7;
                margin-top: 8px;
            }
            
            .controls {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                transition: all 0.3s;
                background: #667eea;
                color: white;
            }
            
            .btn:hover {
                background: #764ba2;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            .btn.secondary {
                background: #f5576c;
            }
            
            .btn.secondary:hover {
                background: #f093fb;
            }
            
            .info-box {
                background: #f5f5f5;
                border-left: 4px solid #667eea;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                font-size: 0.95em;
            }
            
            input[type="text"] {
                padding: 10px 15px;
                border: 2px solid #667eea;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 0.95em;
            }
            
            input[type="text"]:focus {
                outline: none;
                border-color: #764ba2;
                box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
            }
            
            .update-indicator {
                text-align: center;
                color: #667eea;
                font-size: 0.85em;
                margin-top: 20px;
            }
            
            .converting {
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🕐 Digital Clock</h1>
            
            <div class="controls">
                <button class="btn" onclick="refreshClocks()">🔄 Refresh</button>
                <button class="btn secondary" onclick="toggleView()">📊 Toggle View</button>
                <input type="text" id="customTZ" placeholder="Add timezone (e.g., Asia/Tokyo)" />
                <button class="btn" onclick="addTimezone()">➕ Add</button>
            </div>
            
            <div class="clocks-grid" id="clocks"></div>
            
            <div class="info-box">
                <strong>UTC Time:</strong> <span id="utcTime"></span> | 
                <strong>Timestamp:</strong> <span id="timestamp"></span>
            </div>
            
            <div class="update-indicator">
                ⏱️ Updates every second
            </div>
        </div>
        
        <script>
            const API_URL = 'http://localhost:8003';
            let customTimezones = [
                'UTC',
                'America/New_York',
                'Europe/London',
                'Asia/Tokyo'
            ];
            
            async function loadClocks() {
                try {
                    const tzQuery = customTimezones.join(',');
                    const response = await fetch(`${API_URL}/time/custom?timezones=${tzQuery}`);
                    const data = await response.json();
                    
                    displayClocks(data.clocks);
                    document.getElementById('utcTime').textContent = data.utc_time;
                    document.getElementById('timestamp').textContent = data.timestamp.toFixed(0);
                } catch (error) {
                    console.error('Error loading clocks:', error);
                }
            }
            
            function displayClocks(clocks) {
                const container = document.getElementById('clocks');
                container.innerHTML = '';
                
                clocks.forEach((clock, index) => {
                    const clockEl = document.createElement('div');
                    clockEl.className = 'clock';
                    if (index === 0) clockEl.classList.add('active');
                    
                    clockEl.innerHTML = `
                        <div class="clock-timezone">${clock.display_name}</div>
                        <div class="clock-time">${clock.current_time}</div>
                        <div class="clock-date">${clock.current_date}</div>
                        <div class="clock-offset">UTC ${clock.utc_offset}</div>
                    `;
                    
                    container.appendChild(clockEl);
                });
            }
            
            function refreshClocks() {
                loadClocks();
            }
            
            function toggleView() {
                const container = document.getElementById('clocks');
                container.style.gridTemplateColumns = 
                    container.style.gridTemplateColumns === 'repeat(2, 1fr)' 
                    ? 'repeat(auto-fit, minmax(250px, 1fr))'
                    : 'repeat(2, 1fr)';
            }
            
            function addTimezone() {
                const input = document.getElementById('customTZ');
                const tz = input.value.trim();
                
                if (tz && !customTimezones.includes(tz)) {
                    customTimezones.push(tz);
                    input.value = '';
                    loadClocks();
                } else {
                    alert('Invalid or duplicate timezone');
                }
            }
            
            // Initial load and update every second
            loadClocks();
            setInterval(loadClocks, 1000);
        </script>
    </body>
    </html>
    """

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    import uvicorn
    print("🕐 Starting Digital Clock API...")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
