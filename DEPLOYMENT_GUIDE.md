# NEXUS AI - Self-Evolving Agent Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API Key
- pip or conda

### Installation

```bash
# Clone the repository
cd egliamadeusian-max/Neon

# Install dependencies
pip install fastapi uvicorn pydantic openai requests python-dotenv aiofiles

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your-key-here
MAX_ITERATIONS=15
MAX_TOKENS_PER_TASK=15000
REQUEST_TIMEOUT=15
MEMORY_SIZE=20
REFLECTION_THRESHOLD=3
EVOLUTION_THRESHOLD=5
STREAMING_ENABLED=true
PERSISTENCE_DIR=./nexus_data
HOST=0.0.0.0
PORT=8000
EOF

# Run the agent
python nexus_agent/nexus_app_streaming_evolution.py
```

---

## 🌐 Online Deployment

### Option 1: Heroku (Recommended for Quick Start)

```bash
# Create Heroku app
heroku create nexus-ai-agent

# Add buildpack
heroku buildpacks:add heroku/python

# Set environment variables
heroku config:set OPENAI_API_KEY=your-key-here
heroku config:set STREAMING_ENABLED=true

# Create Procfile
echo "web: uvicorn nexus_agent.nexus_app_streaming_evolution:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

### Option 2: AWS EC2

```bash
# Launch EC2 instance (Ubuntu 22.04)
# SSH into instance

# Install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv git

# Clone repo
git clone https://github.com/egliamadeusian-max/Neon.git
cd Neon

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Create .env
nano .env

# Run with supervisor
sudo apt install supervisor

cat > /etc/supervisor/conf.d/nexus.conf << EOF
[program:nexus]
directory=/home/ubuntu/Neon
command=/home/ubuntu/Neon/venv/bin/python -m uvicorn nexus_agent.nexus_app_streaming_evolution:app --host 0.0.0.0 --port 8000
autostart=true
autorestart=true
stderr_logfile=/var/log/nexus.err.log
stdout_logfile=/var/log/nexus.out.log
EOF

sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start nexus
```

### Option 3: Docker (Production)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "nexus_agent.nexus_app_streaming_evolution:app", "--host", "0.0.0.0", "--port", "8000"]
```

Deploy:
```bash
# Build image
docker build -t nexus-ai .

# Run container
docker run -e OPENAI_API_KEY=your-key -p 8000:8000 nexus-ai

# Or use Docker Compose
cat > docker-compose.yml << EOF
version: '3.8'
services:
  nexus:
    build: .
    ports:
      - "8000:8000"
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      STREAMING_ENABLED: "true"
    volumes:
      - ./nexus_data:/app/nexus_data
    restart: always
EOF

docker-compose up -d
```

---

## 📊 API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Chat with Streaming
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find information about AI agents",
    "stream": true
  }'
```

### Non-Streaming Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Calculate 2^10",
    "stream": false
  }'
```

### Get Agent Statistics
```bash
curl http://localhost:8000/stats
```

### View Execution History
```bash
curl http://localhost:8000/history
```

### View Knowledge Base
```bash
curl http://localhost:8000/knowledge
```

### Export Learned Knowledge
```bash
curl -X POST http://localhost:8000/export-knowledge
```

### Reset Agent
```bash
curl -X POST http://localhost:8000/reset
```

---

## 🧬 Features Overview

### 1. **Streaming Responses**
- Real-time event streaming using Server-Sent Events (SSE)
- Watch agent think and evolve live
- Perfect for web frontends

### 2. **Self-Evolution**
- Agent learns from successful patterns
- Automatically improves strategies
- Knowledge persists across sessions

### 3. **Knowledge Base**
- Persistent learning records
- Tool effectiveness tracking
- Task-specific strategies

### 4. **Advanced Memory**
- Success/failure tracking
- Pattern recognition
- FIFO eviction policy

### 5. **Token Management**
- Cost tracking per task
- Budget limits
- Efficiency metrics

### 6. **Multiple Tools**
- Web search
- URL reading
- Mathematical calculations
- Extensible framework

---

## 🔧 Configuration

Edit `.env` file to customize:

```env
# OpenAI
OPENAI_API_KEY=your-key-here

# Limits
MAX_ITERATIONS=15              # Max steps per task
MAX_TOKENS_PER_TASK=15000      # Token budget
REQUEST_TIMEOUT=15             # API timeout (seconds)
MEMORY_SIZE=20                 # Memory capacity
REFLECTION_THRESHOLD=3         # Reflect every N iterations
EVOLUTION_THRESHOLD=5          # Evolve every N iterations

# Server
HOST=0.0.0.0
PORT=8000
STREAMING_ENABLED=true
PERSISTENCE_DIR=./nexus_data
```

---

## 📈 Monitoring & Logging

View logs in real-time:
```bash
tail -f nexus_evolution.log
```

Logs include:
- Agent iterations
- Tool executions
- Learning events
- Evolution triggers
- Error tracking

---

## 🔐 Security Considerations

1. **API Keys**: Never commit `.env` files
2. **Rate Limiting**: Implement rate limiters in production
3. **Input Validation**: All inputs are validated via Pydantic
4. **Safe Eval**: Math operations use restricted namespace
5. **HTTPS**: Use reverse proxy (nginx) in production

---

## 🚀 Advanced Usage

### Custom Tools
Add new tools in `execute_tool()`:

```python
elif action_lower == "custom_tool":
    return await custom_tool_function(param)
```

### Persistence
Knowledge is auto-saved to `nexus_data/knowledge.pkl`

### Multi-Agent Coordination
Run multiple instances on different ports:
```bash
PORT=8000 python nexus_agent/nexus_app_streaming_evolution.py
PORT=8001 python nexus_agent/nexus_app_streaming_evolution.py
```

---

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not set"
```bash
# Set environment variable
export OPENAI_API_KEY="your-key"
```

### Issue: Token limit exceeded
```bash
# Increase token budget in .env
MAX_TOKENS_PER_TASK=20000
```

### Issue: Streaming not working
```bash
# Check if streaming enabled
STREAMING_ENABLED=true

# Test endpoint
curl http://localhost:8000/
```

### Issue: Knowledge not persisting
```bash
# Check directory permissions
chmod 777 nexus_data/
```

---

## 📚 Example Workflows

### Task 1: Research & Summarize
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find the latest developments in quantum computing"
  }'
```

### Task 2: Mathematical Analysis
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Calculate fibonacci(20) and analyze the result"
  }'
```

### Task 3: Multi-Step Planning
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a marketing strategy for a tech startup"
  }'
```

---

## 📊 Performance Metrics

After tasks complete, check stats:

```bash
curl http://localhost:8000/stats | jq
```

Returns:
- Total executions
- Evolution count
- Knowledge base size
- Average success rate
- Tool effectiveness scores

---

## 🎯 Next Steps

1. **Deploy online** using preferred platform
2. **Monitor performance** with provided endpoints
3. **Customize tools** for your use case
4. **Share feedback** and improvements

---

## 📝 License

This project is part of the Neon Artificial Cognitive Development initiative.

---

## 🤝 Support

For issues and questions:
- Check `nexus_evolution.log`
- Review endpoint responses
- Verify `.env` configuration
- Test with non-streaming mode first

---

**NEXUS AI is now online and evolving! 🚀**
