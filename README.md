# NEXUS AI - Self-Evolving Agent
# README and Quick Start Guide

## 🌟 NEXUS: Autonomous Self-Evolving AI Agent

NEXUS is an advanced autonomous AI agent that:
- 🚀 **Streams responses** in real-time using Server-Sent Events
- 🧬 **Self-improves** through learned strategies and reflections
- 📚 **Persists knowledge** across sessions for continuous learning
- 🛠️ **Executes tools** (search, read, calculate) to accomplish goals
- 💰 **Tracks tokens** and costs to stay within budget
- 📊 **Analyzes performance** and adapts strategies automatically

---

## 🎯 Project Structure

```
Neon/
├── nexus_agent/
│   ├── nexus_app_enhanced.py           # v2.0 - Production-grade baseline
│   └��─ nexus_app_streaming_evolution.py # v3.0 - Streaming + Self-Evolution
├── nexus_data/                          # Persistent knowledge storage
│   └── knowledge.pkl                    # Learned strategies
├── requirements.txt                     # Python dependencies
├── DEPLOYMENT_GUIDE.md                 # Detailed deployment instructions
├── README.md                           # This file
└── .env.example                        # Environment template
```

---

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
# Copy template
cp .env.example .env

# Edit with your OpenAI API key
nano .env
```

### 3. Run the Agent
```bash
python nexus_agent/nexus_app_streaming_evolution.py
```

The agent will be available at `http://localhost:8000`

---

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/
```

### Stream Live Agent Thinking
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find information about AI agents",
    "stream": true
  }'
```

### Get Results Immediately
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Calculate 2^8",
    "stream": false
  }'
```

### View Learning Progress
```bash
curl http://localhost:8000/stats | jq
```

---

## 🌐 Online Deployment

### Option 1: Heroku (30 seconds)
```bash
heroku create nexus-ai
heroku config:set OPENAI_API_KEY=your-key
git push heroku main
```

### Option 2: Docker
```bash
docker build -t nexus-ai .
docker run -e OPENAI_API_KEY=your-key -p 8000:8000 nexus-ai
```

### Option 3: AWS EC2 / GCP / Azure
See detailed instructions in `DEPLOYMENT_GUIDE.md`

---

## 🧬 How Self-Evolution Works

### Phase 1: Execution
Agent executes tasks using available tools:
- 🔍 Search web for information
- 📖 Read URL content
- 🧮 Perform calculations

### Phase 2: Learning
Every 5 iterations, agent:
- 📊 Analyzes successful patterns
- 🎯 Identifies effective strategies
- 💾 Stores learnings persistently

### Phase 3: Evolution
Agent:
- 🔄 Reviews past strategies
- ✨ Generates improvements
- 📈 Adapts approach automatically

### Phase 4: Reinforcement
Successful strategies:
- 🏆 Gain higher confidence scores
- 📚 Are prioritized for future tasks
- 🚀 Compound effectiveness over time

---

## 📊 Key Features

### 1. Streaming Responses
Watch the agent think in real-time with Server-Sent Events (SSE)
```bash
# Stream endpoint automatically returns live updates
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Your task", "stream": true}'
```

### 2. Persistent Learning
Knowledge saved to disk and loaded on startup
- Tool effectiveness scores
- Task-specific strategies
- Performance metrics

### 3. Token Management
- Budget limits per task
- Cost estimation (GPT-4o-mini pricing)
- Efficiency tracking

### 4. Advanced Memory
- Success/failure tracking
- Pattern recognition
- FIFO eviction policy

### 5. Async Operations
Fully async implementation for:
- Concurrent API calls
- Responsive streaming
- Efficient resource usage

---

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Required
OPENAI_API_KEY=sk-...

# Limits
MAX_ITERATIONS=15              # Steps per task (default: 10)
MAX_TOKENS_PER_TASK=15000      # Token budget (default: 10000)
REQUEST_TIMEOUT=15             # API timeout in seconds

# Memory & Evolution
MEMORY_SIZE=20                 # Items to keep (default: 10)
REFLECTION_THRESHOLD=3         # Reflect every N iterations
EVOLUTION_THRESHOLD=5          # Evolve every N iterations

# Server
HOST=0.0.0.0                   # Bind address
PORT=8000                      # Port number
STREAMING_ENABLED=true         # Enable streaming responses
PERSISTENCE_DIR=./nexus_data   # Knowledge storage location
```

---

## 📈 Monitoring

### View Agent Statistics
```bash
curl http://localhost:8000/stats | jq
```

Response:
```json
{
  "executions": 42,
  "evolutions": 8,
  "knowledge_base": {
    "total_learnings": 23,
    "task_types": 12,
    "tool_effectiveness": {
      "search_web": 0.92,
      "read_url": 0.87,
      "calculate": 0.98
    }
  },
  "average_success_rate": 0.84
}
```

### View Execution History
```bash
curl http://localhost:8000/history | jq
```

### Export Knowledge
```bash
curl -X POST http://localhost:8000/export-knowledge | jq
```

---

## 🔧 Adding Custom Tools

Edit `nexus_app_streaming_evolution.py`:

```python
async def my_custom_tool(param: str) -> ToolResult:
    """Your custom tool implementation"""
    try:
        result = perform_operation(param)
        return ToolResult(True, result)
    except Exception as e:
        return ToolResult(False, "", str(e))

# Add to execute_tool function:
elif action_lower == "my_tool":
    return await my_custom_tool(param)
```

Then use it:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Use my_tool to do something"}'
```

---

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not set"
```bash
# Verify .env file exists and has the key
cat .env | grep OPENAI_API_KEY

# Or set directly
export OPENAI_API_KEY="sk-..."
```

### Issue: "Connection refused"
```bash
# Check if server is running
ps aux | grep uvicorn

# Verify port is free
lsof -i :8000
```

### Issue: Streaming not working
```bash
# Test with non-streaming first
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "stream": false}'
```

### Issue: Out of memory
```bash
# Reduce memory size in .env
MEMORY_SIZE=10
PERSISTENCE_DIR=/mnt/fast_storage
```

---

## 📚 Examples

### Example 1: Research Query
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Research the latest advancements in quantum computing and provide a summary"
  }'
```

### Example 2: Math Problem
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Calculate the sum of all prime numbers between 1 and 100"
  }'
```

### Example 3: Multi-Step Planning
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a detailed project plan for building a machine learning system"
  }'
```

---

## 🎓 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Guide](https://platform.openai.com/docs/api-reference)
- [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [Async Python](https://docs.python.org/3/library/asyncio.html)

---

## 🔐 Security Notes

1. **Never commit `.env`** - Add to `.gitignore`
2. **Validate all inputs** - Pydantic models do this automatically
3. **Rate limit in production** - Use nginx or cloud provider limits
4. **Use HTTPS** - Deploy behind reverse proxy
5. **Monitor costs** - Track OpenAI API usage

---

## 📈 Performance Tips

1. **Increase MAX_ITERATIONS** for complex tasks
2. **Adjust EVOLUTION_THRESHOLD** based on learning speed
3. **Use PERSISTENCE_DIR** on fast storage for large knowledge bases
4. **Enable streaming** for real-time feedback
5. **Monitor token usage** to optimize prompts

---

## 🤝 Contributing

We welcome improvements! Areas for enhancement:
- [ ] Additional tool implementations
- [ ] Improved evolution strategies
- [ ] Web dashboard for monitoring
- [ ] Multi-agent coordination
- [ ] Vector database for semantic search
- [ ] Fine-tuned model support

---

## 📝 License

Part of the Neon Artificial Cognitive Development Initiative

---

## 🚀 Next Steps

1. **Start local** with `python nexus_agent/nexus_app_streaming_evolution.py`
2. **Test API** with curl commands above
3. **Deploy online** using preferred platform
4. **Monitor** with `/stats` endpoint
5. **Extend** with custom tools

---

**NEXUS is ready to think, learn, and evolve! 🧠✨**

For detailed deployment instructions, see `DEPLOYMENT_GUIDE.md`
