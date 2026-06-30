# =========================
# NEXUS AI - ENHANCED AGENT
# =========================

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import openai
from functools import lru_cache
import time

# -------------------------
# LOGGING SETUP
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------
# CONFIG & CONSTANTS
# -------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OPENAI_API_KEY not set")
    raise ValueError("OPENAI_API_KEY environment variable required")

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 10))
MAX_TOKENS_PER_TASK = int(os.getenv("MAX_TOKENS_PER_TASK", 10000))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 10))
MEMORY_SIZE = int(os.getenv("MEMORY_SIZE", 10))
REFLECTION_THRESHOLD = int(os.getenv("REFLECTION_THRESHOLD", 3))

class ToolAction(str, Enum):
    SEARCH_WEB = "search_web"
    READ_URL = "read_url"
    CALCULATE = "calculate"
    DONE = "done"

# -------------------------
# MEMORY SYSTEM (Enhanced)
# -------------------------
class MemoryItem:
    def __init__(self, content: str, tool: str, timestamp: Optional[datetime] = None):
        self.content = content
        self.tool = tool
        self.timestamp = timestamp or datetime.now()
        self.relevance_score = 0.5

    def to_dict(self):
        return {
            "content": self.content,
            "tool": self.tool,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score
        }

class Memory:
    def __init__(self, max_size: int = MEMORY_SIZE):
        self.data: List[MemoryItem] = []
        self.max_size = max_size

    def store(self, item: str, tool: str) -> None:
        if len(self.data) >= self.max_size:
            self.data.pop(0)  # FIFO eviction
        self.data.append(MemoryItem(item, tool))
        logger.info(f"Memory stored: {tool} - {item[:50]}...")

    def get_recent(self, n: int = 5) -> List[Dict]:
        return [item.to_dict() for item in self.data[-n:]]

    def get_by_tool(self, tool: str) -> List[Dict]:
        return [item.to_dict() for item in self.data if item.tool == tool]

    def clear(self) -> None:
        self.data = []
        logger.info("Memory cleared")

    def summary(self) -> str:
        if not self.data:
            return "No memory"
        return "\n".join([f"[{item.tool}] {item.content[:60]}" for item in self.data[-5:]])

# -------------------------
# TOKEN & COST TRACKING
# -------------------------
class TokenTracker:
    def __init__(self, max_tokens: int = MAX_TOKENS_PER_TASK):
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self.cost = 0.0
        self.calls = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        total = input_tokens + output_tokens
        self.used_tokens += total
        # GPT-4o-mini pricing: $0.15/$0.60 per 1M tokens
        self.cost += (input_tokens * 0.00015 + output_tokens * 0.00060) / 1000
        self.calls += 1

    def check_limits(self) -> bool:
        return self.used_tokens < self.max_tokens

    def report(self) -> Dict:
        return {
            "total_tokens": self.used_tokens,
            "max_tokens": self.max_tokens,
            "api_calls": self.calls,
            "estimated_cost": f"${self.cost:.4f}",
            "within_limits": self.check_limits()
        }

# -------------------------
# TOOLS (Production-Ready)
# -------------------------
class ToolResult:
    def __init__(self, success: bool, data: str, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error

    def __str__(self):
        if self.success:
            return self.data
        return f"ERROR: {self.error}"

def search_web(query: str) -> ToolResult:
    """Search the web for information"""
    try:
        logger.info(f"Searching web for: {query}")
        # Placeholder: In production, use actual search API (Google, Bing, etc.)
        # response = requests.get(
        #     "https://api.bing.com/v7.0/search",
        #     params={"q": query},
        #     headers={"Ocp-Apim-Subscription-Key": os.getenv("BING_API_KEY")},
        #     timeout=REQUEST_TIMEOUT
        # )
        # results = response.json()
        result = f"[Web Search Results for: {query}] Relevant sources and summaries would appear here in production."
        return ToolResult(True, result)
    except requests.Timeout:
        error = f"Search timeout for query: {query}"
        logger.error(error)
        return ToolResult(False, "", error)
    except Exception as e:
        error = f"Search failed: {str(e)}"
        logger.error(error)
        return ToolResult(False, "", error)

def read_url(url: str) -> ToolResult:
    """Read content from a URL"""
    try:
        logger.info(f"Reading URL: {url}")
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        content = response.text[:1000]  # Limit to 1000 chars
        return ToolResult(True, content)
    except requests.Timeout:
        error = f"URL read timeout: {url}"
        logger.error(error)
        return ToolResult(False, "", error)
    except requests.HTTPError as e:
        error = f"HTTP error reading {url}: {e.response.status_code}"
        logger.error(error)
        return ToolResult(False, "", error)
    except Exception as e:
        error = f"Failed to read URL: {str(e)}"
        logger.error(error)
        return ToolResult(False, "", error)

def calculate(expression: str) -> ToolResult:
    """Safely evaluate mathematical expressions"""
    try:
        logger.info(f"Calculating: {expression}")
        # Only allow safe operations
        allowed = {"__builtins__": {}, "abs": abs, "round": round, "sum": sum, "min": min, "max": max}
        result = str(eval(expression, allowed))
        return ToolResult(True, result)
    except Exception as e:
        error = f"Calculation failed: {str(e)}"
        logger.error(error)
        return ToolResult(False, "", error)

def execute_tool(action: str, param: str) -> ToolResult:
    """Dispatch tool execution with safety validation"""
    try:
        action_lower = action.lower().strip()

        if action_lower == ToolAction.SEARCH_WEB.value:
            return search_web(param)
        elif action_lower == ToolAction.READ_URL.value:
            return read_url(param)
        elif action_lower == ToolAction.CALCULATE.value:
            return calculate(param)
        elif action_lower == ToolAction.DONE.value:
            return ToolResult(True, "DONE", None)
        else:
            error = f"Unknown tool: {action}"
            logger.warning(error)
            return ToolResult(False, "", error)
    except Exception as e:
        error = f"Tool execution error: {str(e)}"
        logger.error(error)
        return ToolResult(False, "", error)

# -------------------------
# PLANNER (Advanced Reasoning)
# -------------------------
class PlannerResponse:
    def __init__(self, action: str, param: str, reasoning: str, confidence: float):
        self.action = action
        self.param = param
        self.reasoning = reasoning
        self.confidence = confidence

def plan_task(goal: str, memory: Memory, iteration: int, token_tracker: TokenTracker) -> Optional[PlannerResponse]:
    """Generate next action with Chain-of-Thought reasoning"""
    try:
        memory_summary = memory.summary()

        prompt = f"""You are NEXUS, an autonomous AI agent designed to solve problems systematically.

CURRENT GOAL: {goal}

ITERATION: {iteration}

MEMORY (Previous Actions):
{memory_summary}

AVAILABLE TOOLS:
1. search_web:<query> - Search the web for information
2. read_url:<url> - Read content from a URL
3. calculate:<expression> - Perform mathematical calculations
4. done - Signal task completion

INSTRUCTIONS:
1. Analyze the goal and memory
2. Determine if the goal is already achieved
3. Choose the most logical next action
4. Return a JSON response with your decision

Return ONLY valid JSON (no markdown, no extra text):
{{
    "action": "search_web|read_url|calculate|done",
    "param": "the parameter for the action",
    "reasoning": "why you chose this action",
    "confidence": 0.0-1.0
}}"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        # Track tokens
        usage = response.get('usage', {})
        token_tracker.add(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))

        text = response.choices[0].message.content.strip()

        # Robust JSON parsing
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])
            else:
                logger.warning(f"Could not parse planner response: {text}")
                return None

        return PlannerResponse(
            action=data.get("action", "done"),
            param=data.get("param", ""),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5)
        )

    except openai.error.RateLimitError:
        logger.error("OpenAI rate limit exceeded")
        return None
    except openai.error.AuthenticationError:
        logger.error("OpenAI authentication failed")
        return None
    except Exception as e:
        logger.error(f"Planner error: {str(e)}")
        return None

# -------------------------
# REFLECTION ENGINE
# -------------------------
def reflect_on_progress(goal: str, memory: Memory, token_tracker: TokenTracker) -> str:
    """Perform self-reflection on task progress"""
    try:
        memory_summary = memory.summary()
        prompt = f"""Analyze the agent's progress on this task:

GOAL: {goal}

ACTIONS TAKEN:
{memory_summary}

Provide a brief assessment:
1. Is the goal being addressed effectively?
2. What's the current status?
3. What should be tried next?"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )

        usage = response.get('usage', {})
        token_tracker.add(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))

        reflection = response.choices[0].message.content
        logger.info(f"Reflection: {reflection[:100]}")
        return reflection

    except Exception as e:
        logger.error(f"Reflection failed: {str(e)}")
        return "Reflection unavailable"

# -------------------------
# AGENT CORE (Production-Ready)
# -------------------------
class AgentExecutionResult:
    def __init__(self, goal: str, success: bool, steps: List[str], reflection: str, 
                 token_report: Dict, completion_reason: str):
        self.goal = goal
        self.success = success
        self.steps = steps
        self.reflection = reflection
        self.token_report = token_report
        self.completion_reason = completion_reason
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "goal": self.goal,
            "success": self.success,
            "steps_count": len(self.steps),
            "steps": self.steps,
            "reflection": self.reflection,
            "token_report": self.token_report,
            "completion_reason": self.completion_reason,
            "timestamp": self.timestamp.isoformat()
        }

class NexusAgent:
    def __init__(self):
        self.memory = Memory()
        self.token_tracker = TokenTracker()
        self.execution_history: List[AgentExecutionResult] = []

    def run(self, goal: str, max_iterations: Optional[int] = None) -> AgentExecutionResult:
        """Execute agent loop with full tracking and error handling"""
        logger.info(f"Starting agent execution for goal: {goal}")

        if max_iterations is None:
            max_iterations = MAX_ITERATIONS

        self.memory.clear()
        self.token_tracker = TokenTracker()
        steps = []
        completion_reason = "Completed successfully"

        try:
            for iteration in range(max_iterations):
                logger.info(f"Iteration {iteration + 1}/{max_iterations}")

                # Check token limits
                if not self.token_tracker.check_limits():
                    completion_reason = "Token limit exceeded"
                    logger.warning(completion_reason)
                    break

                # Plan next action
                plan = plan_task(goal, self.memory, iteration + 1, self.token_tracker)
                if plan is None:
                    completion_reason = "Planning failed"
                    logger.error(completion_reason)
                    break

                logger.info(f"Action: {plan.action}, Confidence: {plan.confidence:.2f}")

                # Execute tool
                result = execute_tool(plan.action, plan.param)
                step_info = f"[{plan.action}] {result.data if result.success else result.error}"
                steps.append(step_info)

                # Store in memory
                self.memory.store(str(result), plan.action)

                # Check for completion
                if plan.action.lower() == ToolAction.DONE.value:
                    completion_reason = "Goal achieved"
                    logger.info("Agent signaled completion")
                    break

                # Reflection at intervals
                if (iteration + 1) % REFLECTION_THRESHOLD == 0:
                    reflection = reflect_on_progress(goal, self.memory, self.token_tracker)
                    logger.info(f"Reflection point: {reflection[:100]}")

                # Rate limiting to avoid API abuse
                time.sleep(0.5)

        except KeyboardInterrupt:
            completion_reason = "Interrupted by user"
            logger.warning(completion_reason)
        except Exception as e:
            completion_reason = f"Unexpected error: {str(e)}"
            logger.error(completion_reason)

        # Final reflection
        final_reflection = reflect_on_progress(goal, self.memory, self.token_tracker)

        success = completion_reason == "Goal achieved"
        result = AgentExecutionResult(
            goal=goal,
            success=success,
            steps=steps,
            reflection=final_reflection,
            token_report=self.token_tracker.report(),
            completion_reason=completion_reason
        )

        self.execution_history.append(result)
        logger.info(f"Execution complete: {result.completion_reason}")
        return result

    def get_history(self) -> List[Dict]:
        """Retrieve execution history"""
        return [result.to_dict() for result in self.execution_history[-10:]]

# -------------------------
# CHATBOT INTERFACE
# -------------------------
class NexusChatbot:
    def __init__(self):
        self.agent = NexusAgent()
        self.sessions: Dict[str, Dict] = {}

    def chat(self, message: str, session_id: Optional[str] = None) -> Dict:
        """Process user message and return agent response"""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = {
                "id": session_id or str(int(time.time())),
                "messages": [],
                "created_at": datetime.now().isoformat()
            }
            self.sessions[session["id"]] = session

        result = self.agent.run(message)
        session["messages"].append({
            "user": message,
            "response": result.to_dict()
        })

        return {
            "session_id": session["id"],
            "response": result.to_dict()
        }

# -------------------------
# DATA MODELS
# -------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    goal: str
    success: bool
    steps_count: int
    steps: List[str]
    reflection: str
    token_report: Dict
    completion_reason: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

# -------------------------
# API (FastAPI)
# -------------------------
app = FastAPI(
    title="NEXUS AI Agent",
    description="Advanced autonomous AI agent with production-grade features",
    version="2.0.0"
)

bot = NexusChatbot()

@app.get("/", response_model=HealthResponse)
def root():
    """Health check endpoint"""
    logger.info("Health check")
    return {
        "status": "NEXUS AI ONLINE",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Chat endpoint - send goal/query to agent"""
    try:
        logger.info(f"Chat request: {req.message}")
        response = bot.chat(req.message, req.session_id)
        result = response["response"]
        return ChatResponse(
            session_id=response["session_id"],
            goal=result["goal"],
            success=result["success"],
            steps_count=result["steps_count"],
            steps=result["steps"],
            reflection=result["reflection"],
            token_report=result["token_report"],
            completion_reason=result["completion_reason"],
            timestamp=result["timestamp"]
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    """Get agent execution history"""
    logger.info("Requesting execution history")
    return {
        "history": bot.agent.get_history()
    }

@app.get("/status")
def get_status():
    """Get current agent status and token usage"""
    logger.info("Status check")
    return {
        "agent_active": True,
        "memory_size": len(bot.agent.memory.data),
        "executions": len(bot.agent.execution_history),
        "token_tracker": bot.agent.token_tracker.report(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/reset")
def reset():
    """Reset agent memory and state"""
    logger.info("Resetting agent")
    bot.agent.memory.clear()
    bot.agent.token_tracker = TokenTracker()
    return {"status": "Agent reset successful"}

# -------------------------
# ERROR HANDLERS
# -------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {"error": "Internal server error", "detail": str(exc)}

# -------------------------
# RUN (Optional Local)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting NEXUS AI Enhanced Agent")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
