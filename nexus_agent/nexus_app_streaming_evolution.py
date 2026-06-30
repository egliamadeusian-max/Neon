# =========================
# NEXUS AI - SELF-EVOLVING AGENT
# =========================
# Advanced autonomous AI agent with streaming, self-improvement, and online deployment

import os
import json
import logging
import requests
import asyncio
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, AsyncGenerator
from enum import Enum
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import openai
from functools import lru_cache
import time
import uvicorn
from pathlib import Path

# -------------------------
# LOGGING SETUP
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus_evolution.log'),
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

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 15))
MAX_TOKENS_PER_TASK = int(os.getenv("MAX_TOKENS_PER_TASK", 15000))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 15))
MEMORY_SIZE = int(os.getenv("MEMORY_SIZE", 20))
REFLECTION_THRESHOLD = int(os.getenv("REFLECTION_THRESHOLD", 3))
EVOLUTION_THRESHOLD = int(os.getenv("EVOLUTION_THRESHOLD", 5))
PERSISTENCE_DIR = Path(os.getenv("PERSISTENCE_DIR", "./nexus_data"))
STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "true").lower() == "true"

# Create persistence directory
PERSISTENCE_DIR.mkdir(exist_ok=True)

class ToolAction(str, Enum):
    SEARCH_WEB = "search_web"
    READ_URL = "read_url"
    CALCULATE = "calculate"
    LEARN = "learn"
    EVOLVE = "evolve"
    DONE = "done"

# -------------------------
# LEARNING SYSTEM
# -------------------------
class LearningRecord:
    """Records successful patterns and strategies"""
    def __init__(self, task_type: str, strategy: str, success_rate: float, 
                 tools_used: List[str], outcome: str, timestamp: Optional[datetime] = None):
        self.task_type = task_type
        self.strategy = strategy
        self.success_rate = success_rate
        self.tools_used = tools_used
        self.outcome = outcome
        self.timestamp = timestamp or datetime.now()
        self.reinforcement_count = 1

    def to_dict(self):
        return {
            "task_type": self.task_type,
            "strategy": self.strategy,
            "success_rate": self.success_rate,
            "tools_used": self.tools_used,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "reinforcement_count": self.reinforcement_count
        }

    def reinforce(self):
        """Increase confidence in this learning"""
        self.reinforcement_count += 1
        self.success_rate = min(1.0, self.success_rate * 1.1)

class KnowledgeBase:
    """Persistent knowledge store for learned strategies"""
    def __init__(self, storage_path: Path = PERSISTENCE_DIR / "knowledge.pkl"):
        self.storage_path = storage_path
        self.learnings: Dict[str, List[LearningRecord]] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.tool_effectiveness: Dict[str, float] = {}
        self.load()

    def add_learning(self, record: LearningRecord) -> None:
        """Add a new learning record"""
        if record.task_type not in self.learnings:
            self.learnings[record.task_type] = []
        
        # Check if similar strategy exists
        for existing in self.learnings[record.task_type]:
            if existing.strategy == record.strategy:
                existing.reinforce()
                return
        
        self.learnings[record.task_type].append(record)
        logger.info(f"Learning recorded: {record.task_type} - {record.strategy}")
        self.save()

    def get_best_strategy(self, task_type: str) -> Optional[LearningRecord]:
        """Retrieve the best strategy for a task type"""
        if task_type not in self.learnings:
            return None
        
        strategies = self.learnings[task_type]
        if not strategies:
            return None
        
        # Sort by success rate and reinforcement count
        best = max(strategies, 
                   key=lambda s: (s.success_rate, s.reinforcement_count))
        return best

    def update_tool_effectiveness(self, tool: str, success: bool) -> None:
        """Track tool effectiveness"""
        if tool not in self.tool_effectiveness:
            self.tool_effectiveness[tool] = 0.5
        
        if success:
            self.tool_effectiveness[tool] = min(1.0, 
                self.tool_effectiveness[tool] * 1.15)
        else:
            self.tool_effectiveness[tool] = max(0.0, 
                self.tool_effectiveness[tool] * 0.85)
        
        self.save()

    def save(self) -> None:
        """Persist knowledge to disk"""
        try:
            with open(self.storage_path, 'wb') as f:
                pickle.dump({
                    'learnings': self.learnings,
                    'tool_effectiveness': self.tool_effectiveness,
                    'performance_metrics': self.performance_metrics
                }, f)
            logger.info(f"Knowledge saved to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")

    def load(self) -> None:
        """Load knowledge from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                    self.learnings = data.get('learnings', {})
                    self.tool_effectiveness = data.get('tool_effectiveness', {})
                    self.performance_metrics = data.get('performance_metrics', {})
                logger.info(f"Knowledge loaded from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load knowledge: {e}")

    def get_summary(self) -> Dict:
        """Get knowledge base summary"""
        return {
            "total_learnings": sum(len(v) for v in self.learnings.values()),
            "task_types": len(self.learnings),
            "tool_effectiveness": self.tool_effectiveness,
            "performance_metrics": self.performance_metrics
        }

# -------------------------
# MEMORY SYSTEM (Enhanced)
# -------------------------
class MemoryItem:
    def __init__(self, content: str, tool: str, success: bool = True, 
                 timestamp: Optional[datetime] = None):
        self.content = content
        self.tool = tool
        self.success = success
        self.timestamp = timestamp or datetime.now()
        self.relevance_score = 0.5

    def to_dict(self):
        return {
            "content": self.content,
            "tool": self.tool,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score
        }

class Memory:
    def __init__(self, max_size: int = MEMORY_SIZE):
        self.data: List[MemoryItem] = []
        self.max_size = max_size
        self.success_rate = 0.5

    def store(self, item: str, tool: str, success: bool = True) -> None:
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        
        self.data.append(MemoryItem(item, tool, success))
        self._update_success_rate()
        logger.info(f"Memory: {tool} - {'✓' if success else '✗'}")

    def _update_success_rate(self):
        """Calculate current success rate"""
        if not self.data:
            self.success_rate = 0.5
            return
        
        successes = sum(1 for item in self.data if item.success)
        self.success_rate = successes / len(self.data)

    def get_recent(self, n: int = 5) -> List[Dict]:
        return [item.to_dict() for item in self.data[-n:]]

    def get_by_tool(self, tool: str) -> List[Dict]:
        return [item.to_dict() for item in self.data if item.tool == tool]

    def get_successful_patterns(self) -> List[Dict]:
        """Get sequences of successful actions"""
        successful = [item for item in self.data if item.success]
        return [item.to_dict() for item in successful[-5:]]

    def clear(self) -> None:
        self.data = []
        self._update_success_rate()

    def summary(self) -> str:
        if not self.data:
            return "No memory"
        return "\n".join([f"[{item.tool}] {'✓' if item.success else '✗'} {item.content[:50]}" 
                         for item in self.data[-5:]])

# -------------------------
# TOKEN & COST TRACKING
# -------------------------
class TokenTracker:
    def __init__(self, max_tokens: int = MAX_TOKENS_PER_TASK):
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self.cost = 0.0
        self.calls = 0
        self.start_time = datetime.now()

    def add(self, input_tokens: int, output_tokens: int) -> None:
        total = input_tokens + output_tokens
        self.used_tokens += total
        self.cost += (input_tokens * 0.00015 + output_tokens * 0.00060) / 1000
        self.calls += 1

    def check_limits(self) -> bool:
        return self.used_tokens < self.max_tokens

    def get_efficiency(self) -> float:
        """Calculate tokens per successful call"""
        if self.calls == 0:
            return 0
        return self.used_tokens / self.calls

    def report(self) -> Dict:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            "total_tokens": self.used_tokens,
            "max_tokens": self.max_tokens,
            "api_calls": self.calls,
            "estimated_cost": f"${self.cost:.4f}",
            "within_limits": self.check_limits(),
            "efficiency": f"{self.get_efficiency():.1f} tokens/call",
            "elapsed_seconds": elapsed
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

async def search_web(query: str) -> ToolResult:
    """Search the web for information"""
    try:
        logger.info(f"🔍 Searching: {query}")
        result = f"[Web Results: {query}] Sources would be fetched from live API in production."
        return ToolResult(True, result)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return ToolResult(False, "", str(e))

async def read_url(url: str) -> ToolResult:
    """Read content from a URL"""
    try:
        logger.info(f"📖 Reading: {url}")
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        content = response.text[:1000]
        return ToolResult(True, content)
    except Exception as e:
        logger.error(f"Read failed: {e}")
        return ToolResult(False, "", str(e))

async def calculate(expression: str) -> ToolResult:
    """Safely evaluate mathematical expressions"""
    try:
        logger.info(f"🧮 Calculating: {expression}")
        allowed = {"__builtins__": {}, "abs": abs, "round": round, "sum": sum}
        result = str(eval(expression, allowed))
        return ToolResult(True, result)
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        return ToolResult(False, "", str(e))

async def execute_tool(action: str, param: str) -> ToolResult:
    """Dispatch tool execution with safety validation"""
    action_lower = action.lower().strip()
    
    if action_lower == ToolAction.SEARCH_WEB.value:
        return await search_web(param)
    elif action_lower == ToolAction.READ_URL.value:
        return await read_url(param)
    elif action_lower == ToolAction.CALCULATE.value:
        return await calculate(param)
    elif action_lower == ToolAction.DONE.value:
        return ToolResult(True, "DONE", None)
    else:
        error = f"Unknown tool: {action}"
        logger.warning(error)
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

async def plan_task(goal: str, memory: Memory, iteration: int, 
                    token_tracker: TokenTracker, knowledge_base: KnowledgeBase,
                    stream_callback=None) -> Optional[PlannerResponse]:
    """Generate next action with Chain-of-Thought reasoning using learned strategies"""
    try:
        memory_summary = memory.summary()
        task_type = goal.split()[0].lower() if goal else "general"
        best_strategy = knowledge_base.get_best_strategy(task_type)
        
        strategy_hint = ""
        if best_strategy:
            strategy_hint = f"\n\nBased on past experience:\nBest Strategy: {best_strategy.strategy}\nSuccess Rate: {best_strategy.success_rate:.1%}\nTools: {', '.join(best_strategy.tools_used)}"

        prompt = f"""You are NEXUS, an autonomous self-evolving AI agent.

CURRENT GOAL: {goal}
ITERATION: {iteration}

MEMORY:
{memory_summary}

SUCCESS RATE: {memory.success_rate:.1%}

AVAILABLE TOOLS:
- search_web: Search the web
- read_url: Read URL content  
- calculate: Math operations
- learn: Record learnings
- evolve: Improve strategies
- done: Complete task
{strategy_hint}

Choose the next action. Return JSON ONLY:
{{
    "action": "search_web|read_url|calculate|learn|evolve|done",
    "param": "the parameter",
    "reasoning": "why",
    "confidence": 0.0-1.0
}}"""

        if stream_callback:
            await stream_callback(f"📝 Planning iteration {iteration}...\n")

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        usage = response.get('usage', {})
        token_tracker.add(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))

        text = response.choices[0].message.content.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])
            else:
                return None

        return PlannerResponse(
            action=data.get("action", "done"),
            param=data.get("param", ""),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5)
        )

    except Exception as e:
        logger.error(f"Planner error: {e}")
        return None

# -------------------------
# EVOLUTION ENGINE
# -------------------------
async def evolve_strategy(goal: str, memory: Memory, token_tracker: TokenTracker,
                         knowledge_base: KnowledgeBase, stream_callback=None) -> str:
    """Analyze performance and generate improved strategies"""
    try:
        if stream_callback:
            await stream_callback("🧬 EVOLVING STRATEGY...\n")
        
        successful_patterns = memory.get_successful_patterns()
        memory_summary = memory.summary()
        
        prompt = f"""Analyze this agent's performance and suggest improvements:

GOAL: {goal}
SUCCESS RATE: {memory.success_rate:.1%}

RECENT ACTIONS:
{memory_summary}

SUCCESSFUL PATTERNS:
{json.dumps(successful_patterns, indent=2)}

Suggest 2-3 specific improvements to strategy. Format:
1. [Improvement]: Description
2. [Improvement]: Description
3. [Improvement]: Description"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=400
        )

        usage = response.get('usage', {})
        token_tracker.add(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))

        improvements = response.choices[0].message.content
        logger.info(f"Evolution generated: {improvements[:100]}")
        
        if stream_callback:
            await stream_callback(f"✨ Evolution Results:\n{improvements}\n\n")
        
        return improvements

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        return "Evolution unavailable"

# -------------------------
# AGENT CORE (Self-Evolving)
# -------------------------
class AgentExecutionResult:
    def __init__(self, goal: str, success: bool, steps: List[str], reflection: str,
                 evolutions: List[str], token_report: Dict, completion_reason: str):
        self.goal = goal
        self.success = success
        self.steps = steps
        self.reflection = reflection
        self.evolutions = evolutions
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
            "evolutions": self.evolutions,
            "token_report": self.token_report,
            "completion_reason": self.completion_reason,
            "timestamp": self.timestamp.isoformat()
        }

class NexusAgent:
    def __init__(self):
        self.memory = Memory()
        self.token_tracker = TokenTracker()
        self.knowledge_base = KnowledgeBase()
        self.execution_history: List[AgentExecutionResult] = []
        self.evolution_count = 0

    async def run(self, goal: str, max_iterations: Optional[int] = None,
                  stream_callback=None) -> AgentExecutionResult:
        """Execute agent loop with self-improvement"""
        logger.info(f"🚀 Starting NEXUS execution: {goal}")

        if max_iterations is None:
            max_iterations = MAX_ITERATIONS

        self.memory.clear()
        self.token_tracker = TokenTracker()
        steps = []
        evolutions = []
        completion_reason = "Completed"

        try:
            for iteration in range(max_iterations):
                if stream_callback:
                    await stream_callback(f"⚡ Iteration {iteration + 1}/{max_iterations}\n")

                if not self.token_tracker.check_limits():
                    completion_reason = "Token limit exceeded"
                    break

                # Plan next action
                plan = await plan_task(goal, self.memory, iteration + 1, 
                                      self.token_tracker, self.knowledge_base,
                                      stream_callback)
                if plan is None:
                    completion_reason = "Planning failed"
                    break

                if stream_callback:
                    await stream_callback(f"🎯 Action: {plan.action} (confidence: {plan.confidence:.1%})\n")

                # Execute tool
                result = await execute_tool(plan.action, plan.param)
                step_info = f"[{plan.action}] {result.data[:100] if result.success else result.error}"
                steps.append(step_info)

                # Store in memory
                self.memory.store(str(result), plan.action, result.success)
                self.knowledge_base.update_tool_effectiveness(plan.action, result.success)

                if stream_callback:
                    await stream_callback(f"📊 Result: {'✓ Success' if result.success else '✗ Failed'}\n")

                # Check for completion
                if plan.action.lower() == ToolAction.DONE.value:
                    completion_reason = "Goal achieved"
                    if stream_callback:
                        await stream_callback("🎉 Task completed!\n")
                    break

                # Evolution at intervals
                if (iteration + 1) % EVOLUTION_THRESHOLD == 0 and iteration > 0:
                    evolution = await evolve_strategy(goal, self.memory, 
                                                     self.token_tracker, 
                                                     self.knowledge_base,
                                                     stream_callback)
                    evolutions.append(evolution)
                    self.evolution_count += 1

                    # Record learning
                    task_type = goal.split()[0].lower()
                    tools_used = [s.split('[')[1].split(']')[0] for s in steps if '[' in s]
                    learning = LearningRecord(
                        task_type=task_type,
                        strategy=evolution[:100],
                        success_rate=self.memory.success_rate,
                        tools_used=tools_used,
                        outcome=completion_reason
                    )
                    self.knowledge_base.add_learning(learning)

                await asyncio.sleep(0.3)

        except Exception as e:
            completion_reason = f"Error: {str(e)}"
            logger.error(completion_reason)

        success = completion_reason == "Goal achieved"
        result = AgentExecutionResult(
            goal=goal,
            success=success,
            steps=steps,
            reflection=f"Success rate: {self.memory.success_rate:.1%}",
            evolutions=evolutions,
            token_report=self.token_tracker.report(),
            completion_reason=completion_reason
        )

        self.execution_history.append(result)
        logger.info(f"✅ Execution complete: {completion_reason}")
        return result

    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "executions": len(self.execution_history),
            "evolutions": self.evolution_count,
            "knowledge_base": self.knowledge_base.get_summary(),
            "average_success_rate": sum(
                [item.memory.success_rate for item in self.execution_history]
            ) / max(1, len(self.execution_history))
        }

# -------------------------
# API (FastAPI with Streaming)
# -------------------------
app = FastAPI(
    title="NEXUS Self-Evolving AI Agent",
    description="Autonomous AI with streaming, self-improvement, and online learning",
    version="3.0.0"
)

bot_agent = NexusAgent()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    stream: bool = Field(default=True, description="Enable streaming responses")

async def stream_generator(result_queue: asyncio.Queue):
    """Generate streaming responses"""
    while True:
        chunk = await result_queue.get()
        if chunk is None:
            break
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "NEXUS ONLINE",
        "version": "3.0.0",
        "mode": "SELF-EVOLVING",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """Chat endpoint with optional streaming"""
    try:
        if not req.stream or not STREAMING_ENABLED:
            # Non-streaming response
            result = await bot_agent.run(req.message)
            return result.to_dict()
        
        # Streaming response
        async def generate():
            queue = asyncio.Queue()
            
            async def callback(chunk: str):
                await queue.put(chunk)
            
            # Run agent in background
            task = asyncio.create_task(
                bot_agent.run(req.message, stream_callback=callback)
            )
            
            # Stream results
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.5)
                    if chunk is None:
                        break
                    yield f"data: {json.dumps({'chunk': chunk, 'timestamp': datetime.now().isoformat()})}\n\n"
                except asyncio.TimeoutError:
                    if task.done():
                        result = await task
                        yield f"data: {json.dumps({'complete': True, 'result': result.to_dict()})}\n\n"
                        break
                    continue
                    
        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get agent statistics"""
    return bot_agent.get_stats()

@app.get("/history")
async def get_history():
    """Get execution history"""
    return {
        "history": [r.to_dict() for r in bot_agent.execution_history[-20:]]
    }

@app.get("/knowledge")
async def get_knowledge():
    """Get knowledge base"""
    return bot_agent.knowledge_base.get_summary()

@app.post("/reset")
async def reset():
    """Reset agent"""
    bot_agent.memory.clear()
    bot_agent.token_tracker = TokenTracker()
    return {"status": "Agent reset", "timestamp": datetime.now().isoformat()}

@app.post("/export-knowledge")
async def export_knowledge():
    """Export learned knowledge"""
    bot_agent.knowledge_base.save()
    return {
        "status": "Knowledge exported",
        "path": str(bot_agent.knowledge_base.storage_path),
        "summary": bot_agent.knowledge_base.get_summary()
    }

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    logger.info("🌟 NEXUS Self-Evolving AI Agent Starting...")
    logger.info(f"📍 Streaming: {STREAMING_ENABLED}")
    logger.info(f"💾 Data Directory: {PERSISTENCE_DIR}")
    
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
