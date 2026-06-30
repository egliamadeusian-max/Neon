# NEXUS Chat API - Enhanced FastAPI wrapper with chat features
# Integrates chat_integration.py with the streaming evolution agent

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import asyncio
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Import NEXUS components
from nexus_app_streaming_evolution import NexusAgent, TokenTracker, Memory
from chat_integration import (
    ChatSessionManager, ConversationProcessor, IntentRecognizer,
    ContextManager, SentimentAnalyzer, ChatMessage, MessageType,
    ChatRequest, ChatResponse, ConversationAnalysis
)

# -------------------------
# SETUP
# -------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NEXUS Chat API",
    description="Real-time conversational AI with self-learning capabilities",
    version="4.0.0"
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
# INITIALIZATION
# -------------------------
agent = NexusAgent()
session_manager = ChatSessionManager(max_sessions=100)
conversation_processor = ConversationProcessor(agent)
context_manager = ContextManager()
intent_recognizer = IntentRecognizer()
sentiment_analyzer = SentimentAnalyzer()

# -------------------------
# DATA MODELS
# -------------------------
class ChatInput(BaseModel):
    """Chat input"""
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None
    stream: bool = True

class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    message_count: int
    last_activity: datetime

# -------------------------
# ENDPOINTS - HEALTH
# -------------------------
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "NEXUS Chat API Online",
        "version": "4.0.0",
        "mode": "CONVERSATIONAL",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "Real-time chat",
            "WebSocket streaming",
            "Intent recognition",
            "Sentiment analysis",
            "Conversation analysis",
            "Self-learning agent"
        ]
    }

# -------------------------
# ENDPOINTS - SESSIONS
# -------------------------
@app.post("/session/create", tags=["Sessions"])
async def create_session(user_id: Optional[str] = None):
    """Create new chat session"""
    session = await session_manager.create_session(user_id)
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "created_at": session.created_at.isoformat(),
        "message": "✅ Session created successfully"
    }

@app.get("/session/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Get session details"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()

@app.get("/sessions", tags=["Sessions"])
async def get_all_sessions():
    """Get all sessions"""
    sessions = await session_manager.get_all_sessions()
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }

@app.delete("/session/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    """Delete session"""
    success = await session_manager.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": "Session deleted", "session_id": session_id}

# -------------------------
# ENDPOINTS - CHAT (REST)
# -------------------------
@app.post("/chat", tags=["Chat"])
async def chat(request: ChatInput):
    """Send chat message (non-streaming)"""
    try:
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session = await session_manager.create_session()
            session_id = session.session_id
        else:
            session = await session_manager.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Add user message
        user_msg = ChatMessage(
            type=MessageType.USER,
            content=request.message
        )
        await session_manager.add_message(session_id, user_msg)
        
        # Recognize intent
        intent = intent_recognizer.recognize(request.message)
        
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze(request.message)
        
        # Add thinking message
        thinking_msg = ChatMessage(
            type=MessageType.THINKING,
            content=f"{intent_recognizer.get_intent_description(intent)} | Sentiment: {sentiment['sentiment']}"
        )
        await session_manager.add_message(session_id, thinking_msg)
        
        # Process with agent
        result = await agent.run(request.message)
        
        # Create response message
        response_content = "\n".join(result.steps[-2:]) if result.steps else "No response"
        response_msg = ChatMessage(
            type=MessageType.ASSISTANT,
            content=response_content,
            metadata={
                "success": result.success,
                "intent": intent,
                "sentiment": sentiment,
                "reasoning": result.reflection[:200]
            }
        )
        await session_manager.add_message(session_id, response_msg)
        
        return {
            "session_id": session_id,
            "message_id": response_msg.id,
            "response": response_content,
            "intent": intent,
            "sentiment": sentiment,
            "success": result.success,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatInput):
    """Stream chat response"""
    try:
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session = await session_manager.create_session()
            session_id = session.session_id
        
        # Add user message
        user_msg = ChatMessage(
            type=MessageType.USER,
            content=request.message
        )
        await session_manager.add_message(session_id, user_msg)
        
        # Stream generator
        async def generate():
            try:
                async for chunk in conversation_processor.process_message(session_id, request.message):
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {{\"type\": \"error\", \"content\": \"{str(e)}\"}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# ENDPOINTS - WEBSOCKET
# -------------------------
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket chat endpoint for real-time bidirectional communication"""
    try:
        # Get or create session
        session = await session_manager.get_session(session_id)
        if not session:
            session = await session_manager.create_session()
            session_id = session.session_id
        
        # Connect
        await session_manager.connect(session_id, websocket)
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "content": "✅ Connected to NEXUS Chat",
            "session_id": session_id
        })
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message:
                continue
            
            # Add user message
            user_msg = ChatMessage(
                type=MessageType.USER,
                content=user_message
            )
            await session_manager.add_message(session_id, user_msg)
            
            # Recognize intent and sentiment
            intent = intent_recognizer.recognize(user_message)
            sentiment = sentiment_analyzer.analyze(user_message)
            
            # Send thinking indicator
            await websocket.send_json({
                "type": "thinking",
                "content": f"🤔 {intent_recognizer.get_intent_description(intent)}",
                "metadata": {"intent": intent, "sentiment": sentiment}
            })
            
            # Process with agent
            async def ws_callback(chunk: str):
                await websocket.send_json({
                    "type": "stream",
                    "content": chunk
                })
            
            result = await agent.run(user_message, stream_callback=ws_callback)
            
            # Send response
            response_content = "\n".join(result.steps[-2:]) if result.steps else "Processing..."
            response_msg = ChatMessage(
                type=MessageType.ASSISTANT,
                content=response_content,
                metadata={
                    "success": result.success,
                    "reasoning": result.reflection
                }
            )
            await session_manager.add_message(session_id, response_msg)
            
            # Send complete message
            await websocket.send_json({
                "type": "response",
                "content": response_content,
                "success": result.success,
                "message_id": response_msg.id
            })
            
            # Send completion indicator
            await websocket.send_json({
                "type": "complete",
                "content": "✅ Response complete"
            })
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session {session_id}")
        await session_manager.disconnect(session_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": str(e)
            })
        except:
            pass
        await session_manager.disconnect(session_id, websocket)

# -------------------------
# ENDPOINTS - ANALYSIS
# -------------------------
@app.get("/analyze/conversation/{session_id}", tags=["Analysis"])
async def analyze_conversation(session_id: str) -> ConversationAnalysis:
    """Analyze conversation quality"""
    try:
        analysis = await conversation_processor.analyze_conversation(session_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/conversation/{session_id}", tags=["Analysis"])
async def conversation_summary(session_id: str):
    """Get conversation summary"""
    summary = conversation_processor.get_conversation_summary(session_id)
    return summary

@app.get("/intent/{message}", tags=["Analysis"])
async def detect_intent(message: str):
    """Detect intent of message"""
    intent = intent_recognizer.recognize(message)
    return {
        "message": message,
        "intent": intent,
        "description": intent_recognizer.get_intent_description(intent)
    }

@app.get("/sentiment/{message}", tags=["Analysis"])
async def analyze_sentiment(message: str):
    """Analyze sentiment of message"""
    sentiment = sentiment_analyzer.analyze(message)
    return {
        "message": message,
        "sentiment": sentiment["sentiment"],
        "score": sentiment["score"],
        "details": {
            "positive_words": sentiment["positive_count"],
            "negative_words": sentiment["negative_count"]
        }
    }

# -------------------------
# ENDPOINTS - AGENT STATS
# -------------------------
@app.get("/agent/stats", tags=["Agent"])
async def agent_stats():
    """Get agent statistics"""
    return agent.get_stats()

@app.get("/agent/knowledge", tags=["Agent"])
async def agent_knowledge():
    """Get knowledge base"""
    return agent.knowledge_base.get_summary()

@app.get("/agent/history", tags=["Agent"])
async def agent_history():
    """Get execution history"""
    return {
        "executions": [r.to_dict() for r in agent.execution_history[-10:]]
    }

@app.post("/agent/reset", tags=["Agent"])
async def reset_agent():
    """Reset agent"""
    agent.memory.clear()
    agent.token_tracker = TokenTracker()
    return {"status": "Agent reset", "timestamp": datetime.utcnow().isoformat()}

# -------------------------
# ENDPOINTS - PREFERENCES
# -------------------------
@app.post("/preferences/{user_id}", tags=["Preferences"])
async def set_preferences(user_id: str, preferences: Dict[str, Any]):
    """Set user preferences"""
    context_manager.update_user_preferences(user_id, preferences)
    return {
        "user_id": user_id,
        "preferences": preferences,
        "status": "Updated"
    }

@app.get("/preferences/{user_id}", tags=["Preferences"])
async def get_preferences(user_id: str):
    """Get user preferences"""
    prefs = context_manager.get_user_preferences(user_id)
    return {"user_id": user_id, "preferences": prefs}

# -------------------------
# ERROR HANDLERS
# -------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.utcnow().isoformat()
    }

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting NEXUS Chat API v4.0.0")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
