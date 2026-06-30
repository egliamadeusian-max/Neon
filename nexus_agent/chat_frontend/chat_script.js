// NEXUS Chat Frontend
const API_BASE_URL = 'http://localhost:8002';
const WS_BASE_URL = 'ws://localhost:8002';

let sessionId = null;
let messageCount = 0;
let useWebSocket = false;
let ws = null;

const elements = {
    chatMessages: document.getElementById('chatMessages'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    sessionId: document.getElementById('sessionId'),
    messageCount: document.getElementById('messageCount'),
    newChatBtn: document.getElementById('newChatBtn'),
    settingsBtn: document.getElementById('settingsBtn'),
    settingsModal: document.getElementById('settingsModal'),
    streamToggle: document.getElementById('streamToggle'),
    showIntentToggle: document.getElementById('showIntentToggle'),
    showSentimentToggle: document.getElementById('showSentimentToggle'),
    agentStatus: document.getElementById('agentStatus')
};

// ========================
// INITIALIZATION
// ========================

async function initChat() {
    try {
        // Create session
        const response = await fetch(`${API_BASE_URL}/session/create`, { method: 'POST' });
        const data = await response.json();
        sessionId = data.session_id;
        
        updateSessionInfo();
        connectWebSocket();
        addSystemMessage('Welcome to NEXUS Chat! 🤖');
    } catch (error) {
        console.error('Init error:', error);
        addSystemMessage('Failed to initialize chat');
    }
}

function connectWebSocket() {
    try {
        ws = new WebSocket(`${WS_BASE_URL}/ws/chat/${sessionId}`);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            useWebSocket = true;
            updateAgentStatus(true);
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            useWebSocket = false;
            updateAgentStatus(false);
        };
        
        ws.onclose = () => {
            console.log('WebSocket closed');
            useWebSocket = false;
            updateAgentStatus(false);
            // Reconnect after 3 seconds
            setTimeout(connectWebSocket, 3000);
        };
    } catch (error) {
        console.error('WebSocket connection error:', error);
        useWebSocket = false;
    }
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'thinking':
            addThinkingMessage(data.content, data.metadata);
            break;
        case 'stream':
            addStreamMessage(data.content);
            break;
        case 'response':
            addAssistantMessage(data.content, data.metadata);
            break;
        case 'complete':
            addSystemMessage(data.content);
            break;
        case 'error':
            addSystemMessage(`Error: ${data.content}`, 'error');
            break;
        case 'connection':
            addSystemMessage(data.content);
            break;
    }
}

// ========================
// EVENT LISTENERS
// ========================

elements.sendBtn.addEventListener('click', sendMessage);
elements.messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) sendMessage();
});

elements.newChatBtn.addEventListener('click', newChat);
elements.settingsBtn.addEventListener('click', openSettings);

document.querySelectorAll('.quick-action').forEach(btn => {
    btn.addEventListener('click', () => {
        elements.messageInput.value = btn.dataset.prompt;
        sendMessage();
    });
});

document.querySelector('.close').addEventListener('click', closeSettings);
window.addEventListener('click', (e) => {
    if (e.target === elements.settingsModal) closeSettings();
});

// ========================
// MESSAGE FUNCTIONS
// ========================

async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message) return;
    
    // Clear input
    elements.messageInput.value = '';
    
    // Add user message
    addUserMessage(message);
    
    // Send message
    if (useWebSocket) {
        ws.send(JSON.stringify({ message }));
    } else {
        sendViaREST(message);
    }
}

async function sendViaREST(message) {
    try {
        elements.sendBtn.disabled = true;
        
        if (elements.streamToggle.checked) {
            // Stream response
            const response = await fetch(`${API_BASE_URL}/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    session_id: sessionId,
                    stream: true
                })
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const text = decoder.decode(value);
                const lines = text.split('\n').filter(l => l.startsWith('data: '));
                
                for (const line of lines) {
                    const data = JSON.parse(line.substring(6));
                    handleWebSocketMessage(data);
                }
            }
        } else {
            // Regular response
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    session_id: sessionId,
                    stream: false
                })
            });
            
            const data = await response.json();
            addAssistantMessage(data.response, {
                success: data.success,
                intent: data.intent,
                sentiment: data.sentiment
            });
        }
    } catch (error) {
        console.error('Send error:', error);
        addSystemMessage(`Error: ${error.message}`, 'error');
    } finally {
        elements.sendBtn.disabled = false;
        updateSessionInfo();
    }
}

function addUserMessage(content) {
    const msgEl = createMessageElement('user', content);
    elements.chatMessages.appendChild(msgEl);
    scrollToBottom();
    messageCount++;
    updateMessageCount();
}

function addAssistantMessage(content, metadata = {}) {
    const msgEl = createMessageElement('assistant', content, metadata);
    elements.chatMessages.appendChild(msgEl);
    scrollToBottom();
    messageCount++;
    updateMessageCount();
}

function addThinkingMessage(content, metadata = {}) {
    const msgEl = createMessageElement('system', content, metadata);
    elements.chatMessages.appendChild(msgEl);
    scrollToBottom();
}

function addStreamMessage(content) {
    // Append to last message
    const lastMsg = elements.chatMessages.lastElementChild;
    if (lastMsg && lastMsg.classList.contains('message')) {
        const contentEl = lastMsg.querySelector('.message-content');
        if (contentEl) {
            contentEl.textContent += content;
        }
    }
    scrollToBottom();
}

function addSystemMessage(content, type = 'system') {
    const msgEl = createMessageElement(type, content);
    elements.chatMessages.appendChild(msgEl);
    scrollToBottom();
}

function createMessageElement(type, content, metadata = {}) {
    const msgEl = document.createElement('div');
    msgEl.className = `message ${type}`;
    
    let html = `<div class="message-content">${escapeHtml(content)}</div>`;
    
    if (metadata && Object.keys(metadata).length > 0) {
        let metaHtml = '<div class="message-meta">';
        
        if (metadata.intent && elements.showIntentToggle.checked) {
            metaHtml += `<span>🎯 Intent: ${metadata.intent}</span> `;
        }
        
        if (metadata.sentiment && elements.showSentimentToggle.checked) {
            metaHtml += `<span>💭 ${metadata.sentiment.sentiment} (${(metadata.sentiment.score * 100).toFixed(0)}%)</span>`;
        }
        
        if (metadata.success !== undefined) {
            metaHtml += `<span>${metadata.success ? '✅' : '⚠️'}</span>`;
        }
        
        metaHtml += '</div>';
        html += metaHtml;
    }
    
    msgEl.innerHTML = html;
    return msgEl;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// ========================
// UI FUNCTIONS
// ========================

function updateSessionInfo() {
    elements.sessionId.textContent = `Session: ${sessionId?.substring(0, 8)}...`;
}

function updateMessageCount() {
    elements.messageCount.textContent = `Messages: ${messageCount}`;
}

function updateAgentStatus(online) {
    elements.agentStatus.className = online ? 'status-online' : 'status-offline';
    elements.agentStatus.textContent = online ? '● Online' : '● Offline';
}

async function newChat() {
    if (confirm('Start new chat?')) {
        messageCount = 0;
        elements.chatMessages.innerHTML = '';
        await initChat();
    }
}

function openSettings() {
    elements.settingsModal.style.display = 'block';
}

function closeSettings() {
    elements.settingsModal.style.display = 'none';
}

// ========================
// INIT
// ========================

window.addEventListener('load', initChat);
