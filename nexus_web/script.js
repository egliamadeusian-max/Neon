const CHAT_API = 'http://localhost:8002';

let sessionId = null;
let messageCount = 0;

// Initialize chat when page loads
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${CHAT_API}/session/create`, { method: 'POST' });
        const data = await response.json();
        sessionId = data.session_id;
        console.log('Chat session created:', sessionId);
    } catch (error) {
        console.error('Failed to create session:', error);
        addChatMessage('system', 'Failed to connect to chat service. Please refresh the page.');
    }
});

function setMessage(message) {
    document.getElementById('messageInput').value = message;
}

function scrollToChat() {
    document.getElementById('chat').scrollIntoView({ behavior: 'smooth' });
    document.getElementById('messageInput').focus();
}

async function sendChatMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();

    if (!message) return;

    // Add user message
    addChatMessage('user', message);
    input.value = '';

    if (!sessionId) {
        addChatMessage('system', 'Error: Not connected to chat service');
        return;
    }

    try {
        const streamEnabled = document.getElementById('streamToggle').checked;

        if (streamEnabled) {
            // Stream response
            const response = await fetch(`${CHAT_API}/chat/stream`, {
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
            let assistantMessage = '';
            let messageEl = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const text = decoder.decode(value);
                const lines = text.split('\n').filter(l => l.startsWith('data: '));

                for (const line of lines) {
                    try {
                        const data = JSON.parse(line.substring(6));

                        if (data.type === 'thinking') {
                            addChatMessage('system', `🤔 ${data.content}`);
                        } else if (data.type === 'response') {
                            if (!messageEl) {
                                messageEl = addChatMessage('assistant', data.content, true);
                            } else {
                                messageEl.querySelector('.message-content').textContent = data.content;
                            }
                        } else if (data.type === 'stream') {
                            if (!messageEl) {
                                messageEl = addChatMessage('assistant', data.content, true);
                            } else {
                                const content = messageEl.querySelector('.message-content');
                                content.textContent += data.content;
                            }
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            }
        } else {
            // Non-streaming response
            const response = await fetch(`${CHAT_API}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    session_id: sessionId,
                    stream: false
                })
            });

            const data = await response.json();
            addChatMessage('assistant', data.response);

            if (document.getElementById('showIntentToggle').checked) {
                addChatMessage('system', `🎯 Intent: ${data.intent}`);
            }
        }
    } catch (error) {
        console.error('Chat error:', error);
        addChatMessage('system', `Error: ${error.message}`);
    }
}

function addChatMessage(role, content, noScroll = false) {
    const container = document.getElementById('chatMessages');
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';
    contentEl.textContent = content;

    messageEl.appendChild(contentEl);
    container.appendChild(messageEl);

    if (!noScroll) {
        container.scrollTop = container.scrollHeight;
    }

    return messageEl;
}

// Allow Enter+Ctrl to send message
document.getElementById('messageInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        sendChatMessage();
    }
});
