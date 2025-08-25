// Chatbot JavaScript functionality
class ChatbotUI {
    constructor() {
        this.conversationId = this.generateConversationId();
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.chatMessages = document.getElementById('chatMessages');
        this.charCount = document.getElementById('charCount');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        
        this.initializeEventListeners();
        this.updateCharCount();
    }

    generateConversationId() {
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    initializeEventListeners() {
        // Send message on Enter key
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Update character count
        this.messageInput.addEventListener('input', () => {
            this.updateCharCount();
        });

        // Auto-resize input (optional)
        this.messageInput.addEventListener('input', () => {
            this.autoResizeInput();
        });
    }

    updateCharCount() {
        const currentLength = this.messageInput.value.length;
        const maxLength = this.messageInput.maxLength;
        this.charCount.textContent = `${currentLength}/${maxLength}`;
        
        // Change color based on length
        if (currentLength > maxLength * 0.8) {
            this.charCount.style.color = '#ff6b6b';
        } else if (currentLength > maxLength * 0.6) {
            this.charCount.style.color = '#f39c12';
        } else {
            this.charCount.style.color = '#666';
        }
    }

    autoResizeInput() {
        // Optional: Auto-resize input field
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        // Disable input and send button
        this.setInputState(false);
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.updateCharCount();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // Send message to API
            const response = await this.sendMessageToAPI(message);
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add bot response to chat
            this.addMessage(response.response, 'bot', response.metadata);
            
            // Update analytics if available
            // Only show ticket creation when the backend explicitly creates one
            // Don't automatically decide based on frontend logic
            if (response.metadata && response.metadata.ticket_id && response.metadata.intent === 'off-topic') {
                // Only show ticket creation if the backend explicitly says it's off-topic
                const isOffTopic = true;
                const hasContent = response.metadata.relevant_chunks > 0;
                this.showTicketCreated(response.metadata.ticket_id, isOffTopic, hasContent);
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
        
        // Re-enable input and send button
        this.setInputState(true);
        
        // Scroll to bottom
        this.scrollToBottom();
    }

    async sendMessageToAPI(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_id: this.conversationId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    addMessage(text, sender, metadata = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        
        if (sender === 'bot') {
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
        } else {
            avatar.innerHTML = '<i class="fas fa-user"></i>';
        }
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.innerHTML = this.formatMessage(text);
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = this.getCurrentTime();
        
        content.appendChild(messageText);
        content.appendChild(messageTime);
        
        // Add metadata if available
        if (metadata && sender === 'bot') {
            const metadataDiv = this.createMetadataDisplay(metadata);
            content.appendChild(metadataDiv);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.scrollToBottom();
    }

    formatMessage(text) {
        // Convert URLs to clickable links
        text = text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" style="color: inherit; text-decoration: underline;">$1</a>');
        
        // Convert line breaks to <br> tags
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }

    createMetadataDisplay(metadata) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'message-metadata';
        metadataDiv.style.cssText = 'margin-top: 10px; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 8px; font-size: 12px;';
        
        const metadataText = [];
        if (metadata.intent) {
            metadataText.push(`Intent: ${metadata.intent}`);
        }
        if (metadata.confidence !== undefined) {
            metadataText.push(`Confidence: ${(metadata.confidence * 100).toFixed(1)}%`);
        }
        if (metadata.entities && Object.keys(metadata.entities).length > 0) {
            metadataText.push(`Entities: ${Object.keys(metadata.entities).join(', ')}`);
        }
        if (metadata.response_time) {
            metadataText.push(`Response time: ${metadata.response_time.toFixed(2)}s`);
        }
        
        metadataDiv.textContent = metadataText.join(' • ');
        return metadataDiv;
    }

        showTicketCreated(ticketId, isOffTopic = false, hasContent = true) {
        const ticketDiv = document.createElement('div');
        ticketDiv.className = 'message bot-message';
        ticketDiv.style.animation = 'none';
        
        if (isOffTopic) {
            ticketDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="message-content" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white;">
                    <div class="message-text">
                        ⚠️ <strong>Off-Topic Query Detected!</strong><br>
                        🎫 Support Ticket Created: <code>${ticketId}</code><br>
                        Our supervisor will contact you within 2 hours to assist with your request.
                    </div>
                    <div class="message-time" style="color: rgba(255,255,255,0.8);">Just now</div>
                </div>
            `;
        } else if (!hasContent) {
            ticketDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-search"></i>
                </div>
                <div class="message-content" style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); color: white;">
                    <div class="message-text">
                        🔍 <strong>No Knowledge Base Content Found!</strong><br>
                        🎫 Support Ticket Created: <code>${ticketId}</code><br>
                        Our team will research your question and get back to you within 2 hours.
                    </div>
                    <div class="message-time" style="color: rgba(255,255,255,0.8);">Just now</div>
                </div>
            `;
        } else {
            ticketDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-ticket-alt"></i>
                </div>
                <div class="message-content" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white;">
                    <div class="message-text">
                        🎫 <strong>Support Ticket Created!</strong><br>
                        Ticket ID: <code>${ticketId}</code><br>
                        Our team will review your request and get back to you soon.
                    </div>
                    <div class="message-time" style="color: rgba(255,255,255,0.8);">Just now</div>
                </div>
            `;
        }
        
        this.chatMessages.appendChild(ticketDiv);
        this.scrollToBottom();
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    setInputState(enabled) {
        this.messageInput.disabled = !enabled;
        this.sendBtn.disabled = !enabled;
        
        if (enabled) {
            this.messageInput.focus();
        }
    }

    getCurrentTime() {
        const now = new Date();
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            // Keep only the welcome message
            const welcomeMessage = this.chatMessages.querySelector('.message.bot-message');
            this.chatMessages.innerHTML = '';
            this.chatMessages.appendChild(welcomeMessage);
            
            // Clear backend conversation state
            if (this.conversationId) {
                fetch('/api/clear-chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        conversation_id: this.conversationId
                    })
                }).then(response => response.json())
                .then(data => {
                    console.log('Chat cleared:', data);
                }).catch(error => {
                    console.error('Error clearing chat:', error);
                });
            }
            
            // Generate new conversation ID
            this.conversationId = this.generateConversationId();
            
            // Reset input
            this.messageInput.value = '';
            this.updateCharCount();
        }
    }
    
    resetConversation() {
        if (confirm('Reset conversation and start fresh?')) {
            // Clear backend conversation state
            if (this.conversationId) {
                fetch('/api/clear-chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        conversation_id: this.conversationId
                    })
                }).catch(error => {
                    console.error('Error clearing chat:', error);
                });
            }
            
            // Generate new conversation ID
            this.conversationId = this.generateConversationId();
            
            // Clear all messages except welcome
            const welcomeMessage = this.chatMessages.querySelector('.message.bot-message');
            this.chatMessages.innerHTML = '';
            this.chatMessages.appendChild(welcomeMessage);
            
            // Reset input
            this.messageInput.value = '';
            this.updateCharCount();
        }
    }

    async showAnalytics() {
        try {
            this.showLoading(true);
            
            const response = await fetch('/api/analytics');
            if (!response.ok) throw new Error('Failed to fetch analytics');
            
            const analytics = await response.json();
            this.populateAnalytics(analytics);
            
            document.getElementById('analyticsModal').style.display = 'block';
            
        } catch (error) {
            console.error('Error fetching analytics:', error);
            alert('Failed to load analytics. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }

    populateAnalytics(analytics) {
        document.getElementById('totalQueries').textContent = analytics.total_queries || 0;
        document.getElementById('totalTickets').textContent = analytics.total_tickets || 0;
        document.getElementById('avgResponseTime').textContent = `${(analytics.avg_response_time || 0).toFixed(2)}s`;
        document.getElementById('entityExtractionRate').textContent = `${((analytics.entity_extraction_rate || 0) * 100).toFixed(1)}%`;
        
        // Populate intent distribution
        const intentChart = document.getElementById('intentChart');
        if (analytics.intent_distribution && Object.keys(analytics.intent_distribution).length > 0) {
            let intentHtml = '<div style="text-align: left; width: 100%;">';
            for (const [intent, count] of Object.entries(analytics.intent_distribution)) {
                intentHtml += `
                    <div style="display: flex; justify-content: space-between; margin: 8px 0; padding: 8px; background: white; border-radius: 8px;">
                        <span style="text-transform: capitalize; font-weight: 500;">${intent}</span>
                        <span style="color: #667eea; font-weight: bold;">${count}</span>
                    </div>
                `;
            }
            intentHtml += '</div>';
            intentChart.innerHTML = intentHtml;
        } else {
            intentChart.innerHTML = '<span style="color: #999;">No data available yet</span>';
        }
    }

    closeModal() {
        document.getElementById('analyticsModal').style.display = 'none';
    }

    showLoading(show) {
        this.loadingOverlay.style.display = show ? 'block' : 'none';
    }
    
    isEcommerceRelated(query) {
        const ecommerceKeywords = [
            'order', 'purchase', 'buy', 'shopping', 'product', 'item', 'delivery', 'shipping',
            'payment', 'billing', 'refund', 'return', 'exchange', 'account', 'login', 'password',
            'tracking', 'package', 'invoice', 'charge', 'cost', 'price', 'website', 'app',
            'customer', 'support', 'help', 'issue', 'problem', 'complaint', 'service',
            'cart', 'checkout', 'receipt', 'confirmation', 'status', 'update', 'cancel',
            'modify', 'address', 'shipping', 'tax', 'discount', 'coupon', 'promo', 'sale',
            'inventory', 'stock', 'availability', 'warranty', 'guarantee', 'policy'
        ];
        
        const queryLower = query.toLowerCase();
        return ecommerceKeywords.some(keyword => queryLower.includes(keyword));
    }
}

// Initialize chatbot when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new ChatbotUI();
});

// Global functions for HTML onclick handlers
function sendMessage() {
    if (window.chatbot) {
        window.chatbot.sendMessage();
    }
}

function clearChat() {
    if (window.chatbot) {
        window.chatbot.clearChat();
    }
}

function showAnalytics() {
    if (window.chatbot) {
        window.chatbot.showAnalytics();
    }
}

function closeModal() {
    if (window.chatbot) {
        window.chatbot.closeModal();
    }
}

// Close modal when clicking outside
window.addEventListener('click', (event) => {
    const modal = document.getElementById('analyticsModal');
    if (event.target === modal) {
        closeModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        closeModal();
    }
});
