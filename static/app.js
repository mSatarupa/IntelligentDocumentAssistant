document.addEventListener('DOMContentLoaded', () => {
    // API Configuration
    const API_BASE_URL = "";

    // --- Elements ---
    const chatContainer = document.getElementById('chatContainer');
    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const sendBtn = document.getElementById('sendBtn');
    
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const inlineUploadBtn = document.getElementById('inlineUploadBtn');
    
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const toastContainer = document.getElementById('toastContainer');

    let isProcessing = false;

    // --- Helpers ---
    function setSystemBusy(busy, text = "Processing...") {
        isProcessing = busy;
        sendBtn.disabled = busy;
        queryInput.disabled = busy;
        if (busy) {
            statusIndicator.classList.add('busy');
            statusText.textContent = text;
        } else {
            statusIndicator.classList.remove('busy');
            statusText.textContent = 'System Ready';
        }
    }

    function showToast(type, title, message) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = 'bx-info-circle';
        if (type === 'success') icon = 'bx-check-circle';
        if (type === 'error') icon = 'bx-error-circle';

        toast.innerHTML = `
            <i class='bx ${icon} bx-sm'></i>
            <div class="toast-content">
                <h4>${title}</h4>
                <p>${message}</p>
            </div>
        `;

        toastContainer.appendChild(toast);

        // Remove after 4 seconds
        setTimeout(() => {
            toast.style.animation = 'fadeOut 0.3s ease forwards';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    function addMessage(type, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}-msg`;
        
        const avatar = type === 'user' ? "<i class='bx bx-user'></i>" : "<i class='bx bx-bot'></i>";
        
        msgDiv.innerHTML = `
            <div class="avatar">${avatar}</div>
            <div class="msg-content">${content}</div>
        `;
        
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msgDiv;
    }

    function addTypingIndicator() {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message system-msg typing-msg`;
        msgDiv.innerHTML = `
            <div class="avatar"><i class='bx bx-bot'></i></div>
            <div class="msg-content">
                <div class="typing">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        `;
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msgDiv;
    }

    // --- API Interactions ---

    // 1. Upload Document
    async function handleFileUpload(file) {
        if (!file) return;

        setSystemBusy(true, "Uploading & Indexing...");
        showToast('info', 'Upload Started', `Processing ${file.name}...`);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(`${API_BASE_URL}/api/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                showToast('success', 'Upload Complete', data.message);
                addMessage('system', `<p>✅ <b>${file.name}</b> successfully processed and added to the knowledge base.</p>`);
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            showToast('error', 'Upload Failed', error.message);
            addMessage('system', `<p style="color: var(--error)">❌ Failed to process document: ${error.message}</p>`);
        } finally {
            setSystemBusy(false);
            fileInput.value = ''; // reset
        }
    }

    // Bind Upload Buttons
    const triggerUpload = () => fileInput.click();
    uploadBtn.addEventListener('click', triggerUpload);
    inlineUploadBtn.addEventListener('click', triggerUpload);

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // 2. Submit Query
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = queryInput.value.trim();
        if (!text || isProcessing) return;

        // Add user message
        addMessage('user', `<p>${text}</p>`);
        queryInput.value = '';
        
        setSystemBusy(true, "Thinking...");
        const typingIndicator = addTypingIndicator();

        try {
            // Step 1: Submit Query
            const submitRes = await fetch(`${API_BASE_URL}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text })
            });

            const submitData = await submitRes.json();
            
            if (!submitRes.ok) {
                throw new Error(submitData.detail || 'Failed to submit query');
            }

            const queryId = submitData.query_id;

            // Step 2: In a real async system we might poll, 
            // but the current backend blocks until ready, so we can fetch immediately
            const resultRes = await fetch(`${API_BASE_URL}/api/output?query_id=${queryId}`);
            const resultData = await resultRes.json();

            if (!resultRes.ok) {
                throw new Error(resultData.detail || 'Failed to retrieve response');
            }

            // Remove typing indicator & show result
            typingIndicator.remove();
            
            // Format response (basic markdown-like to HTML if needed, here just raw text)
            // Replace newlines with <br> for basic formatting
            const formattedResponse = resultData.response.replace(/\n/g, '<br>');
            addMessage('system', `<p>${formattedResponse}</p>`);

        } catch (error) {
            console.error('Query error:', error);
            typingIndicator.remove();
            addMessage('system', `<p style="color: var(--error)">⚠️ Error: ${error.message}</p>`);
            showToast('error', 'Query Failed', error.message);
        } finally {
            setSystemBusy(false);
            // Re-focus input
            setTimeout(() => queryInput.focus(), 10);
        }
    });
});
