async function botReply(userText) { 
    appendMessage('bot', 'Searching PDFs...');
    try {
         const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: userText}) 
        }); 
        const data = await response.json(); 
        // Remove "Searching PDFs..." and show actual reply 
        messagesDiv.lastChild.remove();
        appendMessage('bot', data.reply);
     } catch (error) {
        messagesDiv.lastChild.remove();
        appendMessage('bot', 'Sorry, there was an error.'); 
    } 
}