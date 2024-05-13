document.addEventListener("DOMContentLoaded", function() {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    sendBtn.addEventListener("click", function() {
        sendMessage();
    });

    userInput.addEventListener("keypress", function(e) {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    function sendMessage() {
        const message = userInput.value.trim();
        var dropdown = document.getElementById("dropdown-option");

        var selectedOption = dropdown.options[dropdown.selectedIndex].text;

        if (message !== "") {
            displayMessage(message, "user");

            // Send message to Flask backend
            fetch('/send-message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({message: message, model: selectedOption})
            })
            .then(response => response.json())
            .then(data => {
                const botResponse = data.message;
                displayMessage(botResponse, "bot");
            })
            .catch(error => {
                console.error('Error:', error);
            });

            userInput.value = ""; // Clear input after sending
        }
    }

    function displayMessage(message, sender) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message");
        if (sender === "user") {
            messageElement.classList.add("user");
        } else {
            messageElement.classList.add("bot");
        }
        messageElement.innerText = message;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to latest message
    }
});
