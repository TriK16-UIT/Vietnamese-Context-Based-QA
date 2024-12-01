import React, { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
    const [input, setInput] = useState(""); // Single input for both context and question
    const [messages, setMessages] = useState([]);
    const [contextMode, setContextMode] = useState(true); // Toggle between context and question mode
    const [expanded, setExpanded] = useState(false); // State to toggle container expansion
    const chatBoxRef = useRef(null); // Ref for scrolling the chatbox

    useEffect(() => {
        if (chatBoxRef.current) {
            chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
        }
    }, [messages]);

    const handleReset = async () => {
        try {
            const response = await fetch("/ask", {
                method: "POST",
                body: JSON.stringify({ question: "reset" }),
                headers: {
                    "Content-Type": "application/json",
                },
            });
            const data = await response.json();
            if (data.message === "Context has been reset.") {
                setMessages((prev) => [
                    ...prev,
                    { sender: "System", text: "Context reset. Please enter a new context." },
                ]);
                setContextMode(true); // Switch to context mode
                setInput(""); // Clear input field
            }
        } catch (error) {
            console.error("Error resetting context:", error);
        }
    };

    const handleSetContext = async () => {
        try {
            await fetch("/set_context", {
                method: "POST",
                body: JSON.stringify({ context: input }),
                headers: {
                    "Content-Type": "application/json",
                },
            });
            setMessages((prev) => [
                ...prev,
                { sender: "User", text: `Context: ${input}` }, // Show the context as a message
                { sender: "System", text: "Context set successfully!" },
            ]);
            setContextMode(false); // Switch to question mode
            setInput(""); // Clear input field
            setExpanded(true); // Expand the container
        } catch (error) {
            console.error("Error setting context:", error);
        }
    };

    const handleAskQuestion = async () => {
        if (input.trim().toLowerCase() === ".reset") {
            await handleReset();
            return;
        }
        try {
            const response = await fetch("/ask", {
                method: "POST",
                body: JSON.stringify({ question: input }),
                headers: {
                    "Content-Type": "application/json",
                },
            });
            const data = await response.json();
            setMessages((prev) => [
                ...prev,
                { sender: "User", text: input },
                { sender: "Bot", text: data.answer },
            ]);
            setInput(""); // Clear input field
            setExpanded(true); // Ensure container stays expanded
        } catch (error) {
            console.error("Error asking question:", error);
        }
    };

    const handleSubmit = () => {
        if (contextMode) {
            handleSetContext();
        } else {
            handleAskQuestion();
        }
    };

    return (
        <div className={`chat-container ${expanded ? "expanded" : ""}`}>
            <div className="header">Question Answering Chatbot</div>

            <div className="chat-box" ref={chatBoxRef}>
                {messages.map((msg, index) => (
                    <div
                        key={index}
                        className={`message-container ${
                            msg.sender.toLowerCase() === "user" ? "user" : "bot"
                        }`}
                    >
                        <div className="sender">{msg.sender}:</div>
                        <div className="message">{msg.text}</div>
                    </div>
                ))}
            </div>

            <div className="question-input">
                <input
                    type="text"
                    placeholder={
                        contextMode
                            ? "Enter context here..."
                            : "Ask a question or type .reset to clear context..."
                    }
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                />
                <button onClick={handleSubmit}>
                  <i className="fa fa-paper-plane" aria-hidden="true"></i>
                </button>
            </div>
        </div>
    );
}

export default App;
