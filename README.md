# Cronus: Voice-Integrated AI Assistant

Cronus is an end-to-end personal AI assistant inspired by the "Jarvis" concept. It bridges high-level LLM reasoning with local system execution and real-time voice interaction.

## 🚀 The Build
This project wasn't built from a tutorial; it was an exercise in integrating fragmented APIs into a functional tool. 

* **Brain:** Powered by Google's Gemini-Flash for low-latency, high-intelligence reasoning.
* **Voice Engine:** Uses Piper (ONNX) for local, fast Text-to-Speech and SpeechRecognition for STT.
* **Tool-Calling:** Implemented a custom JSON-based tool loop that allows the agent to:
    * Scrape live web data via DuckDuckGo.
    * Execute weather lookups via Open-Meteo.
    * Authenticate and send emails through Gmail SMTP.
* **Architecture:** Managed asynchronous sub-processes to handle audio I/O while maintaining the LLM's conversation state.

## 🛠️ Tech Stack
* **Language:** Python
* **AI/ML:** Google GenAI, LiveKit Agents (Real-time integration)
* **Tools:** SMTP/MIME, Geocoding APIs, Piper TTS

## 💡 Why I Built This
I wanted to see how far I could push a "vanilla" LLM by giving it actual agency over my local machine. Solving the latency between speech recognition and AI response was a significant grind, but getting the loop to run seamlessly was the goal.
