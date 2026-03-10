📚 AI Tutor MCP Toolkit – Agent + MCP Server

AI Tutor MCP Toolkit is a Generative AI learning assistant built using OpenAI Agents SDK, MCP (Model Context Protocol), and Gradio.

The system enables an AI agent to interact with streaming educational tools through an MCP server to help users learn concepts, summarize text, generate flashcards, and take quizzes.

This project demonstrates Agentic AI architecture, where an AI agent dynamically decides which tool to use to fulfill a user request.

System Architecture Diagram

User
 │
 ▼
AI Tutor Agent (OpenAI Agents SDK)
 │
 ▼
MCP Server (Gradio)
 │
 ▼
Available Tools
 ├─ explain_concept
 ├─ summarize_text
 ├─ generate_flashcards
 ├─ quiz_me
 └─ explain_concept_in_language
 

🚀 Features

The AI Tutor provides the following learning tools:

1️⃣ Explain Concept

Explains any concept at different difficulty levels.

Levels supported:

Level 1 → Like explaining to a 5-year-old

Level 2 → Like explaining to a 10-year-old

Level 3 → High school level

Level 4 → College level

Level 5 → Expert level

2️⃣ Summarize Text

Generates concise summaries of long content using a configurable compression ratio.

Example:

compression_ratio = 0.3

Reduces the text to approximately 30% of the original length.

3️⃣ Generate Flashcards

Creates study flashcards for any topic.

Example output:

{"q": "What is photosynthesis?", "a": "The process by which plants convert sunlight into energy."}

Supports:

1 – 20 flashcards
4️⃣ Quiz Generator

Generates multiple-choice quizzes for learning assessment.

Example:

1. What is Python?
A) Programming language
B) Snake
C) Database
D) Browser

Followed by:

ANSWER KEY
1 → A
5️⃣ Multilingual Concept Explanation

Explains concepts in different languages.

Example:

Explain Neural Networks in Telugu
🧠 Architecture

The system follows an Agentic AI architecture.

User
  │
  ▼
AI Agent (OpenAI Agents SDK)
  │
  ▼
MCP Server (Gradio)
  │
  ▼
Streaming Tools
  ├─ explain_concept
  ├─ summarize_text
  ├─ generate_flashcards
  ├─ quiz_me
  └─ explain_concept_in_language

The agent dynamically selects the appropriate tool using MCP tool schema discovery.

⚙️ Tech Stack
Technology	Purpose
Python	Core programming language
OpenAI API	LLM reasoning
OpenAI Agents SDK	Agent orchestration
MCP (Model Context Protocol)	Tool integration
Gradio	MCP server interface
httpx	HTTP communication
dotenv	API key management
📂 Project Structure
AI_Tutor_MCP_Toolkit
│
├── AI_Tutor_agent_client.py
├── MCP_server.py
├── .env
├── requirements.txt
└── README.md
Files

AI_Tutor_agent_client.py

Client that:

connects to MCP server

fetches tool schema

runs the AI agent

manages conversation loop

MCP_server.py

Implements MCP tools:

explain_concept
summarize_text
generate_flashcards
quiz_me
explain_concept_in_language
🔧 Installation
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/AI-Tutor-MCP-Toolkit.git
cd AI-Tutor-MCP-Toolkit
2️⃣ Create Virtual Environment
python -m venv .venv

Activate:

Windows

.venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Add OpenAI API Key

Create .env

OPENAI_API_KEY=your_api_key_here
▶️ Running the Project
Step 1 — Start MCP Server
python MCP_server.py

Server runs on:

http://localhost:7860
Step 2 — Run AI Agent Client

Open another terminal.

python AI_Tutor_agent_client.py
Example Interaction
User: Explain neural networks like I'm 10

Assistant: (calls explain_concept tool)
🧪 MCP Schema Discovery

The agent automatically fetches tool schema from:

http://localhost:7860/gradio_api/mcp/schema

This allows the agent to understand:

tool names

parameters

input formats

🎯 Learning Outcomes

This project demonstrates:

Agentic AI architecture

MCP tool integration

Streaming responses from LLM

Dynamic tool selection by AI agents

Building custom AI tools

AI-powered learning applications

📈 Future Improvements

Possible extensions:

Multi-agent tutor system

LangGraph integration

Knowledge base (RAG)

Student progress tracking

Web UI chat interface

Voice-enabled tutoring

👨‍💻 Author

Rahul Nerella

Generative AI Developer

LinkedIn:
linkedin.com/in/nerella-rahul-goud-a15934205

⭐ If you found this project useful

Please consider giving the repository a star ⭐
