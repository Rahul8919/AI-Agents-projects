# AI Gmail Meeting Scheduler Agent (n8n + LLM)

An **Agentic AI workflow** built using **n8n, LLM reasoning, and Google APIs** that automatically reads incoming Gmail messages and schedules meetings in Google Calendar.

This project demonstrates how **AI agents can automate real-world tasks using workflow orchestration and external tools**.

---

# Project Overview

The system acts as an **AI assistant that monitors incoming emails** and interprets meeting-related requests.

When a user sends an email requesting a meeting, the AI agent:

1. Reads the email content.
2. Understands the meeting intent using an LLM.
3. Checks Google Calendar for existing events.
4. Creates a new calendar event if necessary.
5. Confirms the scheduling action.

This project demonstrates **Agentic AI automation using workflow orchestration**.

---

# System Architecture

Workflow Architecture:

Gmail Trigger
↓
AI Agent (LLM reasoning)
↓
Google Calendar Tools

1️⃣ Gmail Trigger → detects new emails
2️⃣ AI Agent → interprets user intent
3️⃣ Google Calendar Tool → retrieves or creates events

---

# Workflow Components

## Gmail Trigger

The workflow begins by monitoring the Gmail inbox.

Features:

• Polls inbox every minute
• Detects new incoming emails
• Extracts sender and message content

Example data captured:

Sender email
Email subject
Email body snippet

---

## AI Agent

The **AI Agent node** uses a language model to interpret user requests.

Prompt context:

The agent acts as a **calendar assistant** capable of:

• Checking calendar events
• Creating meetings
• Responding to scheduling requests

The agent receives email data such as:

Sender address
Email body content
Timestamp

Using this information, it determines the required action.

---

## LLM Model

The workflow uses an OpenRouter model:

Model:

```
nvidia/nemotron-nano-9b-v2
```

The model is responsible for:

• Natural language understanding
• Extracting meeting details
• Deciding which tool to use

---

## Google Calendar Tools

Two AI tools are connected to the agent.

### 1️⃣ Get Calendar Events

Allows the agent to:

• Check existing events
• Detect scheduling conflicts

### 2️⃣ Create Calendar Event

Allows the agent to:

• Create new meetings
• Set start and end times
• Add meeting details

---

# Example Workflow

Example email:

```
Hi Rahul,
Can we schedule a meeting tomorrow at 3 PM?
```

Workflow behavior:

1️⃣ Gmail Trigger detects the email
2️⃣ AI Agent reads the message
3️⃣ Agent extracts meeting request
4️⃣ Google Calendar tool checks events
5️⃣ Agent creates a new meeting

Result:

A calendar event is automatically scheduled.

---

# Technologies Used

| Technology          | Purpose                    |
| ------------------- | -------------------------- |
| n8n                 | Workflow orchestration     |
| OpenRouter LLM      | Natural language reasoning |
| Gmail API           | Email monitoring           |
| Google Calendar API | Event management           |
| AI Agent Node       | Tool-based reasoning       |

---

# Key Concepts Demonstrated

This project demonstrates several **Agentic AI principles**:

• AI workflow automation
• LLM-based decision making
• Tool calling by agents
• API orchestration
• Event-driven automation

---

# Installation / Setup

1️⃣ Install n8n

```
npm install -g n8n
```

2️⃣ Start n8n

```
n8n start
```

3️⃣ Import the workflow JSON

```
workflow.json
```

4️⃣ Configure credentials

Required APIs:

• Gmail OAuth2
• Google Calendar OAuth2
• OpenRouter API

---

# Project Structure

```
gmail-meeting-scheduler-agent
│
├── workflow.json
├── README.md
└── screenshots
    └── workflow.png
```

---

# Portfolio Value

This project demonstrates:

• Agentic AI workflow design
• Automation with LLM reasoning
• Integration with real-world APIs
• Intelligent decision-making agents

This type of system is commonly used in:

• AI productivity assistants
• workflow automation platforms
• enterprise AI agents

---

# Author

Rahul Nerella
GenAI /Agentic AI Developer

LinkedIn
https://linkedin.com/in/nerella-rahul-goud-a15934205
