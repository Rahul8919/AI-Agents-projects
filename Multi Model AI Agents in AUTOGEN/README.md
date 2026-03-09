# Multi-Model AI Agents using AutoGen

## Overview

This project demonstrates a **multi-model AI agent collaboration system** built using the **Microsoft AutoGen framework**.

Multiple AI agents powered by **different LLM models** collaborate to brainstorm marketing strategies for a sustainable shoe brand.

Each agent has a specific role and expertise, enabling a structured discussion similar to a real business team.

The system also includes a **Human-in-the-Loop (HIL)** agent to guide the conversation and refine ideas.

---

# System Architecture

Human User
↓
Group Chat Manager
↓
Multi-Model Agent Team

Agents:

CMO Agent (Gemini)
Brand Marketing Agent (OpenAI)
Social Media Strategist Agent
Human Proxy Agent

↓

Collaborative Strategy Discussion

---

# Agents in the System

## 1. CMO Agent (Gemini)

Role: Chief Marketing Officer

Responsibilities:

* Define marketing strategy
* Identify target audiences
* Provide high-level direction for campaigns

Model Used:

Gemini 2.0 Flash

---

## 2. Brand Marketing Agent (OpenAI)

Role: Marketing strategist

Responsibilities:

* Generate campaign concepts
* Suggest marketing channels
* Define key performance indicators (KPIs)

Model Used:

GPT-4o-mini

---

## 3. Social Media Strategist Agent

Role: Social media expert

Responsibilities:

* Design social media campaigns
* Suggest platforms and posting schedules
* Develop engagement strategies

---

## 4. User Proxy Agent (Human-in-the-Loop)

Role: Human decision maker

Responsibilities:

* Guide the discussion
* Provide feedback to agents
* Control conversation flow

---

# Key Features

Multi-model AI agent collaboration
Human-in-the-Loop interaction
Group chat orchestration
Role-based AI agents
Creative marketing strategy generation
Conversation management using AutoGen

---

# Technologies Used

Python
Microsoft AutoGen
OpenAI API (GPT-4o-mini)
Google Gemini API
Gradio (for UI integration)
dotenv for environment configuration

---

# Project Structure

```
Multi_Model_AI_Agents_AutoGen
│
├── multi_model_agents_autogen.py
├── requirements.txt
└── README.md
```

---

# Example Scenario

The system simulates a **marketing strategy meeting** for launching a new sustainable shoe brand.

Example prompt:

```
Hello team! We are launching a new sustainable shoe brand.
Let's brainstorm a creative marketing campaign.
```

The agents collaborate to produce:

* marketing campaign ideas
* target audiences
* marketing channels
* engagement strategies
* key performance indicators (KPIs)

---

# Learning Outcomes

This project demonstrates several **Agentic AI engineering concepts**, including:

Multi-agent collaboration
Multi-model LLM orchestration
Human-in-the-Loop workflows
Group chat agent management
Role-based agent design

---

# Future Improvements

Add more specialized agents (SEO, Data Analyst)
Integrate marketing analytics APIs
Add visualization dashboards
Deploy as a web application

---

# Author

Rahul
Agentic AI / Generative AI Developer

Specializing in AI Agents, Agentic AI Systems, and LLM-powered applications.
