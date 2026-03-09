# AI Research Multi-Agent System with Guardrails & handoff

## Overview

This project implements a **multi-agent AI research system** built using the **OpenAI Agents SDK**.
The system automatically performs structured research on a given topic by coordinating multiple specialized AI agents.

Each agent has a specific responsibility such as:

* Research planning
* Web information retrieval
* Financial analysis
* Sentiment analysis
* Report generation

The agents collaborate through **agent handoffs**, producing a structured investment-style research report.

The system also includes **guardrails** to prevent the agent from responding to political queries.

---

# Problem Statement

Traditional AI assistants respond directly to user prompts, which often leads to:

* shallow research
* hallucinated information
* lack of structured reasoning

Complex research tasks require **multiple steps and specialized reasoning stages**.

This project solves that problem by using a **multi-agent architecture**, where each agent focuses on a specific part of the research process.

---

# System Architecture

The workflow follows a structured **multi-agent pipeline**:

User Query
↓
Planner Agent
↓
Search Agent
↓
Fundamentals Analyst
↓
Sentiment Analyst
↓
Writer Agent
↓
Final Research Report

The system uses **agent handoffs** to transfer information between agents.

---

# Agents in the System

## 1. Planner Agent

The planner agent analyzes the user's request and generates a **structured research plan**.

Responsibilities:

* Break user request into multiple research queries
* Identify relevant topics to investigate
* Generate a list of search queries

Output:

SearchPlan containing multiple search tasks.

---

## 2. Search Agent

The search agent retrieves **real-time information from the internet** using the Tavily API.

Responsibilities:

* Perform web searches
* Collect relevant information
* Summarize key findings

Tools Used:

Tavily Web Search API

---

## 3. Fundamentals Analyst Agent

This agent analyzes the **financial fundamentals** of companies.

Responsibilities:

* Evaluate revenue
* Analyze growth trends
* Assess profitability

The agent uses web search results and contextual information to generate insights.

---

## 4. Sentiment Analysis Agent

This agent evaluates **market sentiment and public perception**.

Responsibilities:

* Analyze news sentiment
* Identify positive or negative market signals
* Highlight major trends affecting the company

---

## 5. Writer Agent

The writer agent compiles all research findings into a **structured investment report**.

Responsibilities:

* Combine research outputs from other agents
* Write a detailed markdown report
* Provide executive summary
* Suggest follow-up research questions

Final Output Includes:

* Short summary
* Detailed markdown report
* Follow-up research questions

---

# Guardrails

The system includes an **input guardrail agent** that detects political topics.

If a user asks about:

* politics
* elections
* government
* political leaders

the guardrail prevents the system from continuing the workflow.

This ensures safe and controlled agent behavior.

---

# Agent Handoff System

The system uses **agent handoffs** to move tasks between agents.

Example flow:

Planner Agent → Writer Agent

The planner generates a research plan and then transfers control to the writer agent using a handoff mechanism.

This enables modular agent workflows.

---

# Tools Used

## Tavily Search Tool

The Tavily API is used for retrieving real-time web information.

The tool allows agents to:

* search the web
* gather relevant information
* summarize results

---

# Technologies Used

Python
OpenAI Agents SDK
OpenAI GPT-4o-mini model
Tavily Search API
Async Programming
Pydantic Data Models
SQLite Session Storage

---

# Project Structure

```
AI_Research_Agent
│
├── research_agent.py
├── requirements.txt
├── .env
└── README.md
```

Main Components:

research_agent.py → Contains all agent definitions and orchestration logic.

---

# Key Features

Multi-agent architecture
Agent collaboration via handoffs
Guardrails for safe AI behavior
Tool-enabled agents
Real-time web research
Structured output using Pydantic models
Async agent execution

---

# Example Query

User Input:

What are the most promising battery companies in 2026?

Output:

The system generates:

* executive summary
* structured research report
* follow-up research questions

---

# Future Improvements

Add long-term memory for agents
Integrate financial APIs for live data
Add more specialized agents
Deploy as a web application
Add visualization dashboards

---

# Learning Outcomes

This project demonstrates advanced concepts in **Agentic AI Engineering**, including:

* Multi-agent orchestration
* Tool-enabled LLM agents
* Guardrail implementation
* Agent handoff mechanisms
* Structured LLM outputs
* AI workflow pipelines

---

# Author

Rahul
Agentic AI / Generative AI Developer  
Specializing in AI Agents, Agentic AI Systems, and LLM-powered applications.
