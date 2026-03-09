import os
import requests  # For making HTTP requests
import httpx  # An alternative async-friendly HTTP client (good practice)
import json  # For handling JSON data (manifests, action responses)
from dotenv import load_dotenv
from openai import OpenAI  # or LiteLLM, groq, etc.

from PIL import Image
import asyncio, pathlib

load_dotenv()
OpenAI_API_KEY=os.getenv("OPENAI_API_KEY")
print(OpenAI_API_KEY[:10])

MCP_BASE="http://localhost:7860/gradio_api/mcp/" #here we have to keep mcp server  URL 

from agents.mcp import MCPServerSse

MCP_Tool=MCPServerSse({
    "name":"AI Tutor",
    "url":MCP_BASE,
    "timeeout":30,
    "client_session_timeout":60  
})

