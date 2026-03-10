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

MCP_BASE="http://localhost:7860/gradio_api/mcp/sse" #here we have to keep mcp server  URL 

from agents.mcp import MCPServerSse

MCP_Tool=MCPServerSse({
    "name":"AI Tutor",
    "url":MCP_BASE,
    "timeout":30,
    "client_session_timeout":60  
})

# This code is designed to fetch and display a schema (manifest) from an MCP server
# Typically used to describe available tools on that server.

# Let's use httpx for modern async-friendly requests (though requests works fine too)
client = httpx.Client()  # Create an HTTP client instance
def fetch_schema(server_url):
    """Fetches and parses the MCP schema from a server."""

    schema_url = server_url.replace("/sse", "/schema")
    print(f"Fetching schema from: {schema_url}")

    response = client.get(schema_url, timeout=10)  # Add a timeout
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    schema_data = response.json()
    print("Schema fetched successfully!")
    return schema_data


# Fetch the manifest from the AI Tutor server

# Note that the schema is used by AI agents to know how to call the AI Tutor tool
# programmatically, ensuring the inputs match the expected format.

print("--- Fetching AI Tutor Schema ---")
tutor_schema = fetch_schema(MCP_BASE)

if tutor_schema:
    print("\nAI Tutor Schema Contents:")
    # Pretty print the JSON manifest
    print(json.dumps(tutor_schema, indent=2))

print("\n" + "=" * 50 + "\n")  # Separator



# Let's Build the AI agent
from agents import Agent, Runner

agent = Agent(
    name = "Smart Assistant",
    instructions = """
    Context
    -------

    You are an AI assistant with access to an MCP server exposing **four streaming tools**:

    1. **explain_concept**
       Arguments: { "question": <str>, "level": <int 1-5> }
       • Streams an explanation of any concept at the requested depth.

    2. **summarize_text**
       Arguments: { "text": <str>, "compression_ratio": <float 0.1-0.8> }
       • Streams a concise summary ~compression_ratio × original length.

    3. **generate_flashcards**
       Arguments: { "topic": <str>, "num_cards": <int 1-20> }
       • Streams JSON-lines flashcards: one card per line `{ "q":…, "a":… }`.

    4. **quiz_me**
       Arguments: { "topic": <str>, "level": <int 1-5>, "num_questions": <int 1-15> }
       • Streams an MC-question quiz, then an ANSWER KEY section.
    4. quiz_me
         Arguments: { "topic": <str>, "level": <int 1-5>, "num_questions": <int 1-15> }
        • Streams an MC-question quiz, then an ANSWER KEY section.
    Objective
    Help users learn by:
        • Explaining concepts at the depth they request.
        • Summarising long passages.
        • Generating flashcards for self-study.
        • Quizzing them interactively.
    How to respond
        • For each user request, decide which tool (if any) fulfils it best.
        • Call the tool via MCP by returning only the JSON with "tool" and "arguments" (no extra text).
        • If a follow-up conversation is needed (e.g., clarification), ask the user first.
        • If no tool fits, answer directly in plain language.
    Examples
        User: “Explain quantum tunnelling like I’m 10.”
        → Call "explain_concept" with { "question": "quantum tunnelling", "level": 2 }
        
        User: “Summarise this article to 20 %.” + <article text>
        → Call "summarise" with { "text": <article text>, "ratio": 0.2 }
        
    Chat capability
        ---------------
        After each tool call completes (streaming back to the user), remain in the chat loop ready for the next user turn.
        """,
        model = "gpt-4o-mini",
        mcp_servers = [MCP_Tool],
    )

# This code snippet is implementing a conversational loop with an AI agent that uses MCP tools.

# Opens a connection with an MCP tool.
# This lets our AI agent interact with an external tool (e.g., image generator, calculator, etc.) over a standard protocol using S
    
async def main():
    await MCP_Tool.connect() # open SSE channels
    
    result = None
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            break
         # If there was a previous interaction (result is not None), it appends the new user message to the past messages (maintaining
        if result is not None:
            new_input = result.to_input_list() + [{"role": "user", "content": user_input}]
        else:
            new_input = [{"role": "user", "content": user_input}]
        result = await Runner.run(agent, new_input)

        print("\nAssistant:")
        print(result.final_output)

asyncio.run(main())

#below code is for knowing what are the tools that has been used by th agent
for i in result.to_input_list():
    for key in i.keys():
        if key == 'arguments':
            print("Tool: ", i['name'])
            print("Arguments: ", i['arguments'])