import os
import autogen
import gradio as gr
from openai import OpenAI
from dotenv  import load_dotenv
import random
from google import genai

load_dotenv()


OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
print("setup complete all related  libraries and API keys are loaded successfully")


# Initialize OpenAI and Google GenAI clients

config_list_openai = [
    {
        "model": "gpt-4o-mini",
        "api_key": OPEN_API_KEY
    }
]
llm_config_openai={
    "config_list": config_list_openai,
    "temperature": 0.7,
    "timeout": 120    
}

config_list_gemini = [
    {
        "model": "gemini-2.0-flash",
        "api_key": GOOGLE_GEMINI_API_KEY,
        "api_type": "google"
    }
]
llm_config_gemini={
    "config_list": config_list_gemini,
    "temperature": 0.7,
    "timeout": 120    
}
config_list_claude = [
    {
        "model": "gemini-2.0-flash",
        "api_key": GOOGLE_GEMINI_API_KEY,
        "api_type": "google"
    }
]
llm_config_claude={
    "config_list": config_list_claude,
    "temperature": 0.7,
    "timeout": 120    
}

cmo_prompt = """You are the Chief Marketing Officer (CMO) of a new shoe brand (sustainable).
Provide high-level strategy, define target audiences, and guide the Marketer. 
Focus on the big picture. Be concise."""

brand_marketer_prompt = """You are the Brand Marketer for the shoe brand. 
Brainstorm creative, specific campaign ideas (digital, content, experiences). 
Focus on tactics and details. Suggest KPIs for your ideas."""

social_media_prompt = """You are the Social Media Strategist for a new sustainable shoe brand.
Develop creative social media posts, suggest platforms, posting times, and engagement strategies.
Work with the Brand Marketer and CMO."""

# creating the Chief Marketing Officer (CMO) Agent
CMO_AGENT_GEMINI = autogen.ConversableAgent(
    name="CMO_Agent_gemini",
    system_message=cmo_prompt,
    llm_config=llm_config_gemini,
    human_input_mode="Never"
)
print("CMO agent initialized successfully")

# creating the Brand Marketer Agent
BRAND_MARKERTING_AGENT_OPENAI = autogen.ConversableAgent(
    name="Brand_Marketing_openAI_Agent",
    system_message=brand_marketer_prompt,
    llm_config=llm_config_openai,
    human_input_mode="Never"
)
print("Brand Marketing agent initialized successfully")

# creating the Social Media Strategist Agent
social_media_agent_claude = autogen.ConversableAgent(
    name="Social_Media_Claude_Agent",
    system_message=social_media_prompt,
    llm_config=llm_config_claude,
    human_input_mode="Never"
)
print("Social Media Strategist agent initialized successfully")

initial_task_message = """
Context: We're launching a new sustainable shoe line and need campaign ideas
Instruction: Brainstorm a campaign concept with specific elements
Input: Our sustainable, futuristic shoe brand needs marketing direction
Output: A concise campaign concept with the following structure:
Brand Marketer, let's brainstorm initial campaign ideas for our new sustainable shoe line.
Give me a distinct campaign concept. Outline: core idea, target audience, primary channels, and 1-2 KPIs. Keep it concise. Try to arrive at a final answer.
"""
#start of one agent to another agent conversation for testing
# chat_result_openai_only = CMO_AGENT_GEMINI.initiate_chat(
#     recipient = BRAND_MARKERTING_AGENT_OPENAI,
#     message = initial_task_message,
#     max_turns = 4
# )
# print("maximum turns has been completed")
# print("-------------------------------------------------------------")
# print("--- Conversation Ended (Multi-Model) ---")
#end of one agent to another agent conversation for testing

user_proxy_agent = autogen.UserProxyAgent(
    name = "Human_User_Proxy",
    human_input_mode = "ALWAYS",  # Prompt user for input until 'exit'
    max_consecutive_auto_reply = 1,
    is_termination_msg = lambda x: x.get("content", "").rstrip().lower() in ["exit", "quit", "terminate"],
    code_execution_config = False,
    system_message = "You are the human user interacting with a multi-model AI team (Gemini CMO, OpenAI Marketer). Guide the brainstorm. Type 'exit' to end."
)
print(f"Agent '{user_proxy_agent.name}' created for HIL with multi-model team.")
#start to resert agents before starting conversation it is helpful
CMO_AGENT_GEMINI.reset()
BRAND_MARKERTING_AGENT_OPENAI.reset()
social_media_agent_claude.reset()
user_proxy_agent.reset()
#end to resert agents before starting conversation it is helpful


# This sets up a collaborative chat environment where multiple agents can interact
from autogen import GroupChat, GroupChatManager

groupchat = GroupChat(
    agents = [user_proxy_agent, CMO_AGENT_GEMINI, BRAND_MARKERTING_AGENT_OPENAI,social_media_agent_claude],  # List of agents participating in the group chat
    messages = [],  # Initialize with empty message history
    max_round = 20  # Optional: Limits how many conversation rounds can occur before terminating
)

# The GroupChatManager orchestrates the conversation flow between agents
# It determines which agent should speak next and handles the overall conversation logic

group_manager = GroupChatManager(
    groupchat = groupchat,
    llm_config = llm_config_gemini   # Uses gemini's LLM to manage the conversation
)

try:
    group_chat_result = group_manager.initiate_chat(
        recipient = user_proxy_agent,    # Start by talking to the Gemini CMO
        message = """Hello team!"""
    )
except Exception as e:
    print("Agent system failed:", e)
    
def print_chat_history(group_chat_result):
    """
    Any chat result object has a chat_history attribute that contains the conversation history.
    This function prints the conversation history in a readable format.
    """

    for i in group_chat_result.chat_history:
        print(i['name'])
        print("_"*100)
        print(i['content'])
        print("_"*100)

