import os
import uuid
import getpass
from typing import TypedDict, Annotated, Sequence, List, Tuple, Optional, Any, Union, Literal, Tuple
import operator
from datetime import date
import uuid  # Added for Gradio state

# Langchain specific imports
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
#from langchain.tools import tool
from langchain_core.tools import tool

#importing flight tool and hotel tool from other files
from flights_tool import search_flights_tool
from hotels_tool import search_hotels_tool
from date_tool import get_current_date_tool


# LangGraph imports (Updated based on recent versions)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode  # Preferred way to handle tool execution

# Gradio
import gradio as gr

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
amadeus_api_key = os.environ["AMADEUS_CLIENT_ID"]
amadeus_api_secret = os.environ["AMADEUS_CLIENT_SECRET"]

print(f"Amadeus Key starts with: {amadeus_api_key[:5]}...")
print(f"Amadeus Secret starts with: {amadeus_api_secret[:5]}...")

print("API Keys loaded (partially hidden for security):")
print(f"OpenAI Key starts with: {openai_api_key[:5]}...")
print(f"Tavily Key starts with: {TAVILY_API_KEY[:5]}...")

# Let's define a simple workflow that includes a summarization function
# Let's define a state that includes two information: the original text and its summary
# State is how information persists and flows between nodes
# The state is like a container that stores and passes data between different parts of our workflow
# Each node receives and returns a state object, and the State can include messages, variables, memory, etc.

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],operator.add] #this picks all the orderwise conversation and new conversation will be append to list

# Inner function (call_model_with_tools(state)): This is the execution function.
# It knows how to use the current state (conversation history) and actually runs the model with the tools that were set up by the outer function.

def make_call_model_with_tools(tools: list):
    def call_model_with_tools(state: AgentState):
        print("DEBUG: Entering call_model_with_tools node")
        messages = state["messages"]
        llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.6,streaming=True)
        # Binds the tools to the language model
        model_with_tools = llm.bind_tools(tools)

        # Feeds the conversation history (messages) into the model
        response = model_with_tools.invoke(messages)

        # Return the model response as a new message
        return {"messages": [response]}

    return call_model_with_tools 

# Let's Define Conditional Edge Logic
# This function checks the most recent message in the state and decides whether to route to the 'action' node (ToolNode) or end.
# This function is used to control the flow of your agent, it's like a traffic signal deciding where to send the agent next.
# The function should_continue checks the last message in the agent's memory and decides:
# If the message includes a tool call, it routes to the next step (the action node, where the tool is actually used).
# If there's no tool call, it ends the conversation (__end__).

def should_continue(state: AgentState) -> Literal["action", "__end__"]:
    """Determines the next step: continue with tools or end."""
    print("DEBUG: Entering should_continue node")

    last_message = state["messages"][-1]

    # Check if the last message is an AIMessage with tool_calls
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("DEBUG: Decision: continue (route to action)")
        return "action"  # Route to the node named "action"
    else:
        print("DEBUG: Decision: end (route to END)")
        return END  # Special value indicating the end of the graph
    

# ToolNode is a prebuilt ready-to-use node from LangGraph that is specifically designed to run external tools
# Like search, calculator, database query

#from langgraph.prebuilt import ToolNode
def build_graph_one_tool(tools_list):

    # Let's Instantiate ToolNode
    tool_node = ToolNode(tools_list)

    # Define the call_node_fn, which binds the tools to the LLM and calls OpenAI API
    call_node_fn = make_call_model_with_tools(tools_list)

    # Build the Graph with One Tool using ToolNode
    graph_one_tool = StateGraph(AgentState)

    # Add nodes
    graph_one_tool.add_node("agent", call_node_fn)

    # Add the ToolNode instance directly, naming it "action"
    graph_one_tool.add_node("action", tool_node) 
    # Set entry point
    graph_one_tool.set_entry_point("agent")

    # Add a conditional edge from the agent
    # The dictionary maps the return value of 'should_continue' ("action" or END)
    # to the name of the next node ("action" or the special END value).
    graph_one_tool.add_conditional_edges(
        "agent",                 # Source node name
        should_continue,         # Function to decide the route
        {"action": "action", END: END},   # Mapping: {"decision": "destination_node_name"}
    )

    # Add edge from action (ToolNode) back to agent
    graph_one_tool.add_edge("action", "agent")

    # Compile the graph
    app_search_flight = graph_one_tool.compile()

    return app_search_flight

# Let's set up our search tool that fetches results from Tavily (a search engine wrapper)
# Setting max_results to 3 limits the number of search results.

tavily_search_tool = TavilySearchResults(max_results = 3)
# List of tools for this step
tools_list = [tavily_search_tool , get_current_date_tool ,search_flights_tool ,search_hotels_tool]

def app_call(app_search_flight, messages):
    # Initialize the state with the provided messages
    initial_state = { "messages": [
        SystemMessage(content="You are a helpful research assistant that uses tools when needed."),
        HumanMessage(content=messages)
    ]}

    # Invoke the app with the initial state
    final_state = app_search_flight.invoke(initial_state)

    # Iterate through the messages in the final state
    for i in final_state["messages"]:
        # Print the type of the message 
        print(i.type)

        # Print the content of the message 
        print(i.content)

        # Print any additional kwargs associated with the message
        if i.additional_kwargs != {}:
            print(i.additional_kwargs)

    # Return the content of the last message and the final state
    return final_state["messages"][-1].content, final_state

app_search_flight = build_graph_one_tool(tools_list)


messages = "What's the latest news on France in May 2025? Is it a good time to visit?"
output, history = app_call(app_search_flight, messages)

print("\n==================== OUTPUT ====================")
print(output)

print("\n==================== HISTORY ====================")
print(history)

