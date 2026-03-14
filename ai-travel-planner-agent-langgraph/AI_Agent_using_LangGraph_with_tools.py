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
    
    
@tool
def get_current_date_tool():
    """Returns the current date in 'YYYY-MM-DD' format. Useful for finding flights/hotels relative to today."""
    return date.today().isoformat()

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
    app = graph_one_tool.compile()

    return app

# Service clients
from amadeus import Client, ResponseError

# Configure Amadeus Client
# We'll only initialize it if keys are provided, inside the tool later
amadeus_client = Client(
    client_id = amadeus_api_key,
    client_secret = amadeus_api_secret,
    hostname = "test",  # Start with the test environment
)

@tool
def search_flights_tool(
    origin_code: str,
    destination_code: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    travel_class: str = "ECONOMY",
    currency: str = "USD",
    max_offers: int = 5,
):
    """
    Searches live flight prices and availability via Amadeus Flight Offers Search API.

    Required:
        origin_code, destination_code - IATA airport/city codes (e.g., 'YYZ', 'LHR')
        departure_date - 'YYYY-MM-DD'

    Optional:
        return_date - for round-trips; omit for one-way
        adults - number of adult passengers (default 1)
        travel_class - 'ECONOMY', 'PREMIUM_ECONOMY', 'BUSINESS', 'FIRST'
        currency - 3-letter code for pricing (default USD)
        max_offers - how many offers to list back
    """
    print(
    f"DEBUG: Calling Amadeus Flight Search - "
    f"{origin_code}->{destination_code}, "
    f"Depart {departure_date}, Return {return_date}, "
    f"Adults {adults}, Class {travel_class}"
)

# --- Call Amadeus Flight Offers Search API ---
    flight_search_params = {
        "originLocationCode": origin_code,
        "destinationLocationCode": destination_code,
        "departureDate": departure_date,
        "adults": adults,
        "travelClass": travel_class,
        "currencyCode": currency,
        "max": max_offers
    }

    if return_date:
        flight_search_params["returnDate"] = return_date

    response = amadeus_client.shopping.flight_offers_search.get(**flight_search_params)

# --- Parse the response ---
    if not response.data:
        return (
            f"No flight offers found for {origin_code} -> {destination_code} on "
            f"{departure_date}{' (return ' + return_date + ')' if return_date else ''}."
        )

    results = []
    for offer in response.data[:max_offers]:
        price = offer["price"]["total"]
        airline = offer["validatingAirlineCodes"][0]
        itinerary = offer["itineraries"][0]
        segments = itinerary["segments"]
        first_leg = segments[0]
        last_leg = segments[-1]

        dep_time = first_leg["departure"]["at"][:16].replace("T", " ")
        arr_time = last_leg["arrival"]["at"][:16].replace("T", " ")
        duration = itinerary["duration"].replace("PT", "")

        results.append(
            f"{airline} | {dep_time} -> {arr_time} | {duration} | {price} {currency}"
        )

        return "Found flight options:\n- " + "\n- ".join(results)
    
@tool
def search_hotels_tool(city_code: str, check_in_date: str, check_out_date: str, adults: int = 1):
    """
    Searches for available hotel options in a specific city for given dates using Amadeus.
    Requires the IATA city code (e.g., 'PAR', 'BER') and dates in 'YYYY-MM-DD' format.
    Use get_current_date_tool first if dates are relative.
    """

    print(
        f"DEBUG: Calling Amadeus Hotel Search - City: {city_code}, "
        f"Check-in: {check_in_date}, Check-out: {check_out_date}, Adults: {adults}"
    )

    # Call Amadeus API - Hotel Search (find hotels by city)
    hotel_list_response = amadeus_client.reference_data.locations.hotels.by_city.get(
        cityCode=city_code,
        radius=50,
        radiusUnit="KM"
    )

    if not hotel_list_response.data or len(hotel_list_response.data) == 0:
        return f"No hotels found listed in Amadeus for city code {city_code}."

    # Get hotel IDs from the response (limit to first 5 for offers search)
    hotel_ids = [hotel["hotelId"] for hotel in hotel_list_response.data[:5]]

    # Now search for offers for these specific hotels
    hotel_offer_response = amadeus_client.shopping.hotel_offers_search.get(
        hotelIds=",".join(hotel_ids),
        checkInDate=check_in_date,
        checkOutDate=check_out_date,
        adults=adults
    )
    # Process the response (simplified)
    if hotel_offer_response.data and len(hotel_offer_response.data) > 0:
        results = []
        for offer in hotel_offer_response.data[:5]:  # Limit to showing 3 offers
            hotel_name = offer.get("hotel", {}).get("name", "N/A")
            price = offer.get("offers", [{}])[0].get("price", {}).get("total", "N/A")
            currency = offer.get("offers", [{}])[0].get("price", {}).get("currency", "")
            results.append(f"Hotel: {hotel_name}, Price: {price} {currency} (approx)")

        return "Found hotel options:\n- " + "\n- ".join(results)

    else:
        return f"No available hotel offers found for the dates in {city_code} among the checked hotels."

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

