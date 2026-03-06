import os , requests , asyncio
from dotenv import load_dotenv
from datetime import datetime
from agents import Agent , Runner , function_tool , SQLiteSession ,handoff ,RunContextWrapper ,input_guardrail,CodeInterpreterTool , TResponseInputItem ,GuardrailFunctionOutput ,RunResult
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from openai import OpenAI
from typing_extensions import TypedDict
from pydantic import BaseModel
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

print("OpenAI api key loaded successfully from .env file")
print(f"Openai api key :", OPENAI_API_KEY[:10] if OPENAI_API_KEY else "Missing OpenAI API key")
print(f"Tavily api key:", TAVILY_API_KEY[:10] if TAVILY_API_KEY else "Missing Tavily API key")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
print("OpenAI client configured successfully")

class TavilySearchParams(TypedDict):
    query: str
    max_results: int
@function_tool()
def tavily_search(params: TavilySearchParams) -> str:
    """calls the Tavily API and returns the string summarry of search results.
    params: TavilySearchParams : a dictionary with keys 'query' (the search query) and 'max_results' (the maximum number of results to return)
    returns: a string summarizing the search results from the Tavily API
    """
    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": params["query"],
        "max_results": params.get("max_results", 3)
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        results= response.json().get("results",[]) # Extracting the 'results' field from the API response, defaulting to an empty list if not present
        summary = "\n".join(
            [f"{i+1}. {result['title']} - {result['content']}"
              for i, result in enumerate(results)]) # Creating a summary string by enumerating through the search results and formatting each result with its title and content 
        return summary if summary else "No results found."
    else:
        raise Exception(f"Tavily API error: {response.status_code} - {response.text}")
    

    
# Define a data model for the complete search plan
# This contains a list of SearchPlanItems

class PoliticalTopicOutput(BaseModel):
    is_political: bool
    reason: str

politics_guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about political topics, politicians, elections, government.",
    output_type=PoliticalTopicOutput
)

@input_guardrail
async def politics_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:

    result = await Runner.run(
        politics_guardrail_agent,
        input,
        context=ctx.context
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_political
    )
    
    
    
class SearchPlanItem(BaseModel):
    reason: str
    query: str
class SearchPlan(BaseModel):
    searches: list[SearchPlanItem]


# Get today's date in YYYY-MM-DD format
date = datetime.now().strftime("%Y-%m-%d")

# Create an AI agent called "Planner"
planner_agent = Agent(
    name="Planner",
    instructions=f"""
Current date: {date}
Context: You are a research planner agent tasked with designing a comprehensive research plan for a user request.
You have access to web search tools and should utilize the current date ({date}) when planning.

Instruction: Break down the user's request into 3 distinct web searches, each with a clear reason and a specific query.
Ensure coverage of recent news, company fundamentals, risks, sentiment, and broader context.

Input: The user's research request and the current date.
Output: A list of search plan items, each with a 'reason' and a 'query', formatted as a JSON object matching the SearchPlan.
""",
    model="gpt-4o-mini",
    #testinng
    #tools=[tavily_search], 
    output_type=SearchPlan,
    input_guardrails=[politics_guardrail]  # THIS IS THE GUARDRAIL THAT PREVENTS POLITICS
    
 
)

# async def test_planner():
#        q1="valid state battery companies in 2026"
#        run= await Runner.run(planner_agent, q1)
#        print("Search Plan:", run.final_output)
       
# if __name__ == "__main__":
#     asyncio.run(test_planner())

# Let's define the Search Agent, which finds and summarizes the most recent, relevant information for a research query

class Summary(BaseModel):
    summary: str


search_agent = Agent(
    name="Searcher",
    instructions="""
        Context: You are a search specialist agent with access to the Tavily web search tool.
        Your goal is to provide up-to-date, relevant information for a research task.
        
        Instruction: Use Tavily search to find the most recent and pertinent information related to the user's query.
        Summarize your findings clearly and concisely in no more than 200 words.
        
        Input: The user's search query.
        
        Output: A concise summary (<200 words) of the most relevant and recent information found.
        """,
    tools=[tavily_search],
    model="gpt-4o-mini",
    output_type=Summary
)

# Let's define the fundamentals Analysis Agent

fundamentals_agent = Agent(
    name="FundamentalsAnalyst",
    instructions="""
        Context: You are a financial analyst specializing in company fundamentals.
        
        Instruction: Carefully analyze the provided notes to assess the company's financial fundamentals,
        including revenue, growth, and profitability.
        
        Input: Notes containing relevant financial data and qualitative information about the company.
        
        Output: A concise summary (<200 words) highlighting key points about the company's revenue,
        growth trajectory, and profit margins.
        
        Tools: The following tools are available for comprehensive research on the company:
        - tavily_search: Search the web for information about the company.
        """,
    output_type=Summary,
    model="gpt-4o-mini",
    tools=[tavily_search]
)

SENTIMENT_PROMPT = """
        Context: You are a sentiment analyst specializing in evaluating online sentiment about companies.

        Instruction: Carefully analyze the provided notes and search online sources to determine the current sentiment (positive, negative, or neutral) about the company.

        Input: Notes containing relevant information and search results about the company.

        Output: A concise summary (≤200 words) highlighting the overall sentiment, supporting evidence, and any notable trends or shifts.

        Tools: The following tools are available for comprehensive sentiment research on the company:
        - tavily_search: Search the web for information about the company.
        """

sentiment_agent = Agent(
    name="SentimentAnalyst",
    instructions=SENTIMENT_PROMPT,
    output_type=Summary,
    model="gpt-4o-mini",
    tools=[tavily_search]
)

# Finally, let's define the Risk Analysis Agent, which will analyze the research notes to identify potential risks associated with the company or topic being researched. The agent will provide a concise summary of the key risks in no more than 200 words.
class FinalReport(BaseModel):
    short_summary: str
    markdown_report: str
    follow_up_questions: list[str]
    
async def extract_summary(run_result: RunResult) -> str:
    """Extracts the 'summary' field from the final_output of an agent run."""
    return run_result.final_output.summary



# Define the writer agent

writer_agent = Agent(
    name="Writer",
    instructions="""
        Context: You are an expert research writer preparing a comprehensive investment report on a company.

        Instruction: Thoroughly analyze the provided search snippets and analyst summaries.
        Synthesize these into a cohesive, well-structured report.

        Ensure the writing is precise, professional, and tailored for an investment decision-making context.

        You must always use the 'search' tool to gather and incorporate up-to-date information in your report.

        Input: A set of search snippets and analyst summaries containing relevant information about the company.

        Output: A markdown-formatted report (minimum 600 words) including:
        - an executive summary
        - 3-5 well-crafted follow-up research questions

        Tools:
        - fundamentals: Get fundamentals analysis (optional)
        - search: Get search results (required)
        """,
    model="gpt-4o-mini",
    output_type=FinalReport,
    tools=[
        fundamentals_agent.as_tool("fundamentals",
                                    "get fundamentals analysis",
                                    custom_output_extractor=extract_summary),
            
        search_agent.as_tool("search",
                             "get relevant search information",
                              custom_output_extractor=extract_summary),

        sentiment_agent.as_tool("sentiment",
                             "get sentiment analysis",
                              custom_output_extractor=extract_summary)
        ]
    
)

# async def test_writer():
#       q1="valid state battery companies in 2026"
#       run= await Runner.run(writer_agent, q1)
#       print("Search Plan:", run.final_output)
      
# if __name__ == "__main__":
#      asyncio.run(test_writer())

session=SQLiteSession("research_agent_handoff.db")

#so from here handoff will be started
class PlannerToWriterInput(BaseModel):
    original_query: str
    search_plan: SearchPlan
def on_planner_to_writer(ctx: RunContextWrapper[None], input_data: PlannerToWriterInput):
    print("➡ Transfer: Planner → Writer")
    
handoff_to_writer = handoff(
    agent = writer_agent,
    input_type = PlannerToWriterInput,
    on_handoff = on_planner_to_writer,
    tool_name_override = "transfer_to_writer",
    tool_description_override = "Transfer to writer with original query and search plan"
)

planner_with_handoff = planner_agent.clone(
    instructions=(f"""{RECOMMENDED_PROMPT_PREFIX}\n\n"""
    + planner_agent.instructions
    + "\n\nWhen you have produced the SearchPlan, call the handoff tool 'handoff_to_writer'"),
    handoffs=[handoff_to_writer])
    
# async def run_handoff_from_planna_to_tritter(user_input: str):
#     print(f"""User Query: {user_input}""")
#     report_res = await Runner.run(planner_with_handoff, user_input, session=session)
#     report=report_res.final_output
#     print(f"""Final Report:\n{report.staticmethod}""")
#     print(f"""short summary:\n{report.full_report}""")
    
    
    
async def test_planner_to_writer_handoff():
    user_query = "What are the most promising battery companies in 2026?"
    handoff_result=await Runner.run(planner_with_handoff, user_query, session=session)
    print(f"Final Report:\n{handoff_result.final_output.markdown_report}")
if __name__ == "__main__":  
     asyncio.run(test_planner_to_writer_handoff())
    
    
    