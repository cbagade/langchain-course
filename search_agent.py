from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from agent_response import AgentResponse
from config import OPENAI_CHAT_MODEL
from search_tool import search_web


load_dotenv()

llm = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
    temperature=0,
    use_responses_api=True,
)

agent = create_agent(
    model=llm,
    tools=[search_web],
    response_format=AgentResponse
)


#agent = create_agent(
    #model=llm,
    #tools=[TavilySearch(max_results=1, include_answer=True)],
#)
