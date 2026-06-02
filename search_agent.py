from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from search_tool import search_web


load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-codex",
    temperature=0,
    use_responses_api=True,
)

agent = create_agent(
    model=llm,
    tools=[search_web],
)


#agent = create_agent(
    #model=llm,
    #tools=[TavilySearch(max_results=1, include_answer=True)],
#)
