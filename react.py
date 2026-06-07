from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from config import MODEL_NAME

load_dotenv()

@tool
def triple(num:float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

llm = init_chat_model(model=MODEL_NAME, temperature=0).bind_tools(tools)
