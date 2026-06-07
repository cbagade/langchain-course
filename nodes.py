from dotenv import load_dotenv
import json
import re
import uuid

from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
When a tool is needed, call the tool directly. Do not write fake JSON tool calls in the response.
Call only one tool at a time. If you need to triple a searched value, wait for the search result first.
"""
TOOLS_BY_NAME = {tool.name: tool for tool in tools}
JSON_DECODER = json.JSONDecoder()


def _extract_text_tool_calls(content: str) -> list[dict]:
    tool_calls = []
    for match in re.finditer(r"{", content):
        try:
            payload, _ = JSON_DECODER.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        tool_name = payload.get("name")
        if tool_name not in TOOLS_BY_NAME:
            continue

        tool_calls.append(
            {
                "name": tool_name,
                "args": payload.get("parameters", {}),
                "id": f"call_{uuid.uuid4().hex}",
            }
        )
        break
    return tool_calls


def _requested_triple(state: MessagesState) -> bool:
    return any(
        getattr(message, "type", None) == "human"
        and "triple" in str(message.content).lower()
        for message in state["messages"]
    )


def _already_called_triple(state: MessagesState) -> bool:
    return any(getattr(message, "name", None) == "triple" for message in state["messages"])


def _extract_temperature(content: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?(?:F|fahrenheit|C|celsius)\b", content)
    if match:
        return float(match.group(1))
    return None

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    if not response.tool_calls and isinstance(response.content, str):
        tool_calls = _extract_text_tool_calls(response.content)
        if tool_calls:
            response = AIMessage(content=response.content, tool_calls=tool_calls)
        elif _requested_triple(state) and not _already_called_triple(state):
            temperature = _extract_temperature(response.content)
            if temperature is not None:
                response = AIMessage(
                    content=response.content,
                    tool_calls=[
                        {
                            "name": "triple",
                            "args": {"num": temperature},
                            "id": f"call_{uuid.uuid4().hex}",
                        }
                    ],
                )
    return {"messages": [response]}

tool_node = ToolNode(tools)
