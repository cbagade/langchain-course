from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Scheme for a source used by agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Scheme for the agent response"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used by the agent to arrive at the answer")
