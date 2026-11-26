"""
State schema definitions for LangGraph and CrewAI workflows.
"""

from typing import TypedDict, Annotated, List, Optional, Literal
from pydantic import BaseModel, Field
import operator

class QueryClassification(BaseModel):
    category: Literal["technical", "billing", "product", "general", "escalation"] = Field(
        ..., description="The functional business category of the inquiry."
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        ..., description="Ticket urgency level based on monetary impact or business disruption."
    )
    sentiment: Literal["positive", "neutral", "frustrated", "angry"] = Field(
        ..., description="Sentiment detected in customer's message."
    )
    reasoning: str = Field(..., description="Justification for categorization.")
    requires_human: bool = Field(
        default=False, description="Flag indicating immediate human intervention required."
    )

class CustomerSupportState(TypedDict):
    ticket_id: str
    customer_id: str
    customer_name: str
    customer_tier: str
    messages: Annotated[List[dict], operator.add]
    classification: Optional[QueryClassification]
    assigned_agent: Optional[str]
    retrieved_context: Optional[str]
    agent_scratchpad: Annotated[List[str], operator.add]
    final_response: Optional[str]
    is_escalated: bool
