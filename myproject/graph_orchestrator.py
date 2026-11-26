"""
LangGraph Stateful Multi-Agent Orchestrator.
Coordinates:
1. Query Classification, Sentiment & Priority Detection (Open Model)
2. Dynamic Routing & Branching Logic
3. CrewAI Autonomous Swarms Execution (Billing Crew & Tech Crew)
4. Escalation to Human Support (Human-in-the-Loop)
"""

import json
from typing import Dict, Any, Callable

# Load State schemas (with automatic built-in fallback if state.py is missing)
try:
    from state import CustomerSupportState, QueryClassification
except ImportError:
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

from models import model_provider
from crew_agents import (
    create_billing_crew,
    create_technical_crew,
    execute_product_agent,
    execute_escalation_agent
)

# Compatibility Adapter: If langgraph is installed, use it; otherwise provide native StateGraph execution layer
try:
    from langgraph.graph import StateGraph as NativeStateGraph, END as NATIVE_END  # type: ignore
    USING_NATIVE_LANGGRAPH = True
    END = NATIVE_END
except ImportError:
    USING_NATIVE_LANGGRAPH = False
    END = "__END__"

    class NativeStateGraph:
        """Native implementation of LangGraph StateGraph specification."""
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes: Dict[str, Callable] = {}
            self.edges: Dict[str, str] = {}
            self.conditional_edges: Dict[str, tuple] = {}
            self.entry_point: str = None

        def add_node(self, name: str, func: Callable):
            self.nodes[name] = func

        def set_entry_point(self, name: str):
            self.entry_point = name

        def add_edge(self, start_key: str, end_key: str):
            self.edges[start_key] = end_key

        def add_conditional_edges(self, source_node: str, condition_func: Callable, routing_map: Dict[str, str]):
            self.conditional_edges[source_node] = (condition_func, routing_map)

        def compile(self, checkpointer=None):
            return CompiledGraph(self)

    class CompiledGraph:
        def __init__(self, graph: NativeStateGraph):
            self.graph = graph

        def invoke(self, state: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
            current_node_name = self.graph.entry_point
            current_state = dict(state)

            while current_node_name and current_node_name != END:
                # Execute current node
                node_fn = self.graph.nodes.get(current_node_name)
                if not node_fn:
                    break
                updates = node_fn(current_state)
                if updates:
                    current_state.update(updates)

                # Determine next node via conditional edges or standard edges
                if current_node_name in self.graph.conditional_edges:
                    cond_fn, routing_map = self.graph.conditional_edges[current_node_name]
                    route_key = cond_fn(current_state)
                    next_node = routing_map.get(route_key, END)
                else:
                    next_node = self.graph.edges.get(current_node_name, END)

                current_node_name = next_node

            return current_state

# ----------------- LANGGRAPH NODES -----------------

def triage_classification_node(state: CustomerSupportState) -> Dict[str, Any]:
    """
    Node 1: Query Classification, Sentiment & Urgency Triage.
    Uses Open Model (Groq Llama-3 / HuggingFace) for sub-second structured extraction.
    """
    last_message = state["messages"][-1]["content"]
    system_prompt = (
        "You are an enterprise AI Support Triage Agent. "
        "Analyze the customer query and output a strict JSON object with fields: "
        "'category' ('technical', 'billing', 'product', 'general', 'escalation'), "
        "'priority' ('low', 'medium', 'high', 'urgent'), "
        "'sentiment' ('positive', 'neutral', 'frustrated', 'angry'), "
        "'reasoning' (string), and 'requires_human' (boolean). "
        "If payment is deducted without confirmation, category must be 'billing' and priority 'high'. "
        "Output JSON only."
    )
    
    raw_response = model_provider.call_open_model(last_message, system_instruction=system_prompt)
    
    # Clean and parse JSON
    try:
        cleaned = raw_response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        parsed = json.loads(cleaned)
        classification = QueryClassification(**parsed)
    except Exception as e:
        # Fallback if LLM output was malformed
        fallback_json = json.loads(model_provider._local_classification_fallback(last_message))
        classification = QueryClassification(**fallback_json)

    is_escalated = classification.requires_human or (classification.sentiment == "angry") or (classification.priority == "urgent")

    return {
        "classification": classification,
        "is_escalated": is_escalated,
        "agent_scratchpad": [
            f"[Triage Agent (Open Model)]: Identified Category='{classification.category}', "
            f"Priority='{classification.priority.upper()}', Sentiment='{classification.sentiment}', "
            f"Escalate={is_escalated}"
        ]
    }

def billing_crew_node(state: CustomerSupportState) -> Dict[str, Any]:
    """Node 2: Invokes the CrewAI Billing Resolution Multi-Agent Swarm."""
    last_msg = state["messages"][-1]["content"]
    cust_id = state.get("customer_id", "cust_0091")
    
    crew_output = create_billing_crew(customer_id=cust_id, query=last_msg)
    
    return {
        "final_response": crew_output,
        "assigned_agent": "CrewAI Billing Resolution Swarm (MCP + RAG)",
        "agent_scratchpad": ["[Billing Crew]: Ran forensic MCP gateway lookup and generated grounded customer resolution."]
    }

def technical_crew_node(state: CustomerSupportState) -> Dict[str, Any]:
    """Node 3: Invokes the CrewAI Technical Support Multi-Agent Swarm."""
    last_msg = state["messages"][-1]["content"]
    cust_id = state.get("customer_id", "cust_0091")
    
    crew_output = create_technical_crew(customer_id=cust_id, query=last_msg)
    
    return {
        "final_response": crew_output,
        "assigned_agent": "CrewAI Technical Support Swarm (RAG)",
        "agent_scratchpad": ["[Technical Crew]: Diagnosed root cause via RAG and created remediation guide."]
    }

def product_node(state: CustomerSupportState) -> Dict[str, Any]:
    """Node 4: Invokes the Product Specialist Agent."""
    last_msg = state["messages"][-1]["content"]
    cust_id = state.get("customer_id", "cust_0091")
    
    product_output = execute_product_agent(customer_id=cust_id, query=last_msg)
    
    return {
        "final_response": product_output,
        "assigned_agent": "Product Specialist Agent (RAG)",
        "agent_scratchpad": ["[Product Agent]: Checked catalog specifications and warranty coverage."]
    }

def escalation_node(state: CustomerSupportState) -> Dict[str, Any]:
    """Node 5: Human-in-the-Loop Escalation Node."""
    last_msg = state["messages"][-1]["content"]
    cust_id = state.get("customer_id", "cust_0091")
    classification = state["classification"]
    
    escalation_output = execute_escalation_agent(
        customer_id=cust_id,
        query=last_msg,
        classification_info=classification.model_dump() if classification else {}
    )
    
    return {
        "final_response": escalation_output,
        "assigned_agent": "Escalation Agent (Human Support Handoff)",
        "is_escalated": True,
        "agent_scratchpad": ["[Escalation Agent]: Handoff packet prepared and routed to Tier-3 Human Queue."]
    }

# ----------------- CONDITIONAL ROUTER -----------------

def routing_logic(state: CustomerSupportState) -> str:
    """Evaluates classification output to route to the appropriate agent or swarm."""
    classification = state.get("classification")
    if not classification:
        return "escalation_node"

    # Strict escalation check: If customer is angry or query demands human approval
    if classification.sentiment == "angry" or (classification.priority == "urgent" and classification.requires_human):
        return "escalation_node"

    category = classification.category
    if category == "billing":
        return "billing_crew_node"
    elif category == "technical":
        return "technical_crew_node"
    elif category == "product":
        return "product_node"
    else:
        return "escalation_node"

# ----------------- BUILD GRAPH -----------------

workflow = NativeStateGraph(CustomerSupportState)

# Add Nodes
workflow.add_node("triage_agent", triage_classification_node)
workflow.add_node("billing_crew_node", billing_crew_node)
workflow.add_node("technical_crew_node", technical_crew_node)
workflow.add_node("product_node", product_node)
workflow.add_node("escalation_node", escalation_node)

# Set Entry Point
workflow.set_entry_point("triage_agent")

# Add Conditional Edges
workflow.add_conditional_edges(
    "triage_agent",
    routing_logic,
    {
        "billing_crew_node": "billing_crew_node",
        "technical_crew_node": "technical_crew_node",
        "product_node": "product_node",
        "escalation_node": "escalation_node"
    }
)

# Connect Specialist Nodes to END
workflow.add_edge("billing_crew_node", END)
workflow.add_edge("technical_crew_node", END)
workflow.add_edge("product_node", END)
workflow.add_edge("escalation_node", END)

# Compile Orchestrator
support_orchestrator = workflow.compile()
support_graph = support_orchestrator  # Alias for backwards compatibility

if __name__ == "__main__":
    test_state = {
        "ticket_id": "TCK-1001",
        "customer_id": "cust_0091",
        "customer_name": "Alex Mercer",
        "customer_tier": "enterprise",
        "messages": [{"role": "user", "content": "My payment was deducted, but my order was not confirmed."}],
        "classification": None,
        "assigned_agent": None,
        "retrieved_context": None,
        "agent_scratchpad": [],
        "final_response": None,
        "is_escalated": False
    }
    print("Executing LangGraph Orchestrator with CrewAI Swarms...")
    result = support_orchestrator.invoke(test_state)
    print("\n--- Final Response ---")
    print(result["final_response"])
    print("\n--- Handled By ---")
    print(result["assigned_agent"])
    print("\n--- Scratchpad ---")
    for s in result["agent_scratchpad"]:
        print(s)
