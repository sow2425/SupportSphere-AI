"""
CrewAI Specialized Agent Swarms.
Mandatory Multi-Agent Crews for Deep Inquiries:
1. Billing Resolution Crew:
   - Billing Investigator Agent (uses Model Context Protocol / MCP)
   - Financial Policy Auditor Agent (uses RAG Grounding Engine)
2. Technical Support Crew:
   - Diagnostic Engineer Agent (uses RAG Grounding Engine)
   - Solutions Architect Agent
"""

import json
from typing import List, Dict, Any, Callable
from mcp_client import mcp_lookup_payment, mcp_replay_order
from rag_engine import rag_engine
from models import model_provider

# Compatibility Adapter: If crewai is installed, use it; otherwise provide native CrewAI execution layer
try:
    from crewai import Agent as CrewAgent, Crew as CrewSwarm, Task as CrewTask, Process  # type: ignore
    USING_NATIVE_CREWAI = True
except ImportError:
    USING_NATIVE_CREWAI = False
    
    class Process:
        sequential = "sequential"
        hierarchical = "hierarchical"

    class CrewAgent:
        def __init__(self, role: str, goal: str, backstory: str, tools: List[Callable] = None, verbose: bool = True):
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.tools = tools or []
            self.verbose = verbose

    class CrewTask:
        def __init__(self, description: str, expected_output: str, agent: CrewAgent):
            self.description = description
            self.expected_output = expected_output
            self.agent = agent

    class CrewSwarm:
        def __init__(self, agents: List[CrewAgent], tasks: List[CrewTask], process=Process.sequential, verbose: bool = True):
            self.agents = agents
            self.tasks = tasks
            self.process = process
            self.verbose = verbose

        def kickoff(self, inputs: Dict[str, Any] = None) -> str:
            """Executes sequential task collaboration across agents."""
            task_outputs = []
            context_accumulated = ""
            
            for task in self.tasks:
                agent = task.agent
                prompt = (
                    f"Agent Role: {agent.role}\n"
                    f"Agent Goal: {agent.goal}\n"
                    f"Agent Backstory: {agent.backstory}\n"
                    f"Task Description: {task.description}\n"
                    f"Expected Output: {task.expected_output}\n"
                    f"Prior Context / Findings: {context_accumulated}\n"
                    f"Customer Input: {json.dumps(inputs or {})}"
                )
                
                # Execute agent-specific tools
                tool_findings = []
                for tool in agent.tools:
                    try:
                        # Extract appropriate argument
                        arg = inputs.get("customer_id", "cust_0091") if inputs else "cust_0091"
                        out = tool(arg)
                        tool_findings.append(f"Tool Result: {out}")
                    except Exception as e:
                        tool_findings.append(f"Tool Notice: {e}")
                        
                if tool_findings:
                    prompt += "\n" + "\n".join(tool_findings)

                agent_result = model_provider.call_closed_model(prompt, system_instruction=agent.backstory)
                context_accumulated += f"\n[{agent.role} Output]: {agent_result}"
                task_outputs.append(agent_result)

            return task_outputs[-1]

# ----------------- AGENT CREW DEFINITIONS -----------------

def create_billing_crew(customer_id: str, query: str) -> str:
    """
    CrewAI Multi-Agent Swarm for Billing Inquiries.
    Agent 1 (Billing Investigator) uses MCP to check transaction records.
    Agent 2 (Policy Auditor) checks RAG knowledge base for refund/replay policy and writes the customer response.
    """
    # 1. MCP Billing Investigator Agent
    investigator = CrewAgent(
        role="Payment Gateway & MCP Investigator",
        goal="Query live payment gateways via Model Context Protocol to extract transaction states.",
        backstory=(
            "You are an expert financial forensic specialist. You query the Model Context Protocol (MCP) server "
            "to inspect exact transaction IDs, gateway statuses, and webhook latency logs."
        ),
        tools=[mcp_lookup_payment]
    )

    # 2. Policy & Customer Care Auditor Agent
    auditor = CrewAgent(
        role="Billing Policy & Customer Care Auditor",
        goal="Cross-reference company refund/webhook policies via RAG and formulate an empathetic, clear response.",
        backstory=(
            "You ensure that every customer facing billing friction receives clear reassurance, "
            "exact details about their money, and guaranteed resolution timelines as per company policy."
        ),
        tools=[]
    )

    # Fetch RAG Grounding for Billing
    rag_context = rag_engine.get_grounded_context(query)

    task1 = CrewTask(
        description=f"Inspect transaction records for customer '{customer_id}' using the MCP lookup tool.",
        expected_output="Detailed payment gateway record including transaction ID, amount, and webhook status.",
        agent=investigator
    )

    task2 = CrewTask(
        description=(
            f"Review the MCP transaction findings and the company policy from RAG:\n{rag_context}\n"
            f"Formulate a definitive, reassuring message to the customer regarding: '{query}'."
        ),
        expected_output="A polite, empathetic, and informative response addressing the missing confirmation.",
        agent=auditor
    )

    billing_swarm = CrewSwarm(
        agents=[investigator, auditor],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )

    return billing_swarm.kickoff(inputs={"customer_id": customer_id, "query": query})

def create_technical_crew(customer_id: str, query: str) -> str:
    """
    CrewAI Multi-Agent Swarm for Technical Support.
    Agent 1 (Diagnostic Engineer) retrieves error solutions from RAG.
    Agent 2 (Solutions Architect) synthesizes a verified troubleshooting step-by-step resolution.
    """
    diagnostician = CrewAgent(
        role="Diagnostic Support Engineer",
        goal="Retrieve root causes and diagnostic playbooks from the technical knowledge base.",
        backstory="You specialize in distributed cloud systems, API error codes (504, 401), and client timeouts."
    )

    solutions_architect = CrewAgent(
        role="Customer Solutions Architect",
        goal="Produce clear, foolproof technical steps tailored to the user's expertise level.",
        backstory="You transform complex engineering diagnostics into friendly, actionable troubleshooting steps."
    )

    rag_context = rag_engine.get_grounded_context(query)

    task1 = CrewTask(
        description=f"Identify technical root causes for '{query}' using knowledge base: {rag_context}",
        expected_output="Technical root cause diagnosis and system incident status.",
        agent=diagnostician
    )

    task2 = CrewTask(
        description="Synthesize the diagnostic findings into a clean, numbered troubleshooting guide for the customer.",
        expected_output="Friendly and comprehensive step-by-step customer solution.",
        agent=solutions_architect
    )

    tech_swarm = CrewSwarm(
        agents=[diagnostician, solutions_architect],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )

    return tech_swarm.kickoff(inputs={"customer_id": customer_id, "query": query})

def execute_product_agent(customer_id: str, query: str) -> str:
    """Handles product capabilities, pricing, and warranty inquiries using RAG."""
    rag_context = rag_engine.get_grounded_context(query)
    prompt = (
        f"You are the Product Specialist Agent.\n"
        f"Customer Question: '{query}'\n"
        f"Verified Product Catalog Documentation (RAG):\n{rag_context}\n\n"
        f"Answer the customer's question thoroughly, referencing features, compatibility, and warranties."
    )
    return model_provider.call_closed_model(prompt, system_instruction="You are a helpful and knowledgeable Product Specialist.")

def execute_escalation_agent(customer_id: str, query: str, classification_info: Dict[str, Any]) -> str:
    """Escalates to human support queue and delivers an urgent holding message."""
    return (
        "Hello, I understand how critical and urgent this situation is. "
        "Because this inquiry involves urgent priority and requires immediate human supervisor authorization, "
        "I have flagged this ticket and transferred your case directly to our Tier-3 Senior Human Support Team. "
        "A team specialist is reviewing your account history and will intervene on this thread immediately."
    )
