"""
Model Context Protocol (MCP) Client Adapter.
Enables LangGraph nodes and CrewAI agents to invoke tools exposed by the MCP Server.
"""

from typing import Dict, Any, Optional
from mcp_server import mcp_server

class MCPClient:
    """Lightweight Python client to interact with the local MCP server."""
    
    def __init__(self, server=mcp_server):
        self.server = server
        self.initialized = False
        self._initialize()

    def _initialize(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        resp = self.server.handle_json_rpc(req)
        if "result" in resp:
            self.initialized = True

    def list_tools(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        resp = self.server.handle_json_rpc(req)
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Invokes an MCP tool and extracts text content from the response."""
        req = {
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        resp = self.server.handle_json_rpc(req)
        if "result" in resp:
            content_list = resp["result"].get("content", [])
            if content_list and "text" in content_list[0]:
                return content_list[0]["text"]
        elif "error" in resp:
            return f"MCP Error: {resp['error'].get('message')}"
        return "No output from MCP tool."

# Global singleton client
mcp_client = MCPClient()

# Functional helper tool for agents
def mcp_lookup_payment(customer_id: str) -> str:
    """Tool callable by CrewAI or LangGraph to lookup payment via MCP."""
    return mcp_client.call_tool("lookup_payment_record", {"customer_id": customer_id})

def mcp_replay_order(order_id: str) -> str:
    """Tool callable by CrewAI or LangGraph to replay order webhook via MCP."""
    return mcp_client.call_tool("trigger_order_replay", {"order_id": order_id})
