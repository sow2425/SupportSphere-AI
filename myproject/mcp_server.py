"""
Model Context Protocol (MCP) Server for Customer Support.
Standardized JSON-RPC 2.0 tool server exposing enterprise payment and order context.
"""

import json
from typing import Dict, Any, List

class CustomerSupportMCPServer:
    """
    Standard Model Context Protocol (MCP) Server.
    Provides external enterprise tools to AI agents via JSON-RPC.
    """
    def __init__(self):
        # Mock database for transactions and customer accounts
        self.transactions_db = {
            "cust_0091": {
                "transaction_id": "tx_992182",
                "amount": "$149.00",
                "currency": "USD",
                "status": "CHARGED",
                "order_id": "ORD-58291",
                "order_status": "PENDING_WEBHOOK_CONFIRMATION",
                "payment_gateway": "Stripe",
                "timestamp": "2026-11-24T09:15:00Z",
                "error_details": "Webhook timeout: Bank debited, awaiting merchant receipt confirmation."
            },
            "cust_0042": {
                "transaction_id": "tx_441029",
                "amount": "$29.00",
                "currency": "USD",
                "status": "SETTLED",
                "order_id": "ORD-11029",
                "order_status": "CONFIRMED",
                "payment_gateway": "PayPal",
                "timestamp": "2026-11-23T14:30:00Z",
                "error_details": None
            }
        }
        
        self.tools_manifest = [
            {
                "name": "lookup_payment_record",
                "description": "Retrieve live payment gateway details, charges, and webhook statuses for a customer.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "Unique identifier of the customer (e.g., 'cust_0091')."
                        }
                    },
                    "required": ["customer_id"]
                }
            },
            {
                "name": "trigger_order_replay",
                "description": "Trigger a safe replay of a stuck webhook to finalize a pending order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Order identifier (e.g., 'ORD-58291')."
                        }
                    },
                    "required": ["order_id"]
                }
            }
        ]

    def handle_json_rpc(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Processes JSON-RPC 2.0 protocol requests."""
        req_id = request_payload.get("id", 1)
        method = request_payload.get("method")
        params = request_payload.get("params", {})

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.tools_manifest}
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            return self._execute_tool(req_id, tool_name, arguments)

        elif method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "CustomerSupportMCPServer",
                        "version": "1.0.0"
                    }
                }
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"}
        }

    def _execute_tool(self, req_id: Any, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "lookup_payment_record":
            customer_id = arguments.get("customer_id", "cust_0091")
            record = self.transactions_db.get(customer_id)
            if record:
                content = json.dumps(record, indent=2)
            else:
                content = f"No transaction record found for customer '{customer_id}'."
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": content}]
                }
            }

        elif tool_name == "trigger_order_replay":
            order_id = arguments.get("order_id")
            content = f"Replay signal queued for order '{order_id}'. Confirmation dispatched to customer email within 5 minutes."
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": content}]
                }
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": f"Unknown tool '{tool_name}'"}
        }

# Global Server Instance
mcp_server = CustomerSupportMCPServer()

if __name__ == "__main__":
    # Test JSON-RPC execution
    init_res = mcp_server.handle_json_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    print("MCP Init:", init_res)
    tools_res = mcp_server.handle_json_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    print("MCP Tools List:", len(tools_res["result"]["tools"]), "tools available")
    call_res = mcp_server.handle_json_rpc({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "lookup_payment_record", "arguments": {"customer_id": "cust_0091"}}
    })
    print("MCP Tool Call Result:", call_res["result"]["content"][0]["text"])
