# mcp_server/run.py
"""Entry point for the MCP server."""
import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run(
        transport="stdio",  # Standard MCP transport for universal compatibility
        # For HTTP transport (optional, for remote connections):
        # transport="http", host="0.0.0.0", port=8001
    )