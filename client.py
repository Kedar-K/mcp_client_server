import argparse
import asyncio
import json
import os
import sys
from typing import Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    # methods will go here

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])   

    async def connect_to_server_config(self, server_name: str, server_config: dict[str, Any]):
        """Connect to an MCP server using a config entry."""
        command = server_config.get("command")
        if not command:
            raise ValueError(f"Server '{server_name}' is missing a command")

        args = server_config.get("args", [])
        if not isinstance(args, list):
            raise ValueError(f"Server '{server_name}' args must be a list")

        env = server_config.get("env", {})
        if env is None:
            env = {}
        if not isinstance(env, dict):
            raise ValueError(f"Server '{server_name}' env must be an object")

        merged_env = dict(os.environ)
        for key, value in env.items():
            merged_env[str(key)] = str(value)

        server_params = StdioServerParameters(
            command=str(command),
            args=[str(arg) for arg in args],
            env=merged_env,
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print(f"\nConnected to server '{server_name}' with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

def load_server_configs(config_path: str) -> dict[str, dict[str, Any]]:
    with open(config_path, "r") as config_file:
        data = json.load(config_file)

    servers = data.get("mcpServers", data)
    if not isinstance(servers, dict):
        raise ValueError("Config must contain an object of servers or a 'mcpServers' object")

    return servers

async def main():
    parser = argparse.ArgumentParser(description="MCP client")
    parser.add_argument("script", nargs="?", help="Path to local server script (.py or .js)")
    parser.add_argument("--config", help="Path to MCP servers JSON config")
    parser.add_argument("--server", help="Server name from config")
    parser.add_argument("--list-servers", action="store_true", help="List servers in config and exit")
    args = parser.parse_args()

    if not args.script and not args.config:
        print("Usage: python client.py <path_to_server_script> or --config <config.json> --server <name>")
        sys.exit(1)

    if args.config and args.script:
        print("Provide either a local script or a config file, not both.")
        sys.exit(1)

    client = MCPClient()
    try:
        if args.config:
            servers = load_server_configs(args.config)
            if args.list_servers:
                print("Available servers:", ", ".join(sorted(servers.keys())))
                return
            if not args.server:
                print("--server is required when using --config.")
                sys.exit(1)
            if args.server not in servers:
                print(f"Server '{args.server}' not found in config.")
                sys.exit(1)
            await client.connect_to_server_config(args.server, servers[args.server])
        else:
            await client.connect_to_server(args.script)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
