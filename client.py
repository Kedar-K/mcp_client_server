import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from typing import Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

_INVALID_TOOL_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.sessions: dict[str, ClientSession] = {}
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

        session = await self._open_session(server_params)
        self.session = session
        self.sessions = {"local": session}

        # List available tools
        response = await session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])   

    async def _open_session(self, server_params: StdioServerParameters) -> ClientSession:
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        return session

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

        session = await self._open_session(server_params)
        self.session = session
        self.sessions[server_name] = session

        response = await session.list_tools()
        tools = response.tools
        print(f"\nConnected to server '{server_name}' with tools:", [tool.name for tool in tools])

    async def connect_to_servers_config(self, servers: dict[str, Any]):
        """Connect to all MCP servers in a config."""
        if not servers:
            raise ValueError("No servers found in config")
        for server_name, server_config in servers.items():
            await self.connect_to_server_config(server_name, server_config)

    def _make_safe_tool_name(self, server_name: str, tool_name: str, existing: set[str]) -> str:
        base = f"{server_name}__{tool_name}"
        safe = _INVALID_TOOL_CHARS.sub("_", base)
        if not safe:
            safe = "tool"
        if len(safe) > 128:
            digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
            max_prefix = 128 - 2 - len(digest)
            prefix = safe[:max_prefix] if max_prefix > 0 else ""
            safe = f"{prefix}__{digest}" if prefix else f"tool__{digest}"
        if safe in existing:
            digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
            max_prefix = 128 - 2 - len(digest)
            prefix = safe[:max_prefix] if max_prefix > 0 else ""
            candidate = f"{prefix}__{digest}" if prefix else f"tool__{digest}"
            if candidate in existing:
                suffix = 1
                while True:
                    extra = f"_{suffix}"
                    max_prefix = 128 - len(extra)
                    candidate = f"{safe[:max_prefix]}{extra}"
                    if candidate not in existing:
                        safe = candidate
                        break
                    suffix += 1
            else:
                safe = candidate
        return safe

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        if self.session is None and not self.sessions:
            raise RuntimeError("No MCP server connection established")

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        multi_server = len(self.sessions) > 1
        available_tools = []
        tool_name_map: dict[str, tuple[str, str]] = {}
        single_session: Optional[ClientSession] = None

        if multi_server:
            existing_names: set[str] = set()
            for server_name, session in self.sessions.items():
                response = await session.list_tools()
                for tool in response.tools:
                    exposed_name = self._make_safe_tool_name(server_name, tool.name, existing_names)
                    existing_names.add(exposed_name)
                    tool_name_map[exposed_name] = (server_name, tool.name)
                    description = tool.description or ""
                    label = f"[{server_name}] {tool.name}"
                    if description:
                        description = f"{label}: {description}"
                    else:
                        description = label
                    available_tools.append({
                        "name": exposed_name,
                        "description": description,
                        "input_schema": tool.inputSchema
                    })
        else:
            single_session = self.session or next(iter(self.sessions.values()))
            response = await single_session.list_tools()
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

        while True:
            assistant_message_content = []
            tool_results = []
            for content in response.content:
                assistant_message_content.append(content)
                if content.type == 'text':
                    final_text.append(content.text)
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input

                    if multi_server:
                        mapping = tool_name_map.get(tool_name)
                        if mapping is None:
                            raise ValueError(f"Tool name '{tool_name}' not found in mapping")
                        server_name, actual_tool_name = mapping
                        session = self.sessions.get(server_name)
                        if session is None:
                            raise ValueError(f"Server '{server_name}' not connected")
                    else:
                        session = single_session or self.session
                        if session is None:
                            raise RuntimeError("No MCP server connection established")
                        actual_tool_name = tool_name

                    result = await session.call_tool(actual_tool_name, tool_args)
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result.content
                    })

            if not tool_results:
                break

            messages.append({
                "role": "assistant",
                "content": assistant_message_content
            })
            messages.append({
                "role": "user",
                "content": tool_results
            })

            response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )

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
    parser.add_argument("--all", action="store_true", help="Connect to all servers in config")
    parser.add_argument("--list-servers", action="store_true", help="List servers in config and exit")
    args = parser.parse_args()

    if not args.script and not args.config:
        print("Usage: python client.py <path_to_server_script> or --config <config.json> --server <name>|--all")
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
            if args.server and args.all:
                print("Provide either --server or --all, not both.")
                sys.exit(1)
            if not args.server and not args.all:
                print("--server or --all is required when using --config.")
                sys.exit(1)
            if args.all:
                await client.connect_to_servers_config(servers)
            else:
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


# emailing
# trip with family and friends suggestions