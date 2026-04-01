import asyncio
from fastmcp import Client
import json

async def main():
    try:
        # test SSE connection to the server
        client = Client(transport="sse://127.0.0.1:8000/mcp/sse")
        async with client:
            tools = await client.list_tools()
            print("Successfully fetched tools:", [t.name for t in tools])
    except Exception as e:
        print("Failed with sse://.../mcp/sse:", e)

    try:
        client = Client(transport="http://127.0.0.1:8000/mcp/sse")
        async with client:
            tools = await client.list_tools()
            print("Successfully fetched tools:", [t.name for t in tools])
    except Exception as e:
        print("Failed with http://.../mcp/sse:", e)

    try:
        client = Client(transport="http://127.0.0.1:8000/mcp")
        async with client:
            tools = await client.list_tools()
            print("Successfully fetched tools:", [t.name for t in tools])
    except Exception as e:
        print("Failed with http://.../mcp:", e)

asyncio.run(main())
