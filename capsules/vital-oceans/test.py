import asyncio
from fastmcp import Client

client = Client("main.py")


async def call_tool(city: str):
    async with client:
        result = await client.call_tool("get_weather", {"city": city})
        print(result)


asyncio.run(call_tool(city="Buenos Aires"))
