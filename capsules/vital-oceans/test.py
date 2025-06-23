import asyncio
from fastmcp import Client
from pprint import pprint

client = Client("http://localhost:8000/mcp")

async def test_tools(coords):
    async with client:

        results = []

        result = await client.call_tool("query_fishing_data", {"polygon_coords": coords})
        print("--- Fishing Data ---")
        pprint(result[0].text)

        result = await client.call_tool("query_tourism_data", {"polygon_coords": coords})
        print("--- Tourism Data ---")
        pprint(result[0].text)

        result = await client.call_tool("identify_region", {"polygon_coords": coords})
        print("--- Region ---")
        pprint(result[0].text)

        result = await client.call_tool("query_satellite_data", {"variable": "humidity", "polygon_coords": coords, "agg_func": "mean"})
        print("--- Humidity ---")
        pprint(result[0].text)

        result = await client.call_tool("query_satellite_data", {"variable": "temperature", "polygon_coords": coords, "agg_func": "mean"})
        print("--- Temperature ---")
        pprint(result[0].text)

        print("--- Geodesic ---")
        result = await client.call_tool("calculate_geodesic_perimeter", {"polygon_coords": coords})
        result = await client.call_tool("calculate_geodesic_area", {"polygon_coords": coords})
        print(f"- Perimeter: {result[0].text} m")
        print(f"- Area: {result[0].text} m^2")

coords = [
    (-115.20, 31.25),
    (-113.04, 31.70),
    (-107.90, 24.94),
    (-110.21, 23.60),
    (-115.20, 31.25),
]

asyncio.run(test_tools(coords=coords))