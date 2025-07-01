import asyncio
from fastmcp import Client
from pprint import pprint

client = Client("main.py")

async def test_tools(coords):
    async with client:

        results = []

        print("--- Fishing Data ---")
        result = await client.call_tool("query_activity_data", {"activity": "fishing", "polygon_coords": coords})
        pprint(result[0].text)

        print("--- Tourism Data ---")
        result = await client.call_tool("query_activity_data", {"activity": "tourism", "polygon_coords": coords})
        pprint(result[0].text)
        print(type(result[0].text))

        print("--- Region ---")
        result = await client.call_tool("identify_region", {"polygon_coords": coords})
        pprint(result[0].text)
        print(type(result[0].text))

        print("--- Humidity ---")
        result = await client.call_tool("query_satellite_data", {"variable": "humidity", "polygon_coords": coords, "agg_func": "mean"})
        pprint(result[0].text)
        print(type(result[0].text))

        print("--- Temperature ---")
        result = await client.call_tool("query_satellite_data", {"variable": "temperature", "polygon_coords": coords, "agg_func": "std"})
        pprint(result[0].text)

        print("--- Chlorophylla ---")
        result = await client.call_tool("query_satellite_data", {"variable": "chlorophylla", "polygon_coords": coords, "agg_func": "median"})
        pprint(result[0].text)

        print("--- Geodesic ---")
        result = await client.call_tool("calculate_geodesic_perimeter", {"polygon_coords": coords})
        print(f"- Perimeter: {result[0].text}")
        print(type(result[0].text))
        
        result = await client.call_tool("calculate_geodesic_area", {"polygon_coords": coords})
        print(f"- Area: {result[0].text}")
        print(type(result[0].text))

# Polygon coords from Baja California
coords = [
    {
        "lat": 31.25,
        "lng": -115.20
    },
    {
        "lat": 31.70,
        "lng": -113.04
    },
    {
        "lat": 24.94,
        "lng": -107.90
    },
    {
        "lat": 23.60,
        "lng": -110.21
    },
    {
        "lat": 31.25,
        "lng": -115.20
    }
]
asyncio.run(test_tools(coords=coords))