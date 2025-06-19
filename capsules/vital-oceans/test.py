import asyncio
from fastmcp import Client

client = Client("main.py")

async def test_tools(coords):
    async with client:

        results = []

        result = await client.call_tool("query_fishing_data", {"polygon_coords": coords})
        print(result)

        result = await client.call_tool("query_tourism_data", {"polygon_coords": coords})
        print(result)

        result = await client.call_tool("identify_region", {"polygon_coords": coords})
        print(result)

        result = await client.call_tool("query_satellite_data", {"variable": "humidity", "polygon_coords": coords, "agg_func": "mean"})
        print(result)

        result = await client.call_tool("calculate_geodesic_area", {"polygon_coords": coords})
        print(result)

        result = await client.call_tool("calculate_geodesic_perimeter", {"polygon_coords": coords})
        print(result)

coords = [
    (-115.20263671875001, 31.25595466608565),
    (-113.04931640625001, 31.705679333666268),
    (-107.90771484375001, 24.942172210041733),
    (-110.21484375, 23.60017200992539),
    (-115.20263671875001, 31.25595466608565),
]

asyncio.run(test_tools(coords=coords))