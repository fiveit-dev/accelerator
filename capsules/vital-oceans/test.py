import asyncio
from fastmcp import Client
from pprint import pprint

client = Client("main.py") # http://0.0.0.0:8000/mcp

async def test_tools(coords):
    async with client:

        results = []

        print("\n\n--- Shapefile Data ---")
        result = await client.call_tool("show_available_datasets", {"data_type": "shapefile"})
        pprint(result[0].text)

        print("- AMP (Mexico)")
        result = await client.call_tool("query_shapefile", {"shapefile_path": "areas_prioritarias/areas_protegidas/mexico/anp-mexico.shp", "polygon_coords": coords})
        pprint(result[0].text)

        print("- AMP (EEUU)")
        result = await client.call_tool("query_shapefile", {"shapefile_path": "areas_prioritarias/areas_protegidas/america_norte/PCA_Baja_to_Bering_2005.shp", "polygon_coords": coords})
        pprint(result[0].text)

        print("- Biodiversidad:")
        result = await client.call_tool("query_shapefile", {"shapefile_path": "areas_prioritarias/zonas_importancia_biodiversidad/zona_importancia_Biodiv_CDB.shp", "polygon_coords": coords})
        pprint(result[0].text)

        print("- Biología:")
        result = await client.call_tool("query_shapefile", {"shapefile_path": "areas_prioritarias/zonas_importancia_biológica/zona_importancia_biol_pp.shp", "polygon_coords": coords})
        pprint(result[0].text)

        print("- Refugios Pesqueros:")
        result = await client.call_tool("query_shapefile", {"shapefile_path": "areas_prioritarias/zonas_refugios_pesqueros/zona_refugio_pesquero_COBI2020.shp", "polygon_coords": coords})
        pprint(result[0].text)

        print("- Turismo:")
        result = await client.call_tool("query_shapefile", {"shapefile_path": "socioeconomico/turismo/INEGI_DENUE_26052025.shp", "polygon_coords": coords})
        pprint(result[0].text)

        print("- Pesca:")
        result = await client.call_tool("query_shapefile", {"shapefile_path": "socioeconomico/pesca/INEGI_DENUE_26052025.shp", "polygon_coords": coords})
        pprint(result[0].text)

        print("- Zonas explotación (calamar):")
        result = await client.call_tool("query_shapefile", {"shapefile_path": "socioeconomico/zonas_explotacion_pesquera/calamar/zona_pesca_calamar.shp", "polygon_coords": coords})
        pprint(result[0].text)

        print("\n\n--- NetCDF Data ---")
        result = await client.call_tool("show_available_datasets", {"data_type": "netcdf"})
        pprint(result[0].text)

        print("- Surface wind speed:")
        result = await client.call_tool("query_netcdf", {"netcdf_path": "surface_wind_speed.nc", "polygon_coords": coords, "agg_func": "max"})
        pprint(result[0].text)

        print("- Temperature:")
        result = await client.call_tool("query_netcdf", {"netcdf_path": "humidity.nc", "polygon_coords": coords, "agg_func": "min"})
        pprint(result[0].text)

        print("- Temperature:")
        result = await client.call_tool("query_netcdf", {"netcdf_path": "temperature.nc", "polygon_coords": coords, "agg_func": "series"})
        pprint(result[0].text)

        print("\n\n--- CSV Data ---")
        result = await client.call_tool("show_available_datasets", {"data_type": "csv"})
        pprint(result[0].text)

        print("- Schema:")
        result = await client.call_tool("get_csv_schema", {"csv_path": "tablas_biodiversidad/peces_bahia_lapaz.csv"})
        pprint(result[0].text)

        print("- Query:")
        result = await client.call_tool("query_csv", {"csv_path": "tablas_biodiversidad/peces_bahia_lapaz.csv", "sql_query": "SELECT * FROM peces_bahia_lapaz"})
        pprint(result[0].text)

        print("\n\n--- Geodesic ---")
        result = await client.call_tool("calculate_geodesic_perimeter", {"polygon_coords": coords})
        print(f"- Perimeter: {result[0].text}")
        
        result = await client.call_tool("calculate_geodesic_area", {"polygon_coords": coords})
        print(f"- Area: {result[0].text}")

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