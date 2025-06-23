from dotenv import load_dotenv
import tempfile
import zipfile
import os
import io
import asyncio
from pathlib import Path

from shapely.geometry import Polygon
from pyproj import Geod
import xarray as xr
import geopandas as gpd
import numpy as np

from lakefs.client import Client
from lakefs import Repository

from fastmcp import FastMCP, Context
from starlette.requests import Request
from starlette.responses import PlainTextResponse

dependencies = [
    "httpx",
    "geopandas",
    "shapely",
    "lakefs",
    "xarray[io]",
    "pyproj",
    "numpy",
    "pandas",
]

mcp = FastMCP(name="vital-oceans", dependencies=dependencies)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

# --- Global Data Storage --- #
PERSISTENT_DATA = {}
DATA_DIR = Path("./mcp_data")

# --- lakeFS Client --- #
load_dotenv()
LAKEFS_HOST = os.environ["LAKEFS_HOST"]
LAKEFS_USERNAME = os.environ["LAKEFS_USERNAME"]
LAKEFS_PASSWORD = os.environ["LAKEFS_PASSWORD"]
LAKEFS_REPO_ID = os.getenv("LAKEFS_REPO_ID", "vital-oceans")

lakefs_client = Client(
    host=LAKEFS_HOST,
    username=LAKEFS_USERNAME,
    password=LAKEFS_PASSWORD,
)

repo = Repository(repository_id=LAKEFS_REPO_ID, client=lakefs_client)
ref = repo.ref("main")

def download_and_cache_shapefile(domain: str) -> Path:
    """
    Downloads shapefile components from LakeFS and saves them to persistent storage.
    Returns the path to the main .shp file.
    """
    path = f"datasets/shapefiles/{domain}"
    domain_dir = DATA_DIR / "shapefiles" / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    # Download all related files
    for obj in ref.objects(prefix=path):
        remote_path = obj.path
        basename = os.path.basename(remote_path)
        local_path = domain_dir / basename
        
        # Skip if file already exists and is not empty
        if local_path.exists() and local_path.stat().st_size > 0:
            continue
            
        with ref.object(remote_path).reader(mode="rb") as r:
            with open(local_path, "wb") as f:
                f.write(r.read())
    
    # Find and return the .shp file path
    shp_files = list(domain_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found for domain: {domain}")
    
    return shp_files[0]

def load_shapefile_to_gdf(domain: str) -> gpd.GeoDataFrame:
    """
    Loads a shapefile from persistent storage into a GeoDataFrame.
    """
    if domain not in PERSISTENT_DATA:
        shp_path = download_and_cache_shapefile(domain)
        gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
        gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]
        PERSISTENT_DATA[domain] = gdf
        print(f"Loaded and cached shapefile for domain: {domain}")
    
    return PERSISTENT_DATA[domain]

def download_and_cache_netcdf(variable: str) -> Path:
    """
    Downloads NetCDF file from LakeFS and saves it to persistent storage.
    Returns the path to the NetCDF file.
    """
    file_path = f"datasets/netcdfs/{variable}.nc"
    netcdf_dir = DATA_DIR / "netcdfs"
    netcdf_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = netcdf_dir / f"{variable}.nc"
    
    # Skip if file already exists and is not empty
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    
    with ref.object(file_path).reader(mode="rb") as r:
        with open(local_path, "wb") as f:
            f.write(r.read())
    
    return local_path

def load_netcdf_data(variable: str) -> dict:
    """
    Loads NetCDF data from persistent storage and processes it.
    """
    cache_key = f"netcdf_{variable}"
    
    if cache_key not in PERSISTENT_DATA:
        netcdf_path = download_and_cache_netcdf(variable)
        
        # Open the dataset from file
        nc = xr.open_dataset(netcdf_path, engine="h5netcdf")
        
        # Process nc data
        lats = nc['lat'].values
        lons = nc['lon'].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:, 0], points[:, 1]))
        
        # Extract attributes
        key = list(nc.data_vars)[0]
        
        processed_data = {
            "description": nc[key].attrs.get("long_name"),
            "start_time": nc.attrs["start_time"],
            "end_time": nc.attrs["end_time"],
            "values": nc[key].values,
            "units": nc[key].attrs.get("units"),
            "key": key,
            "gdf": gdf
        }
        
        nc.close()  # Close the dataset to free memory
        PERSISTENT_DATA[cache_key] = processed_data
        print(f"Loaded and cached NetCDF data for variable: {variable}")
    
    return PERSISTENT_DATA[cache_key]

async def initialize_data():
    """
    Initialize and cache commonly used datasets at startup.
    """
    print("Initializing data cache...")
    
    # Pre-load common shapefiles
    common_shapefiles = ["fishing", "tourism", "marine-ecoregions", "countries"]
    for domain in common_shapefiles:
        try:
            load_shapefile_to_gdf(domain)
        except Exception as e:
            print(f"Warning: Could not pre-load shapefile {domain}: {e}")
    
    # Pre-load common NetCDF variables
    common_variables = ["temperature", "chlorophylla", "humidity", "meridional_wind", "zonal_wind", "sea_level_pressure"]
    for variable in common_variables:
        try:
            load_netcdf_data(variable)
        except Exception as e:
            print(f"Warning: Could not pre-load NetCDF {variable}: {e}")
    
    print("Data cache initialization complete.")

# --- Resource handlers (now simplified) --- #

@mcp.resource("lakefs://shapefiles/{domain}")
def get_shapefile_binary(domain: str, context: Context):
    """
    Returns binary content of a zipped shapefile from persistent storage.
    """
    domain_dir = DATA_DIR / "shapefiles" / domain
    
    if not domain_dir.exists():
        # Fallback: download if not cached
        download_and_cache_shapefile(domain)
    
    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in domain_dir.iterdir():
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    
    zip_buffer.seek(0)
    return zip_buffer.read()

@mcp.resource("lakefs://netcdfs/{variable}")
def get_netcdf_binary(variable: str, context: Context):
    """
    Returns binary content of a NetCDF file from persistent storage.
    """
    netcdf_path = DATA_DIR / "netcdfs" / f"{variable}.nc"
    
    if not netcdf_path.exists():
        # Fallback: download if not cached
        netcdf_path = download_and_cache_netcdf(variable)
    
    with open(netcdf_path, "rb") as f:
        return f.read()

# --- Tools (updated to use cached data) --- #

@mcp.tool()
async def query_fishing_data(polygon_coords: list[tuple[float, float]], context: Context) -> str:
    """
    Retrieves fishing activity records from a geospatial dataset whose geometries intersect a given polygon.
    """
    gdf = load_shapefile_to_gdf("fishing")
    
    # Define polygon
    polygon = Polygon(polygon_coords)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    # Filter rows that intersect the polygon
    matches = gdf[gdf.geometry.intersects(polygon)]
    
    if matches.empty:
        raise ValueError("No records were found within the specified polygon.")
    
    # Filter by columns of interest and re-name them
    cols = {
        'nom_estab': 'Nombre de la Unidad Económica',
        'raz_social': 'Razón social',
        'nombre_act': 'Nombre de clase de la actividad',
        'per_ocu': 'Descripcion estrato personal ocupado',
        'entidad': 'Entidad federativa',
        'municipio': 'Municipio',
        'localidad': 'Localidad',
        'geometry': 'Geometry',
    }
    
    matches = matches[list(cols.keys())]
    matches.columns = list(cols.values())
    
    # Return up to 20 rows
    max_rows = 20
    if len(matches) > max_rows:
        output = matches.head(max_rows).to_csv(index=False)
        output += f"\n... ({len(matches) - max_rows} more rows not shown)"
    else:
        output = matches.to_csv(index=False)
    
    return output

@mcp.tool()
async def query_tourism_data(polygon_coords: list[tuple[float, float]], context: Context) -> str:
    """
    Retrieves tourism activity records from a geospatial dataset whose geometries intersect a given polygon.
    """
    gdf = load_shapefile_to_gdf("tourism")
    
    # Define polygon
    polygon = Polygon(polygon_coords)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    # Filter rows that intersect the polygon
    matches = gdf[gdf.geometry.intersects(polygon)]
    
    if matches.empty:
        raise ValueError("No records were found within the specified polygon.")
    
    # Filter by columns of interest and re-name them
    cols = {
        'nom_estab': 'Nombre de la Unidad Económica',
        'raz_social': 'Razón social',
        'nombre_act': 'Nombre de clase de la actividad',
        'per_ocu': 'Descripcion estrato personal ocupado',
        'entidad': 'Entidad federativa',
        'municipio': 'Municipio',
        'localidad': 'Localidad',
        'geometry': 'Geometry',
    }
    
    matches = matches[list(cols.keys())]
    matches.columns = list(cols.values())
    
    # Return up to 20 rows
    max_rows = 20
    if len(matches) > max_rows:
        output = matches.head(max_rows).to_csv(index=False)
        output += f"\n... ({len(matches) - max_rows} more rows not shown)"
    else:
        output = matches.to_csv(index=False)
    
    return output

@mcp.tool()
async def identify_region(polygon_coords: list[tuple[float, float]], context: Context) -> dict:
    """
    Identifies the marine ecoregion (ECOREGION, PROVINCE, REALM) and country that intersect
    with a given polygon defined by geographic coordinates (lon, lat).
    """
    marine_gdf = load_shapefile_to_gdf("marine-ecoregions")
    country_gdf = load_shapefile_to_gdf("countries")
    
    polygon = Polygon(polygon_coords)
    information = {}
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    # Filter marine regions that intersect
    matches = marine_gdf[marine_gdf.geometry.intersects(polygon)]
    if not matches.empty:
        information.update({
            'ECOREGION': matches.iloc[0]['ECOREGION'],
            'PROVINCE': matches.iloc[0]['PROVINCE'],
            'REALM': matches.iloc[0]['REALM'],
            }
        )
    
    # Filter countries that intersect
    matches = country_gdf[country_gdf.geometry.intersects(polygon)]
    if not matches.empty:
        information.update({'COUNTRY': matches.iloc[0]['NAME']})
    
    if information:
        return information
    else:
        raise ValueError("No region was found within the specified polygon.")

@mcp.tool()
async def query_satellite_data(
    variable: str,
    polygon_coords: list[tuple[float, float]],
    context: Context,
    agg_func='mean') -> dict:
    """
    Extracts satellite data for a specified variable within a geographic polygon
    and applies an aggregation function to the selected values.
    """
    nc_data = load_netcdf_data(variable)
    
    if not nc_data:
        raise ValueError(f"Variable '{variable}' not found.")
    
    allowed_tags = ['mean', 'median', 'sum', 'std']
    if not agg_func in allowed_tags:
        raise ValueError(f"Aggregation function '{agg_func}' is not supported.")
    
    polygon = Polygon(polygon_coords)
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    # Check which points are within the polygon
    gdf = nc_data.get("gdf")
    gdf['in_polygon'] = gdf.geometry.within(polygon)
    inside_indices = gdf[gdf['in_polygon']].index
    
    if inside_indices.empty:
        raise ValueError("No satellite data found within the specified polygon.")
    
    # Mask the satellite data using the indices
    values = nc_data.get("values")
    filtered_values = values.flatten()[inside_indices]
    
    # Apply agg function
    result = getattr(np, "nan"+agg_func)(filtered_values)
    
    output = {
        "description": nc_data.get("description"),
        "start_time": nc_data.get("start_time"),
        "end_time": nc_data.get("end_time"),
        "units": nc_data.get("units"),
        agg_func: result
    }
    
    # Check for NaNs
    nan_values = sum(np.isnan(filtered_values))
    total_values = len(filtered_values)
    if nan_values:
        output["nan_warning"] = f"{nan_values} nan values out of {total_values} records"
    
    return output

@mcp.tool()
async def calculate_geodesic_area(
    polygon_coords: list[tuple[float, float]],
    context: Context) -> float:
    """
    Calculate the geodesic area (in square meters) of a polygon defined by (lon, lat) coordinates.
    """
    geod = Geod(ellps="WGS84")
    polygon = Polygon(polygon_coords)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    area, _ = geod.geometry_area_perimeter(polygon)
    return round(abs(area), 2)

@mcp.tool()
async def calculate_geodesic_perimeter(
    polygon_coords: list[tuple[float, float]],
    context: Context) -> float:
    """
    Calculate the geodesic perimeter (in meters) of a polygon defined by (lon, lat) coordinates.
    """
    geod = Geod(ellps="WGS84")
    polygon = Polygon(polygon_coords)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")
    
    _, perimeter = geod.geometry_area_perimeter(polygon)
    return round(perimeter, 2)

if __name__ == "__main__":
    # Initialize data cache before starting the server
    asyncio.run(initialize_data())
    
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
        log_level="info",
    )