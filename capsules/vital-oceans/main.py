from dotenv import load_dotenv
import tempfile
import zipfile
import os
import io

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

# --- lakeFS Client --- #
load_dotenv()
LAKEFS_HOST = os.environ["LAKEFS_HOST"]
LAKEFS_USERNAME = os.environ["LAKEFS_USERNAME"]
LAKEFS_PASSWORD = os.environ["LAKEFS_PASSWORD"]

lakefs_client = Client(
    host=LAKEFS_HOST,
    username=LAKEFS_USERNAME,
    password=LAKEFS_PASSWORD,
)

repo = Repository(repository_id="vital-oceans", client=lakefs_client)
ref = repo.ref("main")

## Shapefiles ##

@mcp.resource("lakefs://shapefiles/{domain}")
def get_shapefile_binary(domain: str, context: Context):
    """
    Downloads shapefile components from a LakeFS path, zips them, and returns binary content.
    """
    path = f"datasets/shapefiles/{domain}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download all related files to the temp directory
        for obj in ref.objects(prefix=path):
            remote_path = obj.path
            basename = os.path.basename(remote_path)
            local_path = os.path.join(tmpdir, basename)

            with ref.object(remote_path).reader(mode="rb") as r:
                with open(local_path, "wb") as f:
                    f.write(r.read())

        # Create in-memory zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in os.listdir(tmpdir):
                file_path = os.path.join(tmpdir, file)
                zf.write(file_path, arcname=file)

        # Return binary content
        zip_buffer.seek(0)
        return zip_buffer.read()

@mcp.resource("lakefs://shapefiles/{domain}")
def get_shapefile(domain: str, context: Context):
    """
    Downloads shapefile components from a LakeFS path, zips them, and returns binary content.
    """
    path = f"datasets/shapefiles/{domain}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download all related files to the temp directory
        for obj in ref.objects(prefix=path):
            remote_path = obj.path
            basename = os.path.basename(remote_path)
            local_path = os.path.join(tmpdir, basename)

            with ref.object(remote_path).reader(mode="rb") as r:
                with open(local_path, "wb") as f:
                    f.write(r.read())

        # Create in-memory zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in os.listdir(tmpdir):
                file_path = os.path.join(tmpdir, file)
                zf.write(file_path, arcname=file)

        # Return binary content
        zip_buffer.seek(0)
        return zip_buffer.read()

def binary_to_gdf(binary_zip):
    """
    Reads a binary ZIP containing shapefile components into a GeoDataFrame.
    
    Parameters:
        binary_zip (bytes): Binary content of the zipped shapefile.

    Returns:
        gpd.GeoDataFrame: The loaded GeoDataFrame.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract the binary zip content to the temp directory
        with zipfile.ZipFile(io.BytesIO(binary_zip)) as zf:
            zf.extractall(tmpdir)

        # Find the .shp file in the extracted contents
        for file in os.listdir(tmpdir):
            if file.endswith(".shp"):
                shapefile_path = os.path.join(tmpdir, file)
                break
        else:
            raise FileNotFoundError("No .shp file found in the provided binary zip.")

        # Load and return the GeoDataFrame
        gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
        gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]

        return gdf

@mcp.tool()
async def query_fishing_data(polygon_coords: list[tuple[float, float]], context: Context) -> str:
    """
    Retrieves fishing activity records from a geospatial dataset whose geometries intersect a given polygon.

    Parameters:
    -----------
    polygon_coords : list of tuple of float
        A list of (longitude, latitude) tuples defining the vertices of the polygon to use as a spatial filter.

    Returns:
    --------
    str
        A CSV-formatted string with matching records.
    """

    content_list = await context.read_resource("lakefs://shapefiles/fishing")    
    gdf = binary_to_gdf(content_list[0].content)

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

    Parameters:
    -----------
    polygon_coords : list of tuple of float
        A list of (longitude, latitude) tuples defining the vertices of the polygon to use as a spatial filter.

    Returns:
    --------
    str
        A CSV-formatted string with matching records.
    """

    content_list = await context.read_resource("lakefs://shapefiles/tourism")
    gdf = binary_to_gdf(content_list[0].content)

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

    Parameters:
    -----------
    polygon_coords : list of tuple of float
        A list of (longitude, latitude) tuples representing the vertices of the polygon.

    Returns:
    --------
    dict
    """

    content_list = await context.read_resource("lakefs://shapefiles/marine-ecoregions")
    marine_gdf = binary_to_gdf(content_list[0].content)

    content_list = await context.read_resource("lakefs://shapefiles/countries")
    country_gdf = binary_to_gdf(content_list[0].content)

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

## NetCDFs ##
@mcp.resource("lakefs://netcdfs/{variable}")
def get_netcdf_binary(variable: str, context: Context):

    file_path = f"datasets/netcdfs/{variable}.nc"

    # Extract raw bytes
    with ref.object(file_path).reader(mode="rb") as r:
        b = r.read()

    return b

def process_netcdf(netcdf_binary):

    # Convert raw bytes into a BytesIO buffer
    buffer = io.BytesIO(netcdf_binary)

    # Open the dataset from the buffer
    nc = xr.open_dataset(buffer, engine="h5netcdf")

    # Process nc data
    lats = nc['lat'].values
    lons = nc['lon'].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:, 0], points[:, 1]))

    # Extract attributes
    key = list(nc.data_vars)[0]

    return {
        "description": nc[key].attrs.get("long_name"),
        "start_time": nc.attrs["start_time"],
        "end_time": nc.attrs["end_time"],
        "values": nc[key].values,
        "units": nc[key].attrs.get("units"),
        "key": key,
        "gdf": gdf
    }

@mcp.tool()
async def query_satellite_data(
    variable: str,
    polygon_coords: list[tuple[float, float]],
    context: Context,
    agg_func='mean') -> dict:
    """
    Extracts satellite data for a specified variable within a geographic polygon
    and applies an aggregation function to the selected values.

    Parameters
    ----------
    variable : str
        Name of the variable to extract.
        Supported options are: 'temperature', 'chlorophylla', 'humidity', 'meridional_wind', 'zonal_wind' and 'sea_level_pressure'

    polygon_coords : list of tuples
        List of (lon, lat) coordinates defining the polygon.

    agg_func : str
        Aggregation function to apply.
        Supported options are: 'mean', 'median', 'sum', 'std'.

    Returns
    -------
    dict
        Dict containing the aggregated value of the variable within the polygon.
    """

    content_list = await context.read_resource(f"lakefs://netcdfs/{variable}")
    nc_data = process_netcdf(content_list[0].content)

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

## Geo Spatial Analysis ##
@mcp.tool()
async def calculate_geodesic_area(
    polygon_coords: list[tuple[float, float]],
    context: Context) -> float:
    """
    Calculate the geodesic area (in square meters) of a polygon defined by (lon, lat) coordinates.

    Parameters
    ----------
    coords : list of tuples
        List of (longitude, latitude) pairs.

    Returns
    -------
    float
        Area in square meters.
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

    Parameters
    ----------
    coords : list of tuples)
        List of (longitude, latitude) pairs.

    Returns
    -------
    float
        Perimeter in meters.
    """
    geod = Geod(ellps="WGS84")
    polygon = Polygon(polygon_coords)

    if not polygon.is_valid:
        raise ValueError("Invalid polygon. Ensure the coordinates form a valid polygon.")

    _, perimeter = geod.geometry_area_perimeter(polygon)
    return round(perimeter, 2)

if __name__ == "__main__":
    mcp.run(
        #transport="streamable-http",
        #host="0.0.0.0",
        #port=8000,
        #path="/mcp",
        #log_level="info",
    )