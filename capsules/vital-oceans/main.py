from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

mcp = FastMCP("vital-oceans", dependencies=["httpx"])


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


@mcp.tool()
async def get_weather(city: str) -> str:
    """
    Fetch the current weather for a given city.
    """
    # Simulate fetching weather data
    return f"The weather in {city} is sunny with a temperature of 25°C."


if __name__ == "__main__":
    # mcp.run(
    #    transport="streamable-http",
    #    host="0.0.0.0",
    #    port=8000,
    #    path="/mcp",
    #    log_level="info",
    # )
    mcp.run()
