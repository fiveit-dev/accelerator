from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.context import Context
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from loguru import logger
import json
from fastmcp.server.dependencies import get_http_headers

# Initialize FastMCP server
mcp = FastMCP(
    "arsat-customer-care",
    dependencies=[
        "httpx",
        "pydantic",
    ],
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Health check endpoint for the MCP server."""
    return PlainTextResponse("OK")


@mcp.tool
async def get_active_tickets(context: Context):
    """
    Get active tickets for the current organization

    Returns
    -------
    list
        A list of open tickets
    """
    from dependencies.tickets import get_tickets_provider

    headers = get_http_headers()
    context_data = headers.get("context", {})
    if isinstance(context_data, str):
        try:
            context_data = json.loads(context_data)
        except json.JSONDecodeError:
            await context.error("Failed to decode context from JSON string")
            raise ValueError("Invalid context format")

    customer_id = context_data.get("session", {}).get("pluspcustomer", None)

    if not customer_id:
        await context.error("Not found customer_id")
        raise ValueError("Customer ID not found in context")

    logger.debug(f"Get active tickets was called! - pluspcustomer: {customer_id}")
    tickets_provider = get_tickets_provider()()
    tickets = await tickets_provider.get_active_tickets(customer_id)
    return [t.model_dump() for t in tickets]


@mcp.tool
async def get_ticket_by_id(ticket_id: str, context: Context):
    """
    Get a ticket by its ID. Returns {} if not found

    Parameters
    ----------
    ticket_id : str
        The ID of the ticket, e.g., SS-575460

    Returns
    -------
    dict
        The ticket associated with the ID. Returns an empty dict if not found
    """
    from dependencies.tickets import get_tickets_provider

    headers = get_http_headers()
    context_data = headers.get("context", {})
    if isinstance(context_data, str):
        try:
            context_data = json.loads(context_data)
        except json.JSONDecodeError:
            await context.error("Failed to decode context from JSON string")
            raise ValueError("Invalid context format")

    customer_id = context_data.get("session", {}).get("pluspcustomer", None)

    if not customer_id:
        await context.error("Not found customer_id")
        raise ValueError("Customer ID not found in context")

    await context.debug(
        f"Get ticket by id was called! - ticket_id: {ticket_id} - pluspcustomer: {customer_id}"
    )

    tickets_provider = get_tickets_provider()()
    ticket = await tickets_provider.get_ticket_by_id(ticket_id, customer_id)
    if ticket:
        return ticket.model_dump()
    return {}


@mcp.tool
async def get_classification(query: str, Context: Context):
    """
    Runs a clasification query agains a text classification model.
    Parameters
    ----------
    query : str
        The text to classify.

    Returns
    -------
    dict
        The classification result, e.g., {"category": "technical", "confidence": 0.95}
    """
    ## TODO: Implement with alquimia core


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/consulting",
        log_level="info",
    )
