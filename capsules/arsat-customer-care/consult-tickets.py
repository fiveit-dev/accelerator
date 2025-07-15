from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from loguru import logger

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
async def get_active_tickets(context={}):
    """
    Get active tickets for the current organization

    Returns
    -------
    list
        A list of open tickets
    """
    from dependencies.tickets import get_tickets_provider

    customer_id = context.get("session", {}).get("pluspcustomer", None)
    logger.debug(f"Get active tickets was called! - pluspcustomer: {customer_id}")
    if customer_id:
        tickets_provider = get_tickets_provider()()
        tickets = await tickets_provider.get_active_tickets(customer_id)
        return [t.model_dump() for t in tickets]
    return []


@mcp.tool
async def get_ticket_by_id(ticket_id="", context={}):
    """
    Get a ticket by its ID. Returns None if not found

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

    customer_id = context.get("session", {}).get("pluspcustomer", None)
    logger.debug(
        f"Get ticket by id was called! - ticket_id: {ticket_id} - pluspcustomer: {customer_id}"
    )
    if customer_id:
        tickets_provider = get_tickets_provider()()
        ticket = await tickets_provider.get_ticket_by_id(ticket_id, customer_id)
        if ticket:
            return ticket.model_dump()
    return {}


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/consulting",
        log_level="info",
    )
