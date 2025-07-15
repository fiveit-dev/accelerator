# ARSAT Customer Care MCP Server

This capsule provides an MCP (Model Context Protocol) server for ARSAT customer care operations, offering tools and prompts for customer support, technical assistance, and service management.

## Features

### Tools

- **Service Status Checking**: Monitor ARSAT service health and uptime
- **Account Management**: Retrieve customer account summaries and billing information
- **Support Ticket Creation**: Create and track customer support tickets
- **Technician Scheduling**: Schedule service visits and repairs
- **Coverage Checking**: Verify service availability for specific addresses

### Prompts

- **Customer Support Greeting**: Standard greeting for customer interactions
- **Technical Support Escalation**: Template for handling technical issues

## Quick Start

### Local Development

1. **Install dependencies**:

   ```bash
   cd capsules/arsat-customer-care
   uv sync
   ```

2. **Run the MCP server**:

   ```bash
   uv run consult-tickets.py
   ```

3. **Test the server** (optional):
   ```bash
   curl http://localhost:8000/health
   ```

### Docker Deployment

1. **Build the Docker image**:

   ```bash
   docker build -t arsat-customer-care-mcp . --platform="linux/amd64"
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name arsat-customer-care \
     -p 8000:8000 \
     -e MAXIMO_BASE_URL="https://your-maximo-url.com" \
     -e MAXIMO_USER_ID="your_user_id" \
     -e MAXIMO_PASSWD="your_password" \
     -e MAXIMO_REQUEST_TIMEOUT="10.0" \
     -e MAXIMO_HTTP_VERIFY_SSL="true" \
     -e MAXIMO_OPEN_TICKET_STATUSES="OPEN,IN_PROGRESS,PENDING" \
     arsat-customer-care-mcp
   ```

### Claude Desktop Integration

Add this configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arsat-customer-care": {
      "command": "uv",
      "args": [
        "--directory",
        "/full/path/to/capsules/arsat-customer-care",
        "run",
        "claude_stdio.py"
      ]
    }
  }
}
```
