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
   uv run main.py
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
     -e ARSAT_API_KEY="your_api_key" \
     -e ARSAT_API_URL="https://api.arsat.com.ar" \
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

## Environment Variables

- `ARSAT_API_KEY`: API key for ARSAT services integration
- `ARSAT_API_URL`: Base URL for ARSAT API (default: https://api.arsat.com.ar)

## Available Tools

1. **check_service_status(service_id)** - Check service operational status
2. **get_account_summary(account_number)** - Retrieve customer account details
3. **create_support_ticket(customer_id, issue_type, description, priority)** - Create support tickets
4. **get_billing_information(account_number)** - Access billing and payment data
5. **schedule_technician_visit(account_number, preferred_date, time_slot, issue_description)** - Schedule service visits
6. **get_service_coverage(address)** - Check service availability by location

## Available Prompts

1. **customer_support_greeting()** - Standard customer service greeting
2. **technical_support_escalation(issue)** - Technical issue escalation template