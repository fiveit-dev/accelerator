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
   docker build -t alquimiaai/arsat-customer --platform=linux/amd64 .
   ```

2. **Deploy using service.yaml**:

   ```bash
   kubectl apply -f deploy.yaml
   ```

3. **Configuration**:
   - **For mock data**: Do not specify the `MAXIMO_BASE_URL` environment variable
   - **For production**: Set `MAXIMO_BASE_URL` to the MAXIMO production URL
