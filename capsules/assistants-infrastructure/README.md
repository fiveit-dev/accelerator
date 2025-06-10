## MCP Project Documentation

This documentation provides a step-by-step guide to set up and run your MCP project.

### Step 1: Build the Dockerfile

Before hosting your MCP server, you need to build the Docker image. Ensure Docker is installed and running on your system. Execute the following command to build the Docker image:

```bash
docker build -t asssistants-infra-mcp . --platform="linux/amd64"
```

### Step 2: Host Your MCP Server

Once the Docker image is built, you can host your MCP server using Docker. Run the following command to start the server:

```bash
docker run -d \
  --name assistants-mcp \
  -p 8000:8000 \
  -e HUGGINGFACE_API_KEY="your_api_key" \
  -e TWYD_API_KEY="your_twyd_api_key" \
  -e TWYD_API_URL="your_twyd_api_url" \
  -e TWYD_TOPIC_ID="your_twyd_topic_id" \
  assistants-infra-mc
```

In case you would like to deploy it as a service in Kubernetes use the script:
**service.yml**

### Step 3: Connect to Claude

In your claude configuration for mcp add this config:

```json
"mcpServers": {
    "assistants-infrastructure":{
      "command": "uv",
            "args": [
                "--directory",
                "full_path/capsules/assistants-infrastructure",
                "run",
                "claude_stdio.py"
      ]
    }
  }
```
