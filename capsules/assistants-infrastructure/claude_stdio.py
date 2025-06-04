from fastmcp import FastMCP

# Create a proxy to a remote server
proxy = FastMCP.as_proxy("http://localhost:8000/mcp", name="Remote Server Proxy")

if __name__ == "__main__":
    proxy.run()  # Runs via STDIO for Claude Desktop
