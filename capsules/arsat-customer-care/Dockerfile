FROM python:3.12-slim

WORKDIR /app

# Copy only pyproject.toml (so Docker caches the install layer)
COPY pyproject.toml .

# Install your project
RUN pip install --no-cache-dir .

# Copy your entire application
COPY . .

# Expose the MCP server port
EXPOSE 8000

# Run the MCP server
CMD ["python", "consult-tickets.py"]
