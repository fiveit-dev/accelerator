FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY consult-tickets.py main.py
COPY claude_stdio.py .
COPY config/ config/
COPY dependencies/ dependencies/
COPY models/ models/
COPY mocks/ mocks/

# Install dependencies
RUN uv pip install --system -e .

# Expose the MCP server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the MCP server
CMD ["python", "main.py"]