import httpx
from fastmcp import FastMCP
import os
from mcp.server.fastmcp.prompts import base
from transformers import PretrainedConfig
from dataclasses import dataclass
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# Initialize FastMCP server
mcp = FastMCP("assistants-infraestructure")


mcp = FastMCP(
    "assistants-infraestructure",
    dependencies=[
        "httpx",
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "beautifulsoup4",
        "transformers",
    ],
)
HUGGINGFACE_ENDPOINT = "https://huggingface.co"
HUGGINGFACE_API_KEY = os.environ.get(
    "HUGGINGFACE_API_KEY", "your_huggingface_api_key_here"
)
DTYPE_SIZES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}
MODEL_TENSORS_BYTES = DTYPE_SIZES["fp16"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TWYD_API_KEY = os.environ.get("TWYD_API_KEY", "your_twyd_api_key_here")
TWYD_API_URL = os.environ.get(
    "TWYD_API_URL", "twyd-alquimia-twyd.apps.alquimiaai.hostmydemo.online"
)
TWYD_TOPIC_ID = os.environ.get("TWYD_TOPIC_ID", "42")

os.environ["HF_TOKEN"] = HUGGINGFACE_API_KEY


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]


@mcp.tool()
async def get_specific_model_insights(model_name: str):
    """
    Fetches information about a Hugging Face model.

    Useful to get the model_id for further operations.

    Args:
        model_name (str): The name of the model to fetch information for.

    Returns:
        dict: A dictionary containing model information.
    """
    url = f"{HUGGINGFACE_ENDPOINT}/api/models?search={model_name}&sort=likes&limit=5&full=true&config=true"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_models() -> str:
    """
    Fetches a list of popular models.
    """
    url = "https://lmarena.ai/leaderboard/text"
    headers = {
        # Typical Chrome User-Agent (can be swapped for any modern browser UA)
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    if table is None:
        raise RuntimeError(
            "Could not find any <table> tag on LMArena Text leaderboard page."
        )

    header_cells = table.find("thead").find_all("th")
    headers = [th.get_text(strip=True) for th in header_cells]

    body_rows = table.find("tbody").find_all("tr")
    rows: list[list[str]] = []
    md = []
    for tr in body_rows:
        # Some rows might have nested tags—extract text only
        cell_texts = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cell_texts:
            rows.append(cell_texts)

        md = []

    md.append("| " + " | ".join(headers) + " |")

    md.append("| " + " | ".join("---" for _ in headers) + " |")

    for row in rows:
        md.append("| " + " | ".join(row) + " |")

    return "\n".join(md)


@mcp.tool()
async def get_hardware_requirements(
    model_id: str,
    model_size: int,
    desired_context_window: int = 20000,
):
    """
    Fetches the hardware requirements for a specific Hugging Face model.

    Args:
        model_id (str): The ID of the model to fetch hardware requirements for.
        model_size (int): The size of the model in billions of parameters.
        desired_context_window (int): The desired context window size for the model.

    Returns:
        dict: A dictionary containing hardware requirements.
    """

    try:
        config_dict = PretrainedConfig.from_pretrained(model_id).to_dict()
    except Exception as e:
        print("Error loading model configuration:", e)
        raise RuntimeError("Failed to load model config; cannot proceed.")

    hidden_size = (
        config_dict["text_config"]["hidden_size"]
        if "text_config" in config_dict
        else config_dict["hidden_size"]
    )

    hidden_layers = (
        config_dict["text_config"]["num_hidden_layers"]
        if "text_config" in config_dict
        else config_dict["num_hidden_layers"]
    )

    total_params = model_size * (10**9)
    taken_space_parameters = total_params * MODEL_TENSORS_BYTES
    rounded_taken_space_parameters = round(taken_space_parameters / 2**30, 3)
    kv_bytes_per_token = hidden_layers * (4 * hidden_size)
    desired_tokens = desired_context_window
    kv_required_bytes = desired_tokens * kv_bytes_per_token
    kv_required_gib = round(kv_required_bytes / 2**30, 3)

    total_needed_gib = round(rounded_taken_space_parameters + kv_required_gib, 3)

    parameters_loading = f"{model_id} requires {rounded_taken_space_parameters} GiB for parameters and {kv_required_gib} GiB for KV cache f({desired_context_window} Context Window), totaling {total_needed_gib} GiB."

    return parameters_loading


@mcp.tool()
async def get_insights_use_case(query: str) -> List[Dict[str, Any]]:
    """
    Provides insights on a use case based on the query.

    Args:
        query (str): The query to analyze for insights.

    Returns:
        list[base.Message]: A list of messages containing insights.
    """
    payload = {"query": query, "k": 2}
    headers = {
        "Authorization": f"Bearer {TWYD_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"https://{TWYD_API_URL}/api/topics/{TWYD_TOPIC_ID}/search",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        results = resp.json()

    insights: List[Dict[str, Any]] = []
    for item in results:
        page_content = item.get("pageContent", "")
        metadata = item.get("metadata", {})
        insights.append({"pageContent": page_content, "metadata": metadata})
    return insights


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
        log_level="info",
    )
