from __future__ import annotations

import json
import os

import gradio as gr
import httpx

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
QUERY_URL = f"{API_BASE}/api/v1/query"
DEFAULT_TOP_N = 30


def run_search(
    query: str,
    top_n: int,
    mode: str,
    use_reranker: bool,
):
    if not query or not query.strip():
        yield "Please enter a query.", ""
        return

    payload = {
        "query": query.strip(),
        "top_n": int(top_n),
        "mode": mode,
        "use_reranker": use_reranker,
    }

    answer = ""
    sources_md = "_No source chunks._"

    try:
        with httpx.Client(timeout=120.0) as client:
            # Use client.stream for SSE (GET request with params)
            with client.stream("GET", QUERY_URL, params=payload) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        data_content = line[5:].strip()
                        if not data_content:
                            continue
                        
                        # Accumulate text chunks as they stream in
                        answer += data_content
                        yield answer, sources_md

    except httpx.ConnectError:
        yield (
            f"Cannot reach API at {API_BASE}. "
            "Start it with: uvicorn app.main:app --host 0.0.0.0 --port 8000",
            "",
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        yield f"API error {exc.response.status_code}: {detail}", ""
    except Exception as exc:
        yield f"Request failed: {exc}", ""


with gr.Blocks(title="LF Jobs RAG") as demo:
    gr.Markdown("# LF Jobs RAG Search (Streaming)")
    gr.Markdown(
        f"UI streaming enabled — hits **GET `{QUERY_URL}`**. "
        "Start FastAPI before using this app."
    )

    with gr.Row():
        query = gr.Textbox(
            label="Query",
            placeholder="e.g. senior remote python engineer",
            scale=4,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)

    with gr.Row():
        top_n = gr.Slider(1, 50, value=DEFAULT_TOP_N, step=1, label="Top N")
        mode = gr.Dropdown(
            choices=["hybrid", "dense", "sparse"],
            value="hybrid",
            label="Retrieval mode",
        )
        use_reranker = gr.Checkbox(value=True, label="Reranker")

    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Markdown(label="Sources")

    inputs = [query, top_n, mode, use_reranker]
    outputs = [answer, sources]

    search_btn.click(fn=run_search, inputs=inputs, outputs=outputs)
    query.submit(fn=run_search, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch()