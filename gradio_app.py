from __future__ import annotations

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
        return "Please enter a query.", ""

    params = {
        "query": query.strip(),
        "top_n": int(top_n),
        "mode": mode,
        "use_reranker": use_reranker,
    }

    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.get(QUERY_URL, params=params)
            response.raise_for_status()

            # Prefer JSON; fall back to raw text
            answer = ""
            sources_md = "_No source chunks._"

            try:
                data = response.json()
            except Exception:
                data = None

            if isinstance(data, str):
                # FastAPI returned a plain string answer
                answer = data
            elif isinstance(data, dict):
                answer = data.get("answer") or data.get("result") or str(data)
                sources = data.get("source_nodes") or data.get("sources") or []
                if sources:
                    lines = []
                    for i, s in enumerate(sources, 1):
                        score = s.get("score") or s.get("rerank_score") or ""
                        text = (s.get("text") or "")[:400]
                        lines.append(f"**{i}.** score={score}\n\n{text}\n")
                    sources_md = "\n".join(lines)
            else:
                # Not JSON (or unexpected shape) → use body as answer
                answer = response.text.strip()

            if not answer:
                answer = f"(empty response)\nstatus={response.status_code}\nbody={response.text[:500]!r}"

            return answer, sources_md

    except httpx.ConnectError:
        return (
            f"Cannot reach API at {API_BASE}. "
            "Start it with: uvicorn app.main:app --host 0.0.0.0 --port 8000",
            "",
        )
    except httpx.HTTPStatusError as exc:
        return f"API error {exc.response.status_code}: {exc.response.text}", ""
    except Exception as exc:
        return f"Request failed: {type(exc).__name__}: {exc}", ""


with gr.Blocks(title="LF Jobs RAG") as demo:
    gr.Markdown("# LF Jobs RAG Search")
    gr.Markdown(f"Hits **GET `{QUERY_URL}`**.")

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

    answer = gr.Textbox(label="Answer", lines=10)
    sources = gr.Markdown(label="Sources")

    inputs = [query, top_n, mode, use_reranker]
    outputs = [answer, sources]

    search_btn.click(fn=run_search, inputs=inputs, outputs=outputs)
    query.submit(fn=run_search, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch()