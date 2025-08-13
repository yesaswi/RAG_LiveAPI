import os
import asyncio
from typing import List

import vertexai
from vertexai.preview import rag
from google import genai
from google.genai import types

# ----------- Config ----------
PROJECT_ID = os.getenv("PROJECT_ID", "vet-vocals")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_ID = os.getenv("MODEL_ID", "gemini-2.0-flash-live-001")
EMBED_MODEL = os.getenv("EMBED_MODEL", "publishers/google/models/text-embedding-005")
LLM_PARSER = os.getenv(
    "LLM_PARSER",
    f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-2.0-flash",
)
KB_DISPLAY = os.getenv("KB_DISPLAY", "kb-corpus")
MEM_DISPLAY = os.getenv("MEM_DISPLAY", "live-session-memory")
KB_GCS_URIS = [p.strip() for p in os.getenv("KB_GCS_URIS", "").split(",") if p.strip()]


# ----------- Helpers ----------
def init_vertex():
    vertexai.init(project=PROJECT_ID, location=LOCATION)


def get_corpus_by_display(display_name: str):
    for c in rag.list_corpora():  # <--- iterate the pager
        if c.display_name == display_name:
            return c
    return None


def ensure_kb_corpus():
    c = get_corpus_by_display(KB_DISPLAY)
    if c:
        print(f"[✓] KB corpus exists: {c.name}")
        return c
    print("[*] Creating KB (document) corpus...")
    created = rag.create_corpus(
        display_name=KB_DISPLAY,
        # default corpus type is DocumentCorpus
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model=EMBED_MODEL
                )
            )
        ),
    )
    print(f"[✓] KB corpus created: {created.name}")
    return created


def ensure_memory_corpus():
    c = get_corpus_by_display(MEM_DISPLAY)
    if c:
        print(f"[✓] Memory corpus exists: {c.name}")
        return c
    print("[*] Creating MemoryCorpus for Live session context...")
    created = rag.create_corpus(
        display_name=MEM_DISPLAY,
        corpus_type_config=rag.RagCorpusTypeConfig(
            corpus_type_config=rag.MemoryCorpus(
                llm_parser=rag.LlmParserConfig(model_name=LLM_PARSER)
            )
        ),
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model=EMBED_MODEL
                )
            )
        ),
    )
    print(f"[✓] Memory corpus created: {created.name}")
    return created


def import_kb_files(kb_corpus_name: str, paths: List[str]):
    if not paths:
        print("[i] No KB_GCS_URIS provided — skipping KB import.")
        return
    print(f"[*] Importing KB files from: {paths}")
    resp = rag.import_files(
        corpus_name=kb_corpus_name,
        paths=paths,
        # tweak as desired:
        chunk_size=512,
        chunk_overlap=64,
        # You can also pass RagFileParsingConfig if needed.
        # max_embedding_requests_per_min defaults to 1000.
    )
    print("[✓] Import started. Tip: large imports index asynchronously.")


async def live_chat(kb_corpus_name: str, mem_corpus_name: str):
    # Configure two VertexRagStore tools: one KB corpus, one Memory corpus (with store_context=True)
    kb_store = types.VertexRagStore(
        rag_resources=[types.VertexRagStoreRagResource(rag_corpus=kb_corpus_name)]
    )
    memory_store = types.VertexRagStore(
        rag_resources=[types.VertexRagStoreRagResource(rag_corpus=mem_corpus_name)],
        store_context=True,  # store chat turns into MemoryCorpus
    )

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    print("\n[Live] Connecting… Type your message and press Enter. Ctrl+C to exit.\n")
    async with client.aio.live.connect(
        model=MODEL_ID,
        config=types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            tools=[
                types.Tool(retrieval=types.Retrieval(vertex_rag_store=kb_store)),
                types.Tool(retrieval=types.Retrieval(vertex_rag_store=memory_store)),
            ],
        ),
    ) as session:

        async def receiver():
            async for msg in session.receive():
                if msg.text:
                    print(f"\n[Gemini]: {msg.text}\n", flush=True)

        recv_task = asyncio.create_task(receiver())

        # Seed message (optional)
        await session.send_client_content(
            turns=types.Content(
                role="user", parts=[types.Part(text="Hi! What can you do?")]
            )
        )

        loop = asyncio.get_event_loop()
        while True:
            try:
                user_text = await loop.run_in_executor(None, input, "[You]: ")
            except (EOFError, KeyboardInterrupt):
                print("\n[Live] Closing session…")
                break
            if not user_text.strip():
                continue
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=user_text)])
            )

        recv_task.cancel()
        with contextlib.suppress(Exception):
            await recv_task


def main():
    init_vertex()
    kb = ensure_kb_corpus()
    mem = ensure_memory_corpus()

    # (Optional) import files into KB corpus
    import_kb_files(kb.name, KB_GCS_URIS)

    # Kick off Live chat
    asyncio.run(live_chat(kb.name, mem.name))


if __name__ == "__main__":
    import contextlib

    main()
