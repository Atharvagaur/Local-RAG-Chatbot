"""Streamlit frontend for the Document Intelligence AI RAG chatbot."""

from __future__ import annotations

import logging
import time
from typing import Optional

import streamlit as st

from backend.chat_manager import ChatManager
from backend.document_processor import compute_file_hash, process_document
from backend.embeddings import build_embedding_manager
from backend.ollama_manager import build_ollama_manager, OllamaError
from backend.rag_pipeline import DEFAULT_SYSTEM_PROMPT, RagPipeline
from backend.vector_store import VectorStore
from config import settings

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - dependency missing
    ChatOllama = None  # type: ignore

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .app-title { font-size: 1.75rem; font-weight: 700; margin: 0; }
    .app-subtitle { color: #6b7280; font-size: 0.95rem; margin-top: 0.25rem; margin-bottom: 1rem; }
    .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; background: #ffffff; }
    .doc-item { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; border-bottom: 1px solid #f3f4f6; font-size: 0.9rem; }
    .doc-item:last-child { border-bottom: none; }
    .badge { display: inline-block; padding: 0.12rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-ok { background: #dcfce7; color: #166534; }
    .badge-warn { background: #fef3c7; color: #92400e; }
    .badge-err { background: #fee2e2; color: #991b1b; }
    .welcome-card { border: 1px dashed #e5e7eb; border-radius: 10px; padding: 2rem; text-align: center; background: #fafafa; }
    .example-question { color: #2563eb; cursor: pointer; text-decoration: underline; }
    .meta-row { display: flex; gap: 1rem; margin-top: 0.5rem; font-size: 0.85rem; color: #4b5563; }
    .source-box { border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; background: #f9fafb; font-size: 0.85rem; }
    .source-box b { color: #111827; }
    .stChatMessage { padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "ollama_manager": None,
        "embedding_manager": None,
        "vector_store": None,
        "chat_model": None,
        "chat_manager": None,
        "rag_pipeline": None,
        "current_generation_model": settings.DEFAULT_GENERATION_MODEL,
        "current_embedding_model": settings.DEFAULT_EMBEDDING_MODEL,
        "top_k": settings.DEFAULT_TOP_K,
        "enable_compare": False,
        "compare_model_b": None,
        "messages": [],
        "uploaded_docs": [],  # list of dicts: filename, content, num_pages, file_hash
        "indexed_hashes": set(),
        "documents_info": [],  # list of dicts for display
        "ollama_available": None,
        "catalog": None,
        "db_clear_confirm": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_ollama_manager():
    if st.session_state.ollama_manager is None:
        st.session_state.ollama_manager = build_ollama_manager(settings.OLLAMA_BASE_URL)
    return st.session_state.ollama_manager


def _get_catalog():
    if st.session_state.catalog is None:
        try:
            st.session_state.catalog = _get_ollama_manager().get_catalog()
            st.session_state.ollama_available = True
        except OllamaError as exc:
            st.session_state.ollama_available = False
            st.session_state.catalog = None
            st.error(str(exc))
    return st.session_state.catalog


def _chat_models() -> list[str]:
    catalog = _get_catalog()
    if not catalog:
        return [settings.DEFAULT_GENERATION_MODEL]
    names = [m.name for m in catalog.chat_models if "embed" not in m.name.lower()]
    if settings.DEFAULT_GENERATION_MODEL not in names:
        names.insert(0, settings.DEFAULT_GENERATION_MODEL)
    return names


def _get_chat_model(model_name: str):
    if ChatOllama is None:
        raise RuntimeError("langchain-ollama is not installed.")
    return ChatOllama(
        model=model_name,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.3,
    )


def _get_embedding_manager(model_name: str):
    if (
        st.session_state.embedding_manager is None
        or st.session_state.embedding_manager.model_name != model_name
    ):
        st.session_state.embedding_manager = build_embedding_manager(
            model_name=model_name,
            base_url=settings.OLLAMA_BASE_URL,
            ollama_manager=_get_ollama_manager(),
        )
        st.session_state.embedding_manager.validate()
    return st.session_state.embedding_manager


def _get_vector_store(embedding):
    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStore(
            embedding=embedding.get_embeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            collection_name=settings.CHROMA_COLLECTION_NAME,
        )
    return st.session_state.vector_store


def _rebuild_vector_store_for_embedding(embedding_manager):
    """Clear and re-create the vector store when the embedding model changes."""
    if st.session_state.vector_store is not None:
        st.session_state.vector_store.clear()
    st.session_state.vector_store = VectorStore(
        embedding=embedding_manager.get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    st.session_state.indexed_hashes = set()
    return st.session_state.vector_store


def _get_chat_manager():
    if st.session_state.chat_manager is None:
        st.session_state.chat_manager = ChatManager(
            max_messages=settings.MAX_CONVERSATION_MESSAGES
        )
    return st.session_state.chat_manager


def _get_rag_pipeline(chat_model, chat_manager):
    if (
        st.session_state.rag_pipeline is None
        or st.session_state.rag_pipeline.chat_model is not chat_model
    ):
        st.session_state.rag_pipeline = RagPipeline(
            vector_store=_get_vector_store(
                st.session_state.embedding_manager
            ),
            chat_model=chat_model,
            chat_manager=chat_manager,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            top_k=st.session_state.top_k,
        )
    else:
        st.session_state.rag_pipeline.chat_model = chat_model
        st.session_state.rag_pipeline.top_k = st.session_state.top_k
        st.session_state.rag_pipeline.chat_manager = chat_manager
        st.session_state.rag_pipeline.vector_store = _get_vector_store(
            st.session_state.embedding_manager
        )
    return st.session_state.rag_pipeline


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📄 Document Intelligence AI")
        st.caption(settings.APP_DESCRIPTION)
        st.divider()

        # Ollama status
        try:
            available = _get_ollama_manager().is_available()
            st.session_state.ollama_available = available
        except Exception:
            available = False
            st.session_state.ollama_available = False

        if available:
            st.success("✅ Ollama is running")
        else:
            st.error("❌ Ollama is unavailable")
            st.info("Start Ollama, then refresh this page.")

        st.divider()

        # Model catalog
        _get_catalog()
        chat_models = _chat_models()

        # Generation model selection
        st.subheader("Generation model")
        default_index = 0
        if st.session_state.current_generation_model in chat_models:
            default_index = chat_models.index(st.session_state.current_generation_model)
        selected_model = st.selectbox(
            "Select LLM",
            options=chat_models,
            index=default_index,
            key="generation_model_select",
        )
        if selected_model != st.session_state.current_generation_model:
            st.session_state.current_generation_model = selected_model
            st.session_state.rag_pipeline = None  # rebuild pipeline w/ new model
            st.rerun()

        st.caption(f"Embedding model: `{settings.DEFAULT_EMBEDDING_MODEL}`")
        new_embed = st.text_input(
            "Embedding model (change triggers re-indexing)",
            value=st.session_state.current_embedding_model,
            key="embedding_model_input",
        )
        if new_embed != st.session_state.current_embedding_model:
            st.session_state.current_embedding_model = new_embed
            st.session_state.vector_store = None
            st.session_state.rag_pipeline = None
            if st.session_state.uploaded_docs:
                _reindex_existing()
            st.rerun()

        st.divider()

        # Retrieval config
        st.subheader("Retrieval")
        st.session_state.top_k = st.slider(
            "Top-k chunks",
            min_value=1,
            max_value=settings.MAX_TOP_K,
            value=st.session_state.top_k,
            key="top_k_slider",
        )

        st.session_state.enable_compare = st.toggle(
            "Compare models",
            value=st.session_state.enable_compare,
            help="Send the same question to two models side by side.",
        )
        if st.session_state.enable_compare:
            other_models = [m for m in chat_models if m != st.session_state.current_generation_model]
            st.session_state.compare_model_b = st.selectbox(
                "Compare with",
                options=other_models,
                key="compare_model_b_select",
            )

        st.divider()

        # Document upload
        st.subheader("Upload documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if uploaded_files:
            _ingest_uploaded_files(uploaded_files)

        st.divider()

        # Document list
        st.subheader("Indexed documents")
        if st.session_state.documents_info:
            for doc in st.session_state.documents_info:
                status_class = "badge-ok" if doc["status"] == "ok" else "badge-err"
                st.markdown(
                    f"""
                    <div class="doc-item">
                        <span>{doc['filename']}</span>
                        <span>
                            <span class="badge {status_class}">{doc['status']}</span>
                            &nbsp;{doc['num_chunks']} chunks
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No documents indexed yet.")

        st.divider()

        # Actions
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Clear chat", use_container_width=True, key="clear_chat_btn"):
                _get_chat_manager().clear()
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button("🗑️ Clear database", use_container_width=True, key="clear_db_btn"):
                st.session_state.db_clear_confirm = True

        if st.session_state.db_clear_confirm:
            st.warning("This deletes all indexed documents. Cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm", type="primary", key="confirm_clear_db"):
                    _clear_database()
                    st.session_state.db_clear_confirm = False
                    st.rerun()
            with c2:
                if st.button("Cancel", key="cancel_clear_db"):
                    st.session_state.db_clear_confirm = False
                    st.rerun()


def _ingest_uploaded_files(uploaded_files) -> None:
    """Process and index newly uploaded files."""
    embedding_manager = _get_embedding_manager(st.session_state.current_embedding_model)
    vector_store = _get_vector_store(embedding_manager)
    chat_manager = _get_chat_manager()

    progress = st.sidebar.progress(0, text="Ingesting documents...")
    newly_indexed = 0

    for index, uploaded_file in enumerate(uploaded_files):
        content = uploaded_file.read()
        file_hash = compute_file_hash(content)

        if file_hash in st.session_state.indexed_hashes:
            continue

        with st.spinner(f"Processing {uploaded_file.name}..."):
            processed = process_document(
                content,
                uploaded_file.name,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )

        if processed.error:
            st.session_state.documents_info.append(
                {
                    "filename": uploaded_file.name,
                    "status": "error",
                    "num_pages": processed.num_pages,
                    "num_chunks": 0,
                    "error": processed.error,
                }
            )
            st.sidebar.error(f"{uploaded_file.name}: {processed.error}")
            continue

        added = vector_store.add_documents(processed.chunks, processed.chunks[0].metadata["document_id"]) if processed.chunks else 0
        # Recompute document_id from chunks for tracking
        document_id = processed.chunks[0].metadata["document_id"] if processed.chunks else ""

        st.session_state.indexed_hashes.add(file_hash)
        st.session_state.uploaded_docs.append(
            {
                "filename": uploaded_file.name,
                "content": content,
                "num_pages": processed.num_pages,
                "file_hash": file_hash,
                "document_id": document_id,
            }
        )
        st.session_state.documents_info.append(
            {
                "filename": uploaded_file.name,
                "status": "ok",
                "num_pages": processed.num_pages,
                "num_chunks": processed.num_chunks,
            }
        )
        newly_indexed += added
        progress.progress((index + 1) / len(uploaded_files), text=f"Indexed {uploaded_file.name}")

    progress.empty()
    if newly_indexed:
        st.sidebar.success(f"{newly_indexed} chunks indexed.")
        # Rebuild RAG pipeline to bind the (possibly new) vector store
        st.session_state.rag_pipeline = None


def _clear_database() -> None:
    if st.session_state.vector_store is not None:
        st.session_state.vector_store.clear()
        st.session_state.vector_store = None
    st.session_state.indexed_hashes = set()
    st.session_state.uploaded_docs = []
    st.session_state.documents_info = []
    st.session_state.rag_pipeline = None
    st.success("Document database cleared.")


def _reindex_existing() -> None:
    """Re-process and re-index all previously uploaded documents.

    Called when the embedding model changes, because embedding dimensions
    and vector spaces may differ.
    """
    if not st.session_state.uploaded_docs:
        return
    embedding_manager = _get_embedding_manager(st.session_state.current_embedding_model)
    vector_store = _get_vector_store(embedding_manager)
    st.session_state.indexed_hashes = set()
    st.session_state.documents_info = []
    progress = st.sidebar.progress(0, text="Re-indexing documents...")
    for index, doc in enumerate(st.session_state.uploaded_docs):
        processed = process_document(
            doc["content"], doc["filename"],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        if processed.error or not processed.chunks:
            st.session_state.documents_info.append(
                {"filename": doc["filename"], "status": "error",
                 "num_pages": doc["num_pages"], "num_chunks": 0,
                 "error": processed.error or "No chunks produced."}
            )
            continue
        document_id = processed.chunks[0].metadata["document_id"]
        added = vector_store.add_documents(processed.chunks, document_id)
        st.session_state.indexed_hashes.add(doc["file_hash"])
        st.session_state.documents_info.append(
            {"filename": doc["filename"], "status": "ok",
             "num_pages": processed.num_pages, "num_chunks": processed.num_chunks}
        )
        progress.progress((index + 1) / len(st.session_state.uploaded_docs),
                          text=f"Re-indexed {doc['filename']}")
    progress.empty()
    st.session_state.rag_pipeline = None


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
def _render_welcome() -> None:
    st.markdown(
        """
        <div class="welcome-card">
            <h2>Welcome to Document Intelligence AI</h2>
            <p style="color:#6b7280;">Upload PDF or TXT documents, then ask questions about their contents.<br>
            Answers are generated locally with Ollama and grounded in your documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Example questions")
    examples = [
        "What is the main topic of the document?",
        "Summarize the key findings in the uploaded PDF.",
        "List the advantages mentioned in the document.",
        "Compare the concepts discussed on page 2 and page 5.",
    ]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        with col:
            if st.button(example, use_container_width=True, key=f"example_{example}"):
                st.session_state.pending_question = example
                st.rerun()


def _render_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant" and sources:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(sources, start=1):
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <b>Source {i}:</b> {source['filename']} &nbsp;
                            <b>Page:</b> {source['page']}<br>
                            <i>{source['snippet']}</i>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def _render_chat() -> None:
    st.markdown(f"<h2 class='app-title'>{settings.APP_TITLE}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='app-subtitle'>{settings.APP_DESCRIPTION}</p>", unsafe_allow_html=True)

    if not st.session_state.documents_info:
        _render_welcome()
    else:
        total_chunks = sum(d["num_chunks"] for d in st.session_state.documents_info if d["status"] == "ok")
        st.info(f"{len(st.session_state.documents_info)} document(s) indexed • {total_chunks} chunks available for retrieval.")

    # Display chat history
    for message in st.session_state.messages:
        _render_message(message["role"], message["content"], message.get("sources"))

    # Pending question from example button
    pending = st.session_state.pop("pending_question", None)

    if user_input := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.documents_info):
        pending = user_input

    if pending:
        _handle_user_question(pending)


def _handle_user_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    _render_message("user", question)

    try:
        embedding_manager = _get_embedding_manager(st.session_state.current_embedding_model)
        chat_model = _get_chat_model(st.session_state.current_generation_model)
        chat_manager = _get_chat_manager()
        pipeline = _get_rag_pipeline(chat_model, chat_manager)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                if st.session_state.enable_compare and st.session_state.compare_model_b:
                    model_a = chat_model
                    model_b = _get_chat_model(st.session_state.compare_model_b)
                    result_a, result_b = pipeline.compare_models(
                        question, model_a, model_b
                    )
                    _display_compare_results(result_a, result_b)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"**{result_a.model_used}:** {result_a.answer}\n\n---\n\n**{result_b.model_used}:** {result_b.answer}",
                         "sources": result_a.sources}
                    )
                else:
                    result = pipeline.query(question)
                    st.markdown(result.answer)
                    st.caption(
                        f"Model: {result.model_used} • "
                        f"Generation time: {result.latency_seconds}s • "
                        f"Chunks retrieved: {len(result.context_chunks)}"
                    )
                    if result.sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(result.sources, start=1):
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <b>Source {i}:</b> {source['filename']} &nbsp;
                                        <b>Page:</b> {source['page']}<br>
                                        <i>{source['snippet']}</i>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                            "latency": result.latency_seconds,
                            "model": result.model_used,
                        }
                    )
    except ValueError as exc:
        st.warning(str(exc))
        st.session_state.messages.pop()
    except OllamaError as exc:
        st.error(str(exc))
        st.session_state.messages.pop()
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        logger.exception("Unexpected error handling question")
        st.error("An unexpected error occurred while generating the answer.")
        st.session_state.messages.pop()


def _display_compare_results(result_a: object, result_b: object) -> None:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"### {result_a.model_used}")
        st.markdown(result_a.answer)
        st.caption(f"Latency: {result_a.latency_seconds}s")
    with col_b:
        st.markdown(f"### {result_b.model_used}")
        st.markdown(result_b.answer)
        st.caption(f"Latency: {result_b.latency_seconds}s")
    st.divider()
    with st.expander("📚 Sources (shared context)", expanded=False):
        for i, source in enumerate(result_a.sources, start=1):
            st.markdown(
                f"""
                <div class="source-box">
                    <b>Source {i}:</b> {source['filename']} &nbsp;
                    <b>Page:</b> {source['page']}<br>
                    <i>{source['snippet']}</i>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    _init_session_state()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
