import sys
from pdf_loader import load_pdfs
from text_splitter import split_documents
from vector_store import build_vector_store, get_retriever, load_vector_store, save_vector_store
from rag_chatbot import RAGChatbot
from utils import (
    ChatbotError,
    ConfigError,
    PDFProcessingError,
    VectorStoreError,
    format_sources,
    get_logger,
)
logger = get_logger("main")

def main() -> None:
    pdf_paths = sys.argv[1:]
    store = None
    if pdf_paths:
        print(f"Processing {len(pdf_paths)} PDF(s)...")
        try:
            docs = load_pdfs(pdf_paths)
            chunks = split_documents(docs)
            store = build_vector_store(chunks)
            save_vector_store(store)
            print(f"Indexed {len(chunks)} chunk(s) from {len(pdf_paths)} file(s).\n")
        except (PDFProcessingError, VectorStoreError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        print("No PDF paths given, trying to load a previously saved index...")
        store = load_vector_store()
        if store is None:
            print("No saved index found. Run: python main.py path/to/file1.pdf [file2.pdf ...]")
            sys.exit(1)
    try:
        retriever = get_retriever(store)
        chatbot = RAGChatbot(retriever=retriever)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)
    print("Chatbot ready. Type your question, or 'exit' / 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            result = chatbot.ask(question)
            print(f"\nBot: {result['answer']}\n")
            sources = format_sources(result["source_documents"])
            if sources:
                print(f"Sources:\n{sources}\n")
        except ChatbotError as exc:
            print(f"\n {exc}\n")

if __name__ == "__main__":
    main()