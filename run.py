"""
Entry point: load env, build retriever + LLM, compile graph, then run an
interactive terminal chat loop.

Start with:

    python run.py

Type questions in the terminal. Type ``exit`` (or ``quit``) to stop.
"""

from __future__ import annotations

from app.runtime import ask_question, close_runtime, create_runtime, new_thread_id


def main() -> None:
    runtime = create_runtime()
    thread_id = new_thread_id()

    print("Self-RAG terminal chat is ready.")
    print(f"Chat session (thread_id): {thread_id}")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    try:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("Exiting chat.")
                break

            result = ask_question(runtime, question, thread_id)
            answer = result.get("answer", "")
            print(f"Assistant: {answer}\n")
    finally:
        close_runtime(runtime)


if __name__ == "__main__":
    main()
