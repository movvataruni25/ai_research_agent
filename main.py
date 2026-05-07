from src.agent import ResearchAgent

BANNER = """
╔══════════════════════════════════════════════════╗
║          AI Research Agent  (ReAct + RAG)        ║
╚══════════════════════════════════════════════════╝
 Commands: type your question, or 'exit' to quit.
 The agent can search documents, calculate, and more.
"""


def main():
    print(BANNER)

    try:
        agent = ResearchAgent()
    except ValueError as err:
        print(f"[Setup Error] {err}")
        return

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q", "bye"):
                print("\nAgent: Goodbye! Have a great day.")
                break

            print("\nAgent: Thinking...\n")
            answer = agent.run(user_input)
            print(f"\nAgent: {answer}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\nAgent: Interrupted. Goodbye!")
            break
        except Exception as err:
            print(f"\n[Error] {err}\n")


if __name__ == "__main__":
    main()
