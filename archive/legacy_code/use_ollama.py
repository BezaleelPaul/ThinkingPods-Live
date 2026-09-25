import os
import ollama
import sys

# Target model for both mentor and extraction — all components use the same LLM
target_model = os.getenv("MENTOR_MODEL", "qwen2.5:3b")

def test_ollama_connection():
    global target_model
    print(f"=== Ollama Connection Test ({target_model}) ===")
    try:
        # FOR OFFLINE USE: Check if model exists without pulling
        print(f"Checking for model: {target_model}...")
        ollama.show(target_model)
        print(f"Successfully connected to Ollama and verified {target_model} is available!\n")
    except Exception as e:
        print(f"Model '{target_model}' not found.")
        print(f"Please pull the model: ollama pull {target_model}")
        print(f"And make sure Ollama is running (ollama serve).")
        sys.exit(1)

def generate_requirement(user_input):
    system_prompt = """You are a Requirements Engineering expert.
Generate high-quality software requirements following the ISO 29148 standard.
Focus on clarity, measurability, and completeness."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {user_input}\n\nTask: Generate 3 specific functional requirements based on this context."}
    ]

    print("Thinking...")
    try:
        response = ollama.chat(model=target_model, messages=messages)
        reply = response['message']['content'].strip()
        import re
        reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
        if not reply:
            reply = "I completed my reasoning but didn't produce a final response. Please try again."
        return reply
    except Exception as e:
        return f"Error during generation: {e}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ReqGPT Ollama Edition")
    parser.add_argument("context", nargs="?", help="Project context to generate requirements for")
    args = parser.parse_args()

    test_ollama_connection()

    if args.context:
        result = generate_requirement(args.context)
        print(f"\nContext: {args.context}")
        print("\n--- Generated Requirements ---")
        print(result)
    else:
        print("=== ReqGPT Interactive Tester (Ollama Edition) ===")
        print("Type your project context below (or 'quit' to exit).")

        while True:
            try:
                user_context = input("\nProject Context > ")
            except EOFError:
                break

            if user_context.lower() in ['quit', 'exit']:
                break

            if not user_context.strip():
                continue

            result = generate_requirement(user_context)
            print("\n--- Generated Requirements ---")
            print(result)