import subprocess
import re


# Models that are obviously not useful for a normal coding agent.
EXCLUDED_KEYWORDS = [
    "image",
    "embedding",
    "embed",
    "tts",
    "audio",
    "speech",
    "translate",
    "robotics",
    "video",
    "veo",
    "lyria",
    "whisper",
    "safety",
    "guard",
    "rerank",
    "paligemma",
    "esm2",
    "esmfold",
    "cosmos",
    "flux",
]


# Models that are particularly interesting for coding.
CODING_KEYWORDS = [
    "coder",
    "code",
    "qwen",
    "deepseek",
    "minimax",
    "glm",
    "step",
    "mistral",
    "llama",
    "nemotron",
    "gpt",
    "phi",
    "gemma",
]


def get_models():
    """Get the models currently visible to OpenCode."""

    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "opencode", "models"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print("Could not get OpenCode model list.")
        print(result.stderr)
        return []

    models = []

    for line in result.stdout.splitlines():

        line = line.strip()

        if not line:
            continue

        # Ignore obvious non-model output.
        if line.startswith("Error"):
            continue

        models.append(line)

    return models


def is_excluded(model):
    """Return True if the model is obviously unsuitable."""

    name = model.lower()

    for keyword in EXCLUDED_KEYWORDS:
        if keyword in name:
            return True

    return False


def coding_score(model):
    """Give coding-oriented models a simple score."""

    name = model.lower()

    score = 0

    for keyword in CODING_KEYWORDS:
        if keyword in name:
            score += 1

    # Explicit coding models get an extra boost.
    if "coder" in name:
        score += 5

    if "code" in name:
        score += 3

    return score


def main():

    print("=" * 60)
    print("OpenCode Automatic Model Manager")
    print("=" * 60)
    print()

    print("Getting models from OpenCode...")
    print()

    models = get_models()

    if not models:
        print("No models found.")
        return

    print(f"Found {len(models)} models.")
    print()

    candidates = []

    for model in models:

        if is_excluded(model):
            continue

        score = coding_score(model)

        candidates.append(
            {
                "name": model,
                "score": score,
            }
        )
    

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )
    MAX_HEALTH_TESTS = 12
    candidates = candidates[:MAX_HEALTH_TESTS]

    print(f"{len(candidates)} coding candidates.")

    print("=" * 60)
    print("CODING CANDIDATES")
    print("=" * 60)
    print()

    for item in candidates:

        print(
            f"[score {item['score']:2d}] "
            f"{item['name']}"
        )

    print()
    print("=" * 60)
    print(f"{len(candidates)} candidates after filtering")
    print("=" * 60)


if __name__ == "__main__":
    main()