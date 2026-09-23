import subprocess
import json
import sys
import time
import os
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

# Models are taken from the models you showed me.
# Put your preferred model first.
MODELS = [
    "nvidia/qwen/qwen3-coder-480b-a35b-instruct",
    "google/gemini-3.7-flash",
    "nvidia/minimaxai/minimax-m3",
    "nvidia/z-ai/glm-5.2",
]

# How many times we allow automatic recovery.
MAX_RECOVERIES = 10

# If an OpenCode process produces absolutely nothing for this
# long, terminate it and resume the session.
IDLE_TIMEOUT = 180

# Wait before retrying after an error.
RETRY_DELAY = 20

# If the same model fails this many times, move to the next.
MAX_FAILURES_PER_MODEL = 2

# IMPORTANT:
# --auto automatically approves permissions that aren't denied.
# Leave False initially. Set True only if you understand that
# OpenCode will be able to perform actions without asking you.
AUTO_APPROVE = False

COMPLETION_MARKER = "TASK_COMPLETE"


# ============================================================
# LOGGING
# ============================================================

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ============================================================
# MODEL / ERROR DETECTION
# ============================================================

def looks_like_provider_error(text):
    error_words = [
        "ResourceExhausted",
        "Worker load exceeded",
        "worker load",
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "DEGRADED",
        "Bad Request",
        "Not Found",
        "Internal Server Error",
        "Service Unavailable",
        "Gateway Timeout",
        "timeout",
        "timed out",
        "overloaded",
    ]

    lower = text.lower()

    return any(word.lower() in lower for word in error_words)


# ============================================================
# RUN OPENCODE
# ============================================================

def run_opencode(prompt, model, session_id=None):

    command = [
        r"C:\Users\devil\AppData\Roaming\npm\opencode.cmd",
        "run",
        "--format",
        "json",
        "--model",
        model,
    ]

    if session_id:
        command.extend([
            "--session",
            session_id,
        ])

    if AUTO_APPROVE:
        command.append("--auto")

    command.append(prompt)

    log("Starting OpenCode")
    log(f"Model: {model}")

    if session_id:
        log(f"Continuing session: {session_id}")

    log("")

    # Windows: run the npm .cmd launcher through cmd.exe.
    command_line = subprocess.list2cmdline(command)

    process = subprocess.Popen(
        ["cmd.exe", "/d", "/s", "/c", command_line],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    session = session_id
    last_output = time.time()
    completion_found = False
    error_found = False

    while True:

        line = process.stdout.readline()

        if line:

            last_output = time.time()

            line = line.strip()

            if not line:
                continue

            print(line, flush=True)

            try:
                event = json.loads(line)
            except json.JSONDecodeError:

                if looks_like_provider_error(line):
                    error_found = True

                continue

            # Get session ID.
            if event.get("sessionID"):
                session = event["sessionID"]

            part = event.get("part", {})

            if part.get("sessionID"):
                session = part["sessionID"]

            # Read model text.
            if event.get("type") == "text":

                text = event.get("part", {}).get("text", "")

                if text:

                    if COMPLETION_MARKER in text:
                        completion_found = True

                    if looks_like_provider_error(text):
                        error_found = True

            # Detect errors anywhere in the JSON event.
            raw = json.dumps(event)

            if looks_like_provider_error(raw):
                error_found = True

        else:

            if process.poll() is not None:
                break

            idle_for = time.time() - last_output

            if idle_for >= IDLE_TIMEOUT:

                log("")
                log(
                    f"OpenCode has produced no output for "
                    f"{IDLE_TIMEOUT} seconds."
                )

                log("Terminating stuck process...")

                process.kill()

                error_found = True

                break

            time.sleep(0.2)

    exit_code = process.poll()

    log("")
    log(f"OpenCode exited with code: {exit_code}")

    return {
        "session": session,
        "complete": completion_found,
        "error": error_found,
        "exit_code": exit_code,
    }


# ============================================================
# MAIN SUPERVISOR
# ============================================================

def main():

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print()
        print('  python supervisor.py "your coding task"')
        print()

        sys.exit(1)

    user_task = " ".join(sys.argv[1:])

    # --------------------------------------------
    # Give the model an explicit completion protocol.
    # --------------------------------------------

    initial_prompt = f"""
You are working on a coding task.

TASK:

{user_task}

IMPORTANT:

Work on the task directly in the project.

Continue working until the requested task is genuinely complete.

Do not claim completion if important work remains.

When you have completely finished the requested task, verify your work
and then put this exact marker on its own line at the very end:

{COMPLETION_MARKER}

Do NOT output {COMPLETION_MARKER} if the task is incomplete.
"""

    model_index = 0
    failure_count = 0
    recovery_count = 0
    session_id = None

    log("==========================================")
    log("OpenCode Autonomous Supervisor")
    log("==========================================")
    log("")
    log(f"Task: {user_task}")
    log("")
    log(f"Starting model: {MODELS[model_index]}")
    log("")

    prompt = initial_prompt

    while recovery_count <= MAX_RECOVERIES:

        model = MODELS[model_index]

        result = run_opencode(
            prompt=prompt,
            model=model,
            session_id=session_id,
        )

        # Save the session ID.
        if result["session"]:
            session_id = result["session"]

        # --------------------------------------------
        # COMPLETED
        # --------------------------------------------

        if result["complete"]:

            log("")
            log("==========================================")
            log("TASK COMPLETE")
            log("==========================================")
            log("")
            log(f"Session: {session_id}")

            return

        # --------------------------------------------
        # NOT COMPLETE
        # --------------------------------------------

        recovery_count += 1

        log("")
        log(
            f"Task did not reach {COMPLETION_MARKER}."
        )
        log(
            f"Recovery attempt "
            f"{recovery_count}/{MAX_RECOVERIES}"
        )

        # --------------------------------------------
        # Provider/model failure
        # --------------------------------------------

        if result["error"]:

            failure_count += 1

            log(
                f"Model failure count: "
                f"{failure_count}/{MAX_FAILURES_PER_MODEL}"
            )

            if failure_count >= MAX_FAILURES_PER_MODEL:

                if model_index < len(MODELS) - 1:

                    model_index += 1
                    failure_count = 0

                    log("")
                    log("Switching model.")
                    log(
                        f"New model: "
                        f"{MODELS[model_index]}"
                    )

                else:

                    log("")
                    log(
                        "All configured fallback models "
                        "have been exhausted."
                    )

                    log(
                        "Waiting before trying the first model again."
                    )

                    model_index = 0
                    failure_count = 0

                    time.sleep(60)

        # --------------------------------------------
        # Resume same session
        # --------------------------------------------

        if not session_id:

            log(
                "No session ID was obtained. "
                "Cannot safely resume."
            )

            break

        time.sleep(RETRY_DELAY)

        prompt = f"""
Continue the coding task from exactly where you stopped.

Do NOT start the task over.

Inspect the current project state and determine what remains incomplete.

Continue implementing and testing the requested changes.

Only when EVERYTHING requested by the original task is genuinely complete,
put this exact marker on its own line at the very end:

{COMPLETION_MARKER}

Do not use the marker if anything important remains unfinished.
"""

    log("")
    log("==========================================")
    log("SUPERVISOR STOPPED")
    log("Maximum recovery attempts reached.")
    log("==========================================")


if __name__ == "__main__":
    main()