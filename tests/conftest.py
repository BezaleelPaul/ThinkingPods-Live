"""pytest configuration for ReqGPT / ThinkingPods test suite."""
import os
import sys

# Ensure project root, audits, and cli directories are in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

audits_dir = os.path.join(root_dir, "audits")
if audits_dir not in sys.path:
    sys.path.insert(0, audits_dir)

cli_dir = os.path.join(root_dir, "cli")
if cli_dir not in sys.path:
    sys.path.insert(0, cli_dir)
