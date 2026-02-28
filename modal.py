#!/usr/bin/env python3
from __future__ import annotations

import shlex
import subprocess
import sys


def main() -> None:
    cmd = [sys.executable, "-m", "modal", "run", "-m", "modal_app", *sys.argv[1:]]
    print(f"Executing: {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
