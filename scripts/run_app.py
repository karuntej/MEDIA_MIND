#!/usr/bin/env python3
"""
Run MediaMind stack with one command, with a --force reload that even
kills whatever is listening on your chosen ports.
"""
import argparse
import logging
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

DEVNULL  = subprocess.DEVNULL
BASE_DIR = Path(__file__).resolve().parents[1]

def parse_args():
    p = argparse.ArgumentParser(description="Launch MediaMind stack")
    p.add_argument("--no-ollama", action="store_true",
                   help="Don’t start the Ollama server")
    p.add_argument("--api-host", default="0.0.0.0", help="Host for FastAPI")
    p.add_argument("--api-port", type=int, default=8000, help="Port for FastAPI")
    p.add_argument("--ui-port",  type=int, default=8502, help="Port for Streamlit UI")
    p.add_argument("--force",    action="store_true",
                   help="Kill & restart services even if ports look busy")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()

def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False

def kill_pattern(pattern: str):
    if shutil.which("pkill"):
        subprocess.run(["pkill", "-f", pattern], stdout=DEVNULL, stderr=DEVNULL)
    else:
        logging.warning("pkill not found; cannot kill '%s'", pattern)

def kill_by_port(port: int):
    """Find whatever is listening on :port and kill it."""
    if not shutil.which("lsof"):
        logging.warning("lsof not found; cannot kill by port %d", port)
        return
    try:
        # -t for terse (just PIDs), -i :port
        pids = subprocess.check_output(["lsof", "-ti", f":{port}"]).decode().split()
        for pid in pids:
            logging.info("Killing PID %s on port %d", pid, port)
            subprocess.run(["kill", "-9", pid], stdout=DEVNULL, stderr=DEVNULL)
    except subprocess.CalledProcessError:
        # no process found
        pass

def start_process(cmd: list[str], cwd: Path = None, silence: bool = True):
    logging.info("Starting: %s", " ".join(cmd))
    if silence:
        return subprocess.Popen(cmd, cwd=cwd, stdout=DEVNULL, stderr=DEVNULL)
    else:
        return subprocess.Popen(cmd, cwd=cwd)

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s ▶ %(message)s",
    )

    # Always kill stray uvicorn/streamlit/ollama processes by name
    for pat in ("ollama serve", "uvicorn", "streamlit run"):
        kill_pattern(pat)

    # If --force, also kill by port
    if args.force:
        kill_by_port(11434)         # Ollama
        kill_by_port(args.api_port) # FastAPI
        kill_by_port(args.ui_port)  # Streamlit

    processes = []

    # 1) Ollama
    if not args.no_ollama:
        if args.force or not is_port_in_use("127.0.0.1", 11434):
            if shutil.which("ollama"):
                processes.append(start_process(["ollama", "serve"]))
                time.sleep(2)
            else:
                logging.warning("ollama not installed; skipping")
        else:
            logging.info("Ollama port in use; skipping")

    # 2) FastAPI (uvicorn)
    if args.force or not is_port_in_use(args.api_host, args.api_port):
        uv_cmd = [
            "uvicorn", "api.main:app",
            "--host", args.api_host,
            "--port", str(args.api_port),
            "--reload", "--log-level", "info"
        ]
        processes.append(start_process(uv_cmd))
        time.sleep(2)
    else:
        logging.info("API port %d in use; skipping", args.api_port)

    # 3) Streamlit UI
    ui_cwd = BASE_DIR / "ui"
    if args.force or not is_port_in_use("127.0.0.1", args.ui_port):
        st_cmd = [
            "streamlit", "run", "app.py",
            "--server.port",    str(args.ui_port),
            "--server.address", "127.0.0.1"
        ]
        # Show Streamlit logs so you know it's working
        processes.append(start_process(st_cmd, cwd=ui_cwd, silence=False))
    else:
        logging.info("UI port %d in use; skipping", args.ui_port)

    # Graceful shutdown
    def shutdown(signum, frame):
        logging.info("Shutting down all services…")
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if processes:
        processes[-1].wait()
    else:
        logging.error("No services started. Use --force to override port checks.")

if __name__ == "__main__":
    main()
