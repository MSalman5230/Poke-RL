"""Bounded JSON-lines transport to the local, pinned Showdown simulator."""
import atexit
from collections import deque
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]


class Engine:
    def __init__(self, timeout=60):
        self.timeout = timeout
        self.queue = queue.Queue()
        self.errors = deque(maxlen=20)
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            [shutil.which("node") or "node", str(ROOT / "simulator/worker.cjs")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        def read():
            try:
                for line in self.process.stdout:
                    try:
                        self.queue.put(json.loads(line))
                    except ValueError:
                        self.queue.put({"error": f"Malformed simulator output: {line[:200]}"})
            finally:
                self.queue.put({"error": "Simulator exited"})
        threading.Thread(target=read, daemon=True).start()
        threading.Thread(target=lambda: self.errors.extend(line.rstrip() for line in self.process.stderr), daemon=True).start()
        atexit.register(self.close)

    def request(self, op, **payload):
        with self.lock:
            key = uuid.uuid4().hex
            self.process.stdin.write(json.dumps({"id": key, "op": op, **payload}) + "\n")
            self.process.stdin.flush()
            try:
                result = self.queue.get(timeout=self.timeout)
            except queue.Empty as exc:
                self.close()
                raise TimeoutError("Local simulator did not respond") from exc
            if result.get("error") or result.get("id") != key:
                raise RuntimeError(result.get("error") or f"Simulator ID mismatch: {result}; {list(self.errors)}")
            return result["result"]

    def batch(self, requests):
        return self.request("batch", requests=requests)

    def close(self):
        process = getattr(self, "process", None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if process:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream and not stream.closed:
                    stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def reference_environment():
    """The snapshot resolves the project's engine, never the original checkout."""
    os.environ["NODE_PATH"] = str(ROOT / "simulator/node_modules")
    os.environ.setdefault("PCA_PROBE_WORKERS", "4")
