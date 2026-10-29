"""One persistent, bounded local Showdown process. No network or tablet access."""
import atexit
from collections import deque
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
import uuid
from opponents.autoplay_reference.bot.runtime import budget as budgets

REVISION = "c0845f33d381402dfcb18ce324c20a94f18af075"
FORMAT = "gen9championsbssregmc"
NODE = Path(__file__).parent / "node"


class Showdown:
    def __init__(self, command=None):
        self.command = command or [shutil.which("node") or "node", str(NODE / "worker.cjs")]
        self.process = None
        self.output = queue.Queue()
        self.stderr = deque(maxlen=100)
        self.lock = threading.RLock()
        self.identity = None

    def start(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                return
            if self.process:
                self.close()
            self.output = queue.Queue()
            self.stderr.clear()
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True, encoding="utf-8", bufsize=1,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            process, output = self.process, self.output
            def reader():
                try:
                    for line in process.stdout:
                        try:
                            output.put(json.loads(line))
                        except (ValueError, TypeError):
                            output.put({"error": "Malformed simulator response", "line": line[:1000]})
                finally:
                    output.put({"error": "Simulator process exited"})
            def errors():
                for line in process.stderr:
                    self.stderr.append(line.rstrip())
            threading.Thread(target=reader, daemon=True).start()
            threading.Thread(target=errors, daemon=True).start()

    def request(self, op, *, timeout=None, **payload):
        with self.lock:
            if timeout is None:
                timeout = budgets.ceiling("turn")
            if timeout <= 0:
                raise TimeoutError("Showdown deadline passed before request")
            deadline = time.monotonic() + timeout
            self.start()
            request_id = uuid.uuid4().hex
            request = {"schema_version": 2, "id": request_id, "op": op,
                       "deadline_ms": int((time.time() + max(0, deadline - time.monotonic() - budgets.SIMULATOR_DEADLINE_MARGIN)) * 1000), **payload}
            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                response = self.output.get(timeout=max(0, deadline - time.monotonic()))
                if response.get("id") != request_id:
                    raise RuntimeError(f"Stale/malformed Showdown response: {response}; stderr={list(self.stderr)}")
                if response.get("error"):
                    raise RuntimeError(response["error"])
                return response["result"]
            except queue.Empty as exc:
                self.close()
                raise TimeoutError("Showdown worker exceeded its deadline; worker terminated") from exc
            except (OSError, BrokenPipeError, RuntimeError):
                self.close()
                raise

    def verify(self, timeout=None):
        result = self.request("ping", timeout=budgets.SIMULATOR_VERIFY_TIMEOUT if timeout is None else timeout)
        identity = result["identity"]
        if identity.get("revision") != REVISION or identity.get("format") != FORMAT:
            raise RuntimeError("Wrong Showdown revision/format")
        self.identity = identity
        return identity

    def close(self):
        with self.lock:
            self.identity = None
            process, self.process = self.process, None
            if process:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=budgets.SIMULATOR_SHUTDOWN_TIMEOUT)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream:
                        stream.close()


_client = None


def get():
    global _client
    if _client is None:
        _client = Showdown()
        atexit.register(_client.close)
    return _client
