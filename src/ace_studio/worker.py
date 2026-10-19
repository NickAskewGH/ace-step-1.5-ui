"""Persistent JSON-line inference worker. Termination releases GPU allocations."""
from contextlib import redirect_stdout
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

from .models import ModelPaths
from .schema import GenerationRequest


def main():
    protocol = sys.stdout
    def emit(**event):
        protocol.write(json.dumps(event) + "\n")
        protocol.flush()
    # This lock is shared by separate CLI / web processes, not just one queue.
    lock_path = Path(f"/tmp/ace-studio-{os.getuid()}.gpu.lock")
    with lock_path.open("a") as lock, redirect_stdout(sys.stderr):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit(stage="failed", message="Another ACE Studio process owns the GPU. Release its model or stop it first.")
            return
        from .engine import Engine
        engine = Engine(ModelPaths(sys.argv[1], sys.argv[2]))
        for line in sys.stdin:
            try:
                command = json.loads(line)
                folder = Path(command["folder"])
                request = GenerationRequest.model_validate_json((folder / "request.json").read_text())
                with (folder / "worker.log").open("a") as log, redirect_stdout(log):
                    stderr_copy = os.dup(2)
                    try:
                        os.dup2(log.fileno(), 2)
                        engine.generate(request, folder, emit)
                    except Exception:
                        traceback.print_exc(file=log)
                        raise
                    finally:
                        os.dup2(stderr_copy, 2)
                        os.close(stderr_copy)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                emit(stage="failed", message=str(exc))
                return  # A failed CUDA context must not be reused.


if __name__ == "__main__":
    main()
