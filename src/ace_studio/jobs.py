"""Serialized jobs, persistent history, warm worker reuse, and cancellation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

from .models import ModelPaths

TERMINAL = {"complete", "failed", "cancelled", "interrupted"}


def process_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class JobManager:
    def __init__(self, output_dir=Path("outputs"), paths=None, worker_command=None):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = paths or ModelPaths()
        self.worker_command = worker_command
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ace-studio")
        self.lock = threading.RLock()
        self.jobs = {}
        self.worker = None
        self.worker_key = None
        self.active_id = None
        self.closed = False

    def directory(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise KeyError("Unknown generation")
        return self.output_dir / job_id

    def _save(self, job):
        file = self.directory(job["id"]) / "status.tmp"
        file.write_text(json.dumps(job, indent=2))
        file.replace(file.with_name("status.json"))

    def get(self, job_id):
        with self.lock:
            if job_id in self.jobs:
                return json.loads(json.dumps(self.jobs[job_id]))
            file = self.directory(job_id) / "status.json"
            if not file.is_file():
                raise KeyError("Unknown generation")
            job = json.loads(file.read_text())
            owner = process_identity(job.get("owner_pid", 0))
            if job["stage"] not in TERMINAL and (not owner or owner != job.get("owner_start")):
                job.update(stage="interrupted", message="The owning process stopped; this job was not restarted")
            return job

    def list(self):
        jobs = []
        for file in self.output_dir.glob("*/status.json"):
            try:
                jobs.append(self.get(file.parent.name))
            except (KeyError, ValueError, OSError):
                continue
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)[:200]

    def submit(self, request):
        with self.lock:
            if self.closed:
                raise ValueError("The studio is shutting down")
            if sum(j["stage"] not in TERMINAL for j in self.jobs.values()) >= 32:
                raise ValueError("Queue is full (32 requests). Pause typing or stop the live session.")
            request = request.resolved()
            job_id = uuid.uuid4().hex
            folder = self.directory(job_id)
            folder.mkdir()
            (folder / "request.json").write_text(request.model_dump_json(indent=2))
            (folder / "prompt.txt").write_text(json.dumps(request.prompt(), indent=2))
            job = dict(id=job_id, title=request.title, stage="queued", message="Queued", progress=0,
                       created_at=datetime.now(timezone.utc).isoformat(), request=request.model_dump(),
                       owner_pid=os.getpid(), owner_start=process_identity(os.getpid()))
            self.jobs[job_id] = job
            self._save(job)
            self.pool.submit(self._run, job_id)
            return self.get(job_id)

    def _stop_worker(self):
        if self.worker:
            if self.worker.poll() is None:
                self.worker.terminate()
                try:
                    self.worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
                    self.worker.wait()
            self.worker.stdin.close()
            self.worker.stdout.close()
        self.worker = None
        self.worker_key = None

    def _start_worker(self, request):
        key = (request["model"], request["offload"], request["thinking"])
        if self.worker and (self.worker_key != key or self.worker.poll() is not None):
            self._stop_worker()
        if self.worker is None:
            cmd = self.worker_command or [sys.executable, "-m", "ace_studio.worker", str(self.paths.base), str(self.paths.xl)]
            with (self.output_dir / "runtime.log").open("a") as log:
                self.worker = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                               stderr=log, text=True, bufsize=1)
            self.worker_key = key
        return self.worker

    def _run(self, job_id):
        process = None
        try:
            with self.lock:
                job = self.jobs[job_id]
                if job["stage"] in TERMINAL:
                    return
                self.active_id = job_id
                job.update(stage="starting", message="Preparing local inference")
                self._save(job)
                process = self._start_worker(job["request"])
                process.stdin.write(json.dumps({"folder": str(self.directory(job_id))}) + "\n")
                process.stdin.flush()
            started = time.monotonic()
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                with self.lock:
                    if job["stage"] == "cancelled":
                        break
                    job.update(event, elapsed_seconds=round(time.monotonic()-started, 3))
                    self._save(job)
                    if job["stage"] in TERMINAL:
                        break
            with self.lock:
                if job["stage"] not in TERMINAL:
                    job.update(stage="failed", message="Worker exited unexpectedly. See runtime.log.")
                    self._save(job)
                if job["stage"] == "failed" and self.worker is process:
                    self._stop_worker()
        except Exception as exc:
            with self.lock:
                if job["stage"] != "cancelled":
                    job.update(stage="failed", message=str(exc))
                    self._save(job)
        finally:
            with self.lock:
                self.active_id = None

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["stage"] in TERMINAL:
                return job
            if job_id not in self.jobs:
                raise ValueError("Cancel this job in the CLI or server that owns it")
            job = self.jobs[job_id]
            job.update(stage="cancelled", message="Cancelled")
            self._save(job)
            if self.active_id == job_id:
                # Terminate but let the reader own the stdout stream until it returns.
                process = self.worker
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            return self.get(job_id)

    def cancel_session(self, session_id):
        for job in list(self.jobs.values()):
            if job["request"].get("session_id") == session_id:
                self.cancel(job["id"])

    def release(self):
        with self.lock:
            if any(j["stage"] not in TERMINAL for j in self.jobs.values()):
                raise ValueError("Cancel or finish queued jobs before releasing the model")
            if self.active_id is not None:
                raise ValueError("Worker is finishing; try again shortly")
            self._stop_worker()

    def close(self):
        with self.lock:
            self.closed = True
            for job_id in list(self.jobs):
                self.cancel(job_id)
            if self.worker and self.worker.poll() is None:
                self.worker.terminate()
                try:
                    self.worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self._stop_worker()
