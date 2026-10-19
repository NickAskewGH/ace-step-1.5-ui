"""Loopback-only studio API and static browser interface."""
from contextlib import asynccontextmanager
import importlib.util
from pathlib import Path
import re
import uuid
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .audio import ensure_mp3
from .jobs import JobManager
from .models import ModelPaths
from .schema import GenerationRequest, PRESETS


def create_app(output_dir=Path("outputs"), paths=None, manager=None):
    manager = manager or JobManager(output_dir, paths or ModelPaths())
    uploads = manager.output_dir / "uploads"
    uploads.mkdir(exist_ok=True)
    static = Path(__file__).parent / "static"

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()

    app = FastAPI(title="ACE-Step · Local Studio", lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin requests are not allowed"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/system")
    def system():
        return {**manager.paths.report(), "loaded": manager.worker_key}

    @app.get("/api/presets")
    def presets():
        return {key: GenerationRequest(**value).model_dump() for key, value in PRESETS.items()}

    @app.get("/api/schema")
    def schema():
        return GenerationRequest.model_json_schema()

    @app.post("/api/preview")
    def preview(request: GenerationRequest):
        return request.prompt()

    def prepare(request):
        errors = manager.paths.check(request.model, request.thinking)
        if errors:
            raise HTTPException(409, "\n".join(errors))
        if manager.worker_command is None and importlib.util.find_spec("acestep") is None:
            raise HTTPException(503, "Install the inference runtime: uv sync --locked --extra inference")
        for field in ("source_audio", "reference_audio"):
            value = getattr(request, field)
            if value:
                # API accepts upload IDs only, never arbitrary local filesystem paths.
                if not re.fullmatch(r"[a-f0-9]{32}\.(wav|flac|mp3|ogg)", value):
                    raise HTTPException(422, "Upload reference/source audio in this Studio first")
                path = uploads / value
                if not path.is_file():
                    raise HTTPException(422, "Uploaded audio no longer exists")
                setattr(request, field, str(path))
        return request

    @app.post("/api/jobs", status_code=202)
    def submit(request: GenerationRequest):
        try:
            return manager.submit(prepare(request))
        except ValueError as exc:
            raise HTTPException(429, str(exc)) from exc

    @app.post("/api/compare", status_code=202)
    def compare(request: GenerationRequest):
        request = request.resolved()
        requests = [prepare(request.model_copy(update={"model": model})) for model in ("turbo", "xl")]
        with manager.lock:
            if sum(j["stage"] not in {"complete", "failed", "cancelled", "interrupted"} for j in manager.jobs.values()) > 30:
                raise HTTPException(429, "Comparison needs two free queue slots")
            return [manager.submit(item) for item in requests]

    @app.get("/api/jobs")
    def jobs():
        return manager.list()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Generation not found") from exc

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job(job_id)
        try:
            return manager.cancel(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/sessions/{session_id}/cancel")
    def cancel_session(session_id: str):
        manager.cancel_session(session_id)
        return {"cancelled": True}

    @app.post("/api/release")
    def release():
        try:
            manager.release()
            return {"released": True}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/uploads", status_code=201)
    async def upload(file: UploadFile):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".wav", ".flac", ".mp3", ".ogg"}:
            raise HTTPException(422, "Use WAV, FLAC, MP3, or OGG audio")
        name = uuid.uuid4().hex + suffix
        path = uploads / name
        try:
            total = 0
            with path.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        raise HTTPException(413, "Audio upload limit is 100 MB")
                    target.write(chunk)
            import soundfile as sf
            try:
                info = sf.info(path)
                if info.duration <= 0 or info.duration > 600:
                    raise ValueError("Source audio must be between 0 and 600 seconds")
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(422, str(exc)) from exc
            return {"id": name, "name": file.filename, "duration": info.duration}
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get("/api/jobs/{job_id}/files/{filename}")
    def file(job_id: str, filename: str):
        saved = job(job_id)
        if filename not in {"audio.wav", "audio.mp3", "request.json", "metrics.json", "result.json", "prompt.txt", "worker.log"}:
            raise HTTPException(404, "File not found")
        if filename in {"audio.wav", "audio.mp3"} and saved["stage"] != "complete":
            raise HTTPException(409, "Audio is not ready")
        path = manager.directory(job_id) / filename
        if filename == "audio.mp3":
            try:
                ensure_mp3(path.with_suffix(".wav"))
            except (RuntimeError, OSError) as exc:
                raise HTTPException(503, str(exc)) from exc
        if not path.is_file():
            raise HTTPException(404, "File not ready")
        return FileResponse(path, filename=filename)

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
