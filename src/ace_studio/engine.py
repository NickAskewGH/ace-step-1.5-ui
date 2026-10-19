"""Adapter to the pinned upstream Python API, using only local model weights."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from .schema import GenerationRequest


class Engine:
    def __init__(self, paths):
        self.paths = paths
        self.handler = None
        self.lm = None
        self.staging = None

    def load(self, request, emit):
        errors = self.paths.check(request.model, request.thinking)
        if errors:
            raise RuntimeError("\n".join(errors))
        emit(stage="loading", message="Loading local weights; subsequent requests reuse this model", progress=0.02)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          TOKENIZERS_PARALLELISM="false", ACESTEP_DISABLE_TQDM="1")
        import torch
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and inference dependencies.")
        # Upstream can synchronize model Python/config files. Stage small files so the
        # original downloaded checkpoint is never modified; weights remain symlinks.
        self.staging = tempfile.TemporaryDirectory(prefix="ace-studio-")
        root = Path(self.staging.name)
        checkpoints = root / "checkpoints"
        checkpoints.mkdir()
        name = "acestep-v15-xl-turbo" if request.model == "xl" else "acestep-v15-turbo"
        target = checkpoints / name
        target.mkdir()
        for source in self.paths.dit(request.model).iterdir():
            if source.is_file():
                if source.suffix in {".json", ".py", ".pt"}:
                    shutil.copy2(source, target / source.name)
                else:
                    (target / source.name).symlink_to(source)
        for name_shared in ("vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"):
            (checkpoints / name_shared).symlink_to(self.paths.base / name_shared, target_is_directory=True)
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(checkpoints)
        paths = self.paths
        class LocalHandler(AceStepHandler):
            # Upstream normally downloads absent components here. This app must
            # fail locally instead, including when the XL download is unfinished.
            def _ensure_models_present(self, **kwargs):
                missing = paths.check(request.model, request.thinking)
                return ("\n".join(missing), False) if missing else None

        self.handler = LocalHandler()
        message, ok = self.handler.initialize_service(project_root=str(root), config_path=name,
            device="cuda", use_flash_attention=False, compile_model=False,
            offload_to_cpu=request.offload, offload_dit_to_cpu=request.offload, use_mlx_dit=False)
        if not ok:
            raise RuntimeError(message)
        self.lm = LLMHandler()
        if request.thinking:
            emit(stage="loading", message="Loading composition language model; first tokenizer setup can take about 90 seconds", progress=0.06)
            message, ok = self.lm.initialize(checkpoint_dir=str(checkpoints),
                lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda",
                offload_to_cpu=request.offload)
            if not ok:
                raise RuntimeError(message)

    def generate(self, request: GenerationRequest, folder: Path, emit):
        started = time.monotonic()
        cold = self.handler is None
        if cold:
            self.load(request, emit)
        loaded = time.monotonic()
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        import soundfile as sf
        import numpy as np
        import torch
        torch.cuda.reset_peak_memory_stats()
        emit(stage="rendering", message="Composing and rendering audio", progress=0.1)

        def progress(value, desc=None, **kwargs):
            emit(stage="rendering", message=desc or "Rendering", progress=max(0.1, min(0.95, float(value))))

        result = generate_music(self.handler, self.lm, GenerationParams(**request.inference_params()),
            GenerationConfig(batch_size=1, use_random_seed=False, seeds=[request.seed], audio_format="wav"),
            save_dir=str(folder / "raw"), progress=progress)
        if not result.success or not result.audios:
            raise RuntimeError(result.error or "ACE-Step returned no audio")
        shutil.copy2(result.audios[0]["path"], folder / "audio.wav")
        audio, rate = sf.read(folder / "audio.wav", always_2d=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise RuntimeError("The model produced empty or non-finite audio")
        elapsed = time.monotonic() - loaded
        metadata = dict(cold_start=cold, load_seconds=round(loaded-started, 3),
            render_seconds=round(elapsed, 3), total_seconds=round(time.monotonic()-started, 3),
            audio_seconds=round(len(audio)/rate, 3), sample_rate=rate,
            peak_vram_gb=round(torch.cuda.max_memory_allocated()/1024**3, 2),
            realtime_factor=round(elapsed/(len(audio)/rate), 3), peak=float(np.max(np.abs(audio))),
            seed=request.seed, model=request.model)
        (folder / "metrics.json").write_text(json.dumps(metadata, indent=2))
        (folder / "result.json").write_text(json.dumps(result.audios[0].get("params", {}), indent=2, default=str))
        emit(stage="complete", message="Audio ready", progress=1, metrics=metadata)
