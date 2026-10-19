"""Local checkpoint discovery and inexpensive incomplete-download checks."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tomllib


def weights_error(path: Path):
    try:
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 100_000_000:
                return "Invalid safetensors header (possibly a Git LFS pointer)"
            header = json.loads(stream.read(size))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        if path.stat().st_size != 8 + size + end:
            return "Incomplete weights: download is still in progress"
    except (OSError, ValueError, KeyError, struct.error):
        return "Missing or invalid weights"
    return None


def checkpoint_errors(path: Path, *, dit=False):
    errors = []
    if not (path / "config.json").is_file():
        errors.append(f"{path.name}: missing config.json")
    index = path / "model.safetensors.index.json"
    try:
        names = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else [
            "diffusion_pytorch_model.safetensors" if path.name == "vae" else "model.safetensors"]
        for name in names:
            if Path(name).name != name:
                errors.append("Invalid weight shard path")
                continue
            error = weights_error(path / name)
            if error:
                errors.append(f"{path.name}/{name}: {error}")
    except (OSError, ValueError, KeyError):
        errors.append(f"{path.name}: invalid shard index")
    if dit and not (path / "silence_latent.pt").is_file():
        errors.append(f"{path.name}: missing silence_latent.pt")
    return errors


class ModelPaths:
    def __init__(self, base=None, xl=None):
        with Path(__file__).with_name("config.toml").open("rb") as stream:
            defaults = tomllib.load(stream)["models"]
        self.base = Path(base or os.getenv("ACE_STUDIO_MODELS", defaults["base"])).expanduser().resolve()
        self.xl = Path(xl or os.getenv("ACE_STUDIO_XL", defaults["xl"])).expanduser().resolve()

    def dit(self, model):
        return self.xl if model == "xl" else self.base / "acestep-v15-turbo"

    def check(self, model, thinking=False):
        errors = checkpoint_errors(self.dit(model), dit=True)
        for component in ("vae", "Qwen3-Embedding-0.6B"):
            errors += checkpoint_errors(self.base / component)
        if thinking:
            errors += checkpoint_errors(self.base / "acestep-5Hz-lm-1.7B")
        return errors

    def report(self):
        result = {}
        for model, label in [("turbo", "Turbo · 2B"), ("xl", "XL Turbo · 4B")]:
            errors = self.check(model)
            result[model] = dict(label=label, path=str(self.dit(model)), ready=not errors, errors=errors)
        gpu = "NVIDIA GPU unavailable"
        try:
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return dict(models=result, gpu=gpu, inference_installed=importlib.util.find_spec("acestep") is not None,
                    lm_ready=not checkpoint_errors(self.base / "acestep-5Hz-lm-1.7B"))
