"""Command-line entrypoints for the same requests used by the Studio."""
import json
from pathlib import Path
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from urllib.request import Request, urlopen
import uuid

import click
from pydantic import ValidationError

from .jobs import JobManager, TERMINAL
from .models import ModelPaths
from .schema import GenerationRequest, PRESETS


def output(value):
    click.echo(json.dumps(value, indent=2, default=str))


@click.group()
@click.option("--models-path", type=click.Path(path_type=Path), default=None, envvar="ACE_STUDIO_MODELS")
@click.option("--xl-path", type=click.Path(path_type=Path), default=None, envvar="ACE_STUDIO_XL")
@click.option("--output-dir", type=click.Path(path_type=Path), default="outputs", show_default=True)
@click.pass_context
def app(ctx, models_path, xl_path, output_dir):
    """ACE-Step local music studio and experimental speech lab."""
    ctx.obj = dict(paths=ModelPaths(models_path, xl_path), output_dir=output_dir)


@app.command()
@click.option("--port", type=click.IntRange(1024, 65535), default=7861, show_default=True)
@click.pass_obj
def serve(settings, port):
    """Open the local Studio (weights load on first generation)."""
    import uvicorn
    from .server import create_app
    click.echo(f"ACE Studio: http://127.0.0.1:{port}")
    uvicorn.run(create_app(**settings), host="127.0.0.1", port=port)


@app.command()
@click.pass_obj
def doctor(settings):
    """Check local checkpoints, dependency installation, and NVIDIA hardware."""
    output(settings["paths"].report())


@app.command()
def schema():
    """Print all supported JSON request fields and validation limits."""
    output(GenerationRequest.model_json_schema())


@app.command()
@click.argument("name", type=click.Choice(list(PRESETS)))
def preset(name):
    output(GenerationRequest(**PRESETS[name]).model_dump())


def make_request(config, preset_name, overrides, changes):
    data = dict(PRESETS.get(preset_name, {}))
    if config:
        data.update(json.loads(config.read_text()))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise click.ClickException("Use --set field=JSON, for example --set offload=false")
        try:
            data[key] = json.loads(value)
        except ValueError:
            data[key] = value
    data.update({k: v for k, v in changes.items() if v is not None})
    try:
        return GenerationRequest(**data)
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc


def wait_for(manager, jobs, play=False):
    for job in jobs:
        previous = None
        while True:
            job = manager.get(job["id"])
            status = (job["stage"], job["message"])
            if status != previous:
                click.echo(f"[{job['id'][:8]}] {job['stage']}: {job['message']}", err=True)
                previous = status
            if job["stage"] in TERMINAL:
                break
            time.sleep(0.15)
        if job["stage"] != "complete":
            raise click.ClickException(job["message"])
        output({"id": job["id"], "audio": manager.directory(job["id"]) / "audio.wav", "metrics": job.get("metrics")})
        if play:
            if not shutil.which("ffplay"):
                raise click.ClickException("Playback requires FFmpeg's ffplay")
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(manager.directory(job["id"]) / "audio.wav")], check=True)
    return job


def generation_options(fn):
    options = [click.option("--config", type=click.Path(exists=True, path_type=Path)),
               click.option("--preset", "preset_name", type=click.Choice(list(PRESETS))),
               click.option("--set", "overrides", multiple=True, help="Override any schema field: --set thinking=true"),
               click.option("--caption"), click.option("--lyrics-file", type=click.Path(exists=True, path_type=Path)),
               click.option("--model", type=click.Choice(["turbo", "xl"])),
               click.option("--duration", type=float), click.option("--seed", type=int),
               click.option("--steps", type=int), click.option("--title"),
               click.option("--instrumental/--vocals", default=None),
               click.option("--thinking/--no-thinking", default=None),
               click.option("--offload/--no-offload", default=None),
               click.option("--dry-run", is_flag=True), click.option("--play", is_flag=True),
               click.option("--output", "output_path", type=click.Path(path_type=Path)),
               click.option("--force", is_flag=True)]
    for option in reversed(options):
        fn = option(fn)
    return fn


def run_generation(settings, compare, config, preset_name, overrides, lyrics_file, dry_run, play, output_path, force, **changes):
    if lyrics_file:
        changes["lyrics"] = lyrics_file.read_text()
    request = make_request(config, preset_name, overrides, changes).resolved()
    requests = [request.model_copy(update={"model": model}) for model in ("turbo", "xl")] if compare else [request]
    if dry_run:
        output([dict(request=item.model_dump(), prompt=item.prompt()) for item in requests])
        return
    if output_path:
        if compare:
            raise click.ClickException("Use --output-dir before compare to select a comparison library")
        if output_path.suffix.lower() not in {".wav", ".mp3"}:
            raise click.ClickException("--output must end with .wav or .mp3")
        if not force and (output_path.exists() or output_path.with_suffix(".json").exists()):
            raise click.ClickException("Output or JSON sidecar already exists; use --force to replace")
    for item in requests:
        errors = settings["paths"].check(item.model, item.thinking)
        if errors:
            raise click.ClickException("\n".join(errors))
    manager = JobManager(**settings)
    try:
        job = wait_for(manager, [manager.submit(item) for item in requests], play)
        if output_path:
            from .audio import ensure_mp3
            source = manager.directory(job["id"]) / "audio.wav"
            if output_path.suffix.lower() == ".mp3":
                source = ensure_mp3(source)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output_path)
            shutil.copy2(manager.directory(job["id"]) / "request.json", output_path.with_suffix(".json"))
            click.echo(str(output_path))
    finally:
        manager.close()


@app.command()
@generation_options
@click.pass_obj
def generate(settings, **kwargs):
    """Generate audio from a preset, JSON config, or command-line settings."""
    run_generation(settings, False, **kwargs)


@app.command()
@generation_options
@click.pass_obj
def compare(settings, **kwargs):
    """Run Turbo and XL with identical settings and seed; save both results."""
    run_generation(settings, True, **kwargs)


@app.command()
@click.option("--text", default="")
@click.option("--voice-style", default="Warm, clear British narrator, conversational delivery")
@click.option("--model", type=click.Choice(["turbo", "xl"]), default="turbo")
@click.option("--duration", type=click.FloatRange(0.5, 30), default=2, help="Seconds per experimental speech clip.")
@click.option("--seed", type=int, default=7)
@click.option("--offload/--no-offload", default=False, help="Keep speech weights on GPU by default for lower latency.")
@click.option("--live", is_flag=True, help="Submit each completed word as you type. Ctrl+D finishes.")
@click.option("--play", is_flag=True)
@click.pass_obj
def speak(settings, text, voice_style, model, duration, seed, offload, live, play):
    """Experimental ACE-Step-only speech. This is not a streaming TTS model."""
    click.echo("Experimental: each word is a separate generated clip. Speech, voice consistency and real-time latency are not guaranteed.", err=True)
    manager = JobManager(**settings)
    session = uuid.uuid4().hex
    jobs = []
    player = None
    played = 0
    def submit(word):
        # Apply backpressure rather than aborting the entire live session when
        # typing outruns generation. Terminal input remains queued by the OS.
        while sum(manager.get(job["id"])["stage"] not in TERMINAL for job in jobs) >= 8:
            playback()
            time.sleep(0.05)
        request = GenerationRequest(mode="speech", model=model, text=word, voice_style=voice_style,
            duration=duration, seed=seed, offload=offload, title=word[:160], session_id=session, sequence=len(jobs))
        jobs.append(manager.submit(request))
    def playback():
        nonlocal player, played
        if not play or (player and player.poll() is None) or played >= len(jobs):
            return
        job = manager.get(jobs[played]["id"])
        if job["stage"] == "complete":
            player = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(manager.directory(job["id"]) / "audio.wav")])
            played += 1
        elif job["stage"] in TERMINAL:
            click.echo(job["message"], err=True)
            played += 1
    try:
        if play and not shutil.which("ffplay"):
            raise click.ClickException("Playback requires ffplay")
        if live:
            if not sys.stdin.isatty():
                raise click.ClickException("--live requires a terminal; use --text for scripts")
            old = termios.tcgetattr(sys.stdin)
            word = ""
            click.echo("Type words, then space/Enter. Ctrl+D drains the queue; Ctrl+C cancels.")
            try:
                tty.setcbreak(sys.stdin.fileno())
                while True:
                    playback()
                    if not select.select([sys.stdin], [], [], 0.05)[0]:
                        continue
                    char = sys.stdin.read(1)
                    if char == "\x04":
                        if word.strip():
                            submit(word)
                        break
                    if char in {"\x7f", "\b"}:
                        word = word[:-1]
                        click.echo("\b \b", nl=False)
                    elif char.isspace():
                        if word.strip():
                            submit(word)
                            word = ""
                        click.echo(char, nl=False)
                    elif char.isprintable():
                        word += char
                        click.echo(char, nl=False)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            while any(manager.get(j["id"])["stage"] not in TERMINAL for j in jobs) or (play and played < len(jobs)):
                playback()
                time.sleep(0.1)
            if player:
                player.wait()
            output([manager.get(j["id"]) for j in jobs])
        else:
            if not text.strip():
                raise click.ClickException("Supply --text or --live")
            submit(text)
            wait_for(manager, jobs, play)
    except (ValueError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        if player and player.poll() is None:
            player.terminate()
        manager.close()


@app.command()
@click.pass_obj
def history(settings):
    manager = JobManager(**settings)
    try:
        output(manager.list())
    finally:
        manager.close()


@app.command()
@click.argument("job_id")
@click.pass_obj
def status(settings, job_id):
    """Inspect a saved or running generation."""
    manager = JobManager(**settings)
    try:
        output(manager.get(job_id))
    except KeyError as exc:
        raise click.ClickException("Generation not found") from exc
    finally:
        manager.close()


@app.command("export")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
def export_audio(source):
    """Create a 320 kbps MP3 next to an existing WAV."""
    from .audio import ensure_mp3
    try:
        click.echo(str(ensure_mp3(source)))
    except (RuntimeError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


@app.command()
@click.argument("job_id")
@click.option("--url", default="http://127.0.0.1:7861")
def cancel(job_id, url):
    """Cancel a job owned by the running Studio server."""
    try:
        with urlopen(Request(f"{url}/api/jobs/{job_id}/cancel", method="POST"), timeout=10) as response:
            output(json.load(response))
    except OSError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    app()
