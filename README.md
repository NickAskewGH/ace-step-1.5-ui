# ACE-Step · Local Studio

A local music studio and CLI for ACE-Step 1.5 Turbo and XL Turbo, with an experimental ACE-Step-only speech lab. The browser and CLI share request validation, prompts, a persistent inference worker, and saved generation history.

## Start

This implementation targets **Linux, Python 3.12, and an NVIDIA CUDA GPU**. It uses your existing model downloads; it does not fetch replacement weights.

```sh
uv sync --locked --extra inference
uv run --extra inference ace-studio doctor
uv run --extra inference ace-studio serve
```

Open **http://127.0.0.1:7861**. The port differs from the MiniMax studio so both interfaces can be open. Avoid running both models on the GPU simultaneously.

The task runner is managed by uv:

```sh
uv run just setup
uv run --extra inference just studio
uv run --extra inference just doctor
uv run --extra inference just test
```

Default model directories are configured in [src/ace_studio/config.toml](src/ace_studio/config.toml):

- Shared components and Turbo: `~/Downloads/ACE-Step--Ace-Step1.5`
- XL Turbo decoder: `~/Downloads/ACE-Step--acestep-v15-xl-turbo`

Edit `[models].base` and `[models].xl` in that file and restart the Studio to change the defaults. Paths support `~` for your home directory. CLI flags (`ace-studio --models-path /path --xl-path /path serve`) override environment variables (`ACE_STUDIO_MODELS` / `ACE_STUDIO_XL`), which override the config file. `--output-dir` is a global option before the subcommand. The UI binds only to loopback; there is no public hosting or account requirement.

## Music studio

Choose a starter or write a caption, tagged lyrics, BPM, key, time signature, and language. Instrumental mode preserves your lyrics in the editor but sends an instrumental instruction to the model. Preview the assembled prompt, import/export JSON, and restore saved settings from the player. Drafts are saved in this browser.

The downloaded turbo models support:

- **Create music:** caption and lyrics, with optional style-reference audio.
- **Cover:** reinterpret uploaded source audio; adjust cover strength.
- **Repaint:** replace a selected region of uploaded source audio.

Uploads accept WAV, FLAC, MP3, or OGG, up to 100 MB and 600 seconds. Stem extraction, layer completion, LoRA training, and other base-model-only operations are not included in this interface. The [upstream model/task table](https://github.com/ACE-Step/ACE-Step-1.5#-model-zoo) documents these differences.

Select **Turbo** or **XL Turbo** in the engine strip. **Refresh models** checks for completed weight files, including shard sizes and the silence latent. XL remains disabled while the download is incomplete. Shared VAE, text encoder, and optional 1.7B language model are reused from the standard download. No extra copy of the large weights is needed.

**Compare Turbo + XL** queues the same request and actual seed for both models, then displays both audio players with loading time, rendering time, duration, and peak allocated GPU memory. These are sequential runs, not simultaneous generation. Same seeds provide a controlled comparison, not identical performances across different models. A warm model may have a loading advantage; compare the separate render figures and repeat if needed.

The default is eight diffusion steps, 30 seconds, language-model planning off, and CPU offload on. Offload conserves VRAM at a speed cost. Turn it off to benchmark warm inference when there is sufficient VRAM. Planning uses the downloaded 1.7B LM with the PyTorch backend; it adds latency and does not rewrite your caption. Its first tokenizer initialization can take roughly 90 seconds. Long clips can need substantially more memory. CUDA errors are reported in the UI; the failed worker is discarded before another job runs.

**Release GPU** unloads the idle worker. The next request loads weights again. Cancellation stops an active worker or removes a queued request. A model/settings change also replaces the worker, so the previous model does not remain resident.

## Experimental speech lab

**This is ACE-Step prompting for spoken word, not a supported streaming TTS engine.** Upstream describes music clips of 10–600 seconds and provides a `[spoken word]` lyric tag. Fast whole-clip music synthesis does not imply low-latency, faithful, or consistent speech. The speech lab deliberately permits requests for 0.5–30 seconds to explore that boundary. **The pinned backend pads short requests to at least 128 latent frames (5.12 seconds); a request for 2 seconds returned 5.12 seconds in GPU tests.** Requested duration is therefore not a promise of clip length. The queue reports actual clip duration and playback backlog. It uses only ACE-Step, as requested.

1. Select a model and describe the voice (accent, register, timbre, delivery).
2. Choose **Every completed word** or **Every sentence**, clip duration, seed, and offloading.
3. Click **Start live typing**, then type. Space/Enter or terminal punctuation submits completed words. Unfinished text stays editable. “Render remaining text” flushes the remaining suffix.
4. The app generates each unit as a separate clip with the same voice prompt, retains the loaded model between requests, and schedules playback in order. It cannot emit partial audio from an unfinished diffusion pass.
5. Watch queue depth and measured time to audio. If generation falls behind, input is buffered instead of discarded. Stop cancels the session’s jobs and immediately stops playback.

Speech defaults to retaining weights on the GPU for lower latency. Enable CPU offload if memory is insufficient.

Editing already submitted text requires stopping and starting a new session. Starting again renders the textarea from the beginning. Changing the global model picker during an active speech session affects future sessions; the active session retains its initial model and voice settings. The queue holds at most eight in-flight speech requests and 128 further text units in the browser. Words can be omitted, sung, repeated, or surrounded by silence/music; the speaker can change between clips. Sentence mode offers more context but is still experimental. A valid WAV is not proof of intelligible speech.

## CLI

```sh
# Generate a song or instrumental.
uv run --extra inference ace-studio generate --preset after-hours --duration 30 --seed 7
uv run --extra inference ace-studio generate --preset neon --model xl --output outputs/neon.wav

# Same seed and settings, both models.
uv run --extra inference ace-studio compare --preset neon --duration 30 --seed 7

# All request fields are available through JSON or --set field=JSON.
uv run --extra inference ace-studio generate --config examples/daybreak.json
uv run --extra inference ace-studio generate --caption 'Solo jazz piano' --instrumental \
  --set bpm=88 --set thinking=true --no-offload
uv run --extra inference ace-studio generate --lyrics-file lyrics.txt --duration 60

# Preview without loading the GPU.
uv run --extra inference ace-studio generate --preset daybreak --dry-run
uv run --extra inference ace-studio schema
uv run --extra inference ace-studio preset neon > neon.json

# Experimental speech. --live submits each completed word in a terminal.
uv run --extra inference ace-studio speak --text 'Good morning, welcome home.' \
  --voice-style 'Warm British alto, conversational, dry recording' --duration 5 --play
uv run --extra inference ace-studio speak --live --play --duration 2

# Saved generations, cancel a server-owned job, or export an existing WAV.
uv run --extra inference ace-studio history
uv run --extra inference ace-studio status GENERATION_ID
uv run --extra inference ace-studio cancel GENERATION_ID
uv run --extra inference ace-studio export outputs/GENERATION_ID/audio.wav
```

CLI flags override `--set`, which overrides the config file, which overrides the preset. Use `--set source_audio=/absolute/path.wav` for CLI cover/repaint input. Browser requests accept uploaded audio IDs; an exported browser config refers to that Studio’s uploads and must use local paths when passed to the CLI. `--output` copies a finished WAV or MP3 and a request JSON sidecar to a chosen location; use `--force` to replace existing files.

In live CLI mode, Ctrl+D submits any unfinished word and drains the queue. Ctrl+C cancels work. `--play` uses system `ffplay`. Browser playback uses Web Audio and needs a user gesture; Start live typing supplies that gesture. MP3 export requires system FFmpeg with `libmp3lame`, and keeps the WAV master intact.

A plain `uv run ace-studio …` may remove optional inference packages from the environment. Use `--extra inference` for generation, serving, and tests with the complete runtime; the Just recipes do this for you.

## Saved files and API

Each generation lives under `outputs/<id>/` with `audio.wav`, optional `audio.mp3`, `request.json`, `prompt.txt`, `metrics.json`, `result.json`, `status.json`, `worker.log`, and upstream raw output. The actual random seed is persisted before inference. Audio is native 48 kHz. MP3 downloads are generated on demand at 320 kbps.

The process keeps a warm worker until it exits, is cancelled, switches configuration, or you release it. A per-user Linux file lock prevents two ACE Studio workers from claiming the GPU concurrently. A separate CLI invocation will report that the Studio owns the GPU; release it in the UI first. This lock does not coordinate unrelated apps such as MiniMax. Completed jobs survive restarts; abandoned jobs are marked interrupted and never silently restarted.

Local API docs: **http://127.0.0.1:7861/docs**. Main endpoints:

- `GET /api/system`, `/api/schema`, `/api/presets`, `/api/jobs`
- `POST /api/preview`, `/api/jobs`, `/api/compare`, `/api/uploads`, `/api/release`
- `GET /api/jobs/{id}` and `/api/jobs/{id}/files/audio.wav` (or `audio.mp3`)
- `POST /api/jobs/{id}/cancel` and `/api/sessions/{session_id}/cancel`

## Runtime and checks

ACE-Step and its bundled nano-vLLM are pinned to upstream commit `ca1e85fe9430179831e6bc6be790c332190a3866`. TorchCodec is pinned to 0.10.0 to match upstream’s PyTorch 2.10 ([compatibility table](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)). The adapter uses `AceStepHandler`, `LLMHandler`, and `generate_music`; the language-model backend is PyTorch. Upstream code can synchronize checkpoint configuration/Python files, so the worker stages those small files in a temporary directory and links weights. The original downloads remain unchanged. Hugging Face offline mode is enabled inside the worker.

```sh
uv run --extra inference pytest
node --check src/ace_studio/static/app.js
node tests/speech.test.mjs
```

Tests cover request validation, prompts, incomplete weight files, warm reuse, model switching, queue cancellation/recovery, persisted history, cross-origin rejection, uploads, paired seeds, CLI overrides, and speech boundaries. Real inference measurements and verification limits are recorded in [VALIDATION.md](VALIDATION.md).

References: [upstream project](https://github.com/ACE-Step/ACE-Step-1.5), [Python inference API](https://github.com/ACE-Step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md), [prompting tutorial](https://github.com/ACE-Step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md). The model and upstream code retain their original licenses.
