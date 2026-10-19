# Local validation — 18 October 2026

Tested on Linux with an NVIDIA GeForce RTX 3090 (24 GB), Python 3.12, PyTorch 2.10.0+cu128, TorchCodec 0.10.0, and ACE-Step commit `ca1e85fe9430179831e6bc6be790c332190a3866`. All inference used the user's downloaded standard Turbo model, eight diffusion steps, and seed 7. No replacement model weights were downloaded.

## Real GPU generation

Times are measured by the worker: load includes model initialization, render includes conditioning, diffusion, decoding, and saving. Queue wait and browser polling are additional. Peak VRAM is PyTorch's peak allocated memory, not total device usage. These are single runs, not statistical benchmarks.

| Request | CPU offload | Load (s) | Render (s) | Actual audio (s) | Peak VRAM (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Instrumental, 10 seconds, first cold run | Off | 32.412 | 2.374 | 10.00 | 6.69 |
| Spoken “Hello”, requested 2 seconds, warm | Off | 0 | 1.232 | 5.12 | 6.39 |
| Spoken “world”, requested 2 seconds, warm | Off | 0 | 1.214 | 5.12 | 6.39 |
| Spoken sentence, requested 5 seconds, warm | Off | 0 | 1.219 | 5.12 | 6.39 |
| Cover, 10 seconds, warm | Off | 0 | 1.380 | 10.00 | 6.85 |
| Repaint seconds 2–5 of a 10-second clip, warm | Off | 0 | 1.339 | 10.00 | 6.92 |
| Daybreak vocal song, 30 seconds, cold worker | On | 9.842 | 10.719 | 30.00 | 4.53 |
| Daybreak, 10 seconds, optional 1.7B LM enabled | On | 58.950 | 31.003 | 10.00 | 4.82 |

The first run included initial import/cache overhead; subsequent cold workers can load faster. The optional language model initialized and generated successfully, but adds substantial latency. It is disabled by default.

The six-case GPU smoke test is reproducible with `uv run --extra inference python tests/gpu_smoke.py` after releasing the Studio's GPU worker. It writes `outputs/gpu-validation.json` and job artifacts. The 30-second song was also generated through the real CLI and exported to `outputs/daybreak-validation.mp3`, retaining its WAV master and request JSON. FFprobe verified a stereo 48 kHz MP3 at approximately 320 kbps and 30.024 seconds.

All generated WAVs were checked for finite, nonempty stereo 48 kHz audio. This verifies execution and file integrity, not musical quality or faithful speech. No perceptual listening or transcription assessment was completed.

## Speech findings

The pinned upstream conditioning code uses a minimum of 128 latent frames at 25 Hz, producing a **5.12-second minimum clip**. Both requested 2-second words and a requested 5-second sentence returned 5.12 seconds. Fast rendering therefore does not establish practical real-time word-by-word narration: clips can accumulate a playback backlog even when rendering takes about 1.2 seconds per word.

In Chrome, typing `Hello world ` submitted two independent jobs in order, reused the warm worker, and scheduled Web Audio playback. The first word took 11.673 seconds to load and 2.090 seconds to render; the second rendered in 1.176 seconds. Both clips were 5.12 seconds. The UI showed time to audio, actual duration, and a decreasing playback buffer. Stop ended the session and playback. Voice consistency and correct pronunciation remain unverified. This is explicitly an ACE-Step-only experiment, not a validated TTS system.

## Application checks

- Eleven automated Python tests pass: validation, incomplete weights, warm worker reuse and switching, cancellation/recovery, interrupted history, upload restrictions, origin checks, paired seeds, CLI overrides, queue limits, and MP3 preservation.
- JavaScript syntax checks and word/sentence boundary tests pass.
- `uv lock --check` and `uv build` pass; the wheel includes all four browser assets.
- Chrome checks covered the music and speech layouts, presets, prompt preview, draft persistence, live submission/playback, completed audio players, and the optional-LM generation. No browser console warnings or errors were reported during these checks.
- Local setup uses the project's own uv environment. The server runs on loopback at `http://127.0.0.1:7861`.

## Not yet verified

XL Turbo's download was incomplete during validation. Its selector is disabled until all expected weights, configuration, and silence latent are present. Model-switching orchestration and equal-seed comparison are covered by mocked tests; **real XL loading, VRAM requirements, speed, and audio quality have not been tested**. Use Refresh models after the download finishes, then compare a short clip first.

Long-duration generation, subjective quality, reliable speech intelligibility, and other GPU/OS combinations have not been validated.
