default:
    @just --list

# Install the complete local CUDA runtime.
setup:
    uv sync --locked --extra inference

# Start the Studio on http://127.0.0.1:7861.
studio:
    uv run --extra inference ace-studio serve

doctor:
    uv run --extra inference ace-studio doctor

# Forward CLI arguments, for example: just generate --preset neon.
generate *args:
    uv run --extra inference ace-studio generate {{args}}

compare *args:
    uv run --extra inference ace-studio compare {{args}}

speak *args:
    uv run --extra inference ace-studio speak {{args}}

test:
    uv run --extra inference pytest
    node --check src/ace_studio/static/app.js
    node tests/speech.test.mjs
