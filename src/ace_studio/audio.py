"""Listening copies without changing the original generated WAV."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def ensure_mp3(source: Path):
    target = source.with_suffix('.mp3')
    if target.is_file():
        return target
    if not shutil.which('ffmpeg'):
        raise RuntimeError('MP3 export requires FFmpeg. The WAV master is available.')
    with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
        output = Path(temporary) / 'audio.mp3'
        result = subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(source),
                                 '-codec:a','libmp3lame','-b:a','320k',str(output)],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'MP3 conversion failed')
        output.replace(target)
    return target
