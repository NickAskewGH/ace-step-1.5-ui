"""Opt-in real inference check: uv run --extra inference python tests/gpu_smoke.py.

Uses the user's local downloads and saves inspectable audio under outputs/.
It is intentionally outside normal pytest discovery.
"""
import json
import time
from pathlib import Path

from ace_studio.jobs import JobManager, TERMINAL
from ace_studio.schema import GenerationRequest


def run(manager, request):
    job = manager.submit(request)
    last = None
    while True:
        job = manager.get(job['id'])
        if (job['stage'],job['message']) != last:
            print(job['id'][:8], job['stage'], job['message'], flush=True)
            last = job['stage'],job['message']
        if job['stage'] in TERMINAL:
            break
        time.sleep(.5)
    print(json.dumps(job,indent=2),flush=True)
    if job['stage'] != 'complete':
        raise RuntimeError(job['message'])
    return job


if __name__ == '__main__':
    manager=JobManager()
    results=[]
    try:
        music=run(manager,GenerationRequest(title='GPU check · Turbo music',caption='Gentle solo acoustic guitar, warm fingerpicked melody, instrumental',instrumental=True,duration=10,seed=7,offload=False))
        results.append(music)
        for word in ['Hello','world']:
            results.append(run(manager,GenerationRequest(title=f'GPU check · speech {word}',mode='speech',text=word,duration=2,seed=7,offload=False)))
        results.append(run(manager,GenerationRequest(title='GPU check · spoken sentence',mode='speech',text='Good morning. Welcome to the studio.',duration=5,seed=7,offload=False)))
        source=str(manager.directory(music['id'])/'audio.wav')
        results.append(run(manager,GenerationRequest(title='GPU check · cover',task='cover',source_audio=source,caption='Gentle solo piano',instrumental=True,duration=10,seed=7,offload=False)))
        results.append(run(manager,GenerationRequest(title='GPU check · repaint',task='repaint',source_audio=source,caption='Gentle solo acoustic guitar',instrumental=True,duration=10,repaint_start=2,repaint_end=5,seed=7,offload=False)))
    finally:
        manager.close()
        Path('outputs/gpu-validation.json').write_text(json.dumps(results,indent=2))
