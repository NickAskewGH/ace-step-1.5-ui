import json
from pathlib import Path
import struct
import sys
import time

from click.testing import CliRunner
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from ace_studio.cli import app
from ace_studio.jobs import JobManager, TERMINAL
from ace_studio.models import ModelPaths, weights_error
from ace_studio.schema import GenerationRequest
from ace_studio.server import create_app


def wait(manager, job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = manager.get(job['id'])
        if result['stage'] in TERMINAL:
            return result
        time.sleep(.02)
    pytest.fail('Worker did not finish')


@pytest.fixture
def manager(tmp_path):
    worker = tmp_path / 'fake_worker.py'
    worker.write_text('''import json, sys, time, os
from pathlib import Path
for line in sys.stdin:
    folder = Path(json.loads(line)['folder'])
    req = json.loads((folder/'request.json').read_text())
    print(json.dumps({'stage':'rendering','message':'Rendering','worker_pid':os.getpid()}), flush=True)
    time.sleep(2 if req['title']=='slow' else .03)
    if req['title']=='failure':
        print(json.dumps({'stage':'failed','message':'Deliberate failure'}), flush=True)
    else:
        print(json.dumps({'stage':'complete','message':'Done'}), flush=True)
''')
    manager = JobManager(tmp_path / 'output', worker_command=[sys.executable, str(worker)])
    yield manager
    manager.close()


def test_validation_and_prompt_separation():
    req = GenerationRequest(caption='jazz', lyrics='hello', seed=4)
    assert req.inference_params()['lyrics'] == 'hello'
    assert req.inference_params()['caption'] == 'jazz'
    assert not req.inference_params()['use_cot_caption']
    assert req.resolved().seed == 4
    assert GenerationRequest().resolved().seed >= 0
    for data in [{'duration': 2}, {'duration': float('nan')}, {'task':'cover'}, {'steps':0}, {'mode':'speech','text':' '}, {'mode':'speech','text':'Hi','thinking':True}]:
        with pytest.raises(ValidationError):
            GenerationRequest(**data)
    speech = GenerationRequest(mode='speech', text='Hello', duration=2, voice_style='soft alto')
    assert speech.prompt()['lyrics'] == '[spoken word]\nHello'
    assert 'soft alto' in speech.prompt()['caption']
    assert speech.inference_params()['thinking'] is False


def test_weights_detect_truncation_and_lfs(tmp_path):
    path = tmp_path / 'model.safetensors'
    header = json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    path.write_bytes(struct.pack('<Q',len(header)) + header + b'1234')
    assert weights_error(path) is None
    path.write_bytes(path.read_bytes()[:-1])
    assert 'Incomplete' in weights_error(path)
    path.write_text('version https://git-lfs.github.com/spec/v1')
    assert 'Invalid' in weights_error(path)


def test_warm_reuse_and_model_switch(manager):
    first = wait(manager, manager.submit(GenerationRequest(seed=7)))
    second = wait(manager, manager.submit(GenerationRequest(seed=8)))
    third = wait(manager, manager.submit(GenerationRequest(model='xl', seed=7)))
    assert first['stage'] == second['stage'] == third['stage'] == 'complete'
    assert first['worker_pid'] == second['worker_pid']
    assert first['worker_pid'] != third['worker_pid']
    assert json.loads((manager.directory(first['id'])/'request.json').read_text())['seed'] == 7


def test_cancel_active_queued_and_recovery(manager):
    active = manager.submit(GenerationRequest(title='slow'))
    queued = manager.submit(GenerationRequest(title='queued'))
    deadline = time.monotonic()+5
    while manager.get(active['id'])['stage'] != 'rendering':
        assert time.monotonic() < deadline
        time.sleep(.02)
    manager.cancel(queued['id'])
    manager.cancel(active['id'])
    assert wait(manager, active)['stage'] == 'cancelled'
    assert wait(manager, queued)['stage'] == 'cancelled'
    recovery = wait(manager, manager.submit(GenerationRequest(title='recovery')))
    assert recovery['stage'] == 'complete'


def test_failure_recovers_and_history_survives(manager):
    failed = wait(manager, manager.submit(GenerationRequest(title='failure')))
    assert failed['stage'] == 'failed'
    assert wait(manager, manager.submit(GenerationRequest()))['stage'] == 'complete'
    reader = JobManager(manager.output_dir)
    try:
        assert len(reader.list()) == 2
        with pytest.raises(KeyError): reader.get('../secrets')
    finally:
        reader.close()


def test_restart_marks_interrupted(manager):
    job = wait(manager, manager.submit(GenerationRequest()))
    job.update(stage='rendering', owner_pid=999999999, owner_start='no')
    (manager.directory(job['id'])/'status.json').write_text(json.dumps(job))
    reader = JobManager(manager.output_dir)
    try:
        assert reader.get(job['id'])['stage'] == 'interrupted'
    finally:
        reader.close()


def test_api_validation_upload_and_same_seed_compare(manager, monkeypatch):
    monkeypatch.setattr(manager.paths,'check',lambda *a: [])
    monkeypatch.setattr(manager.paths,'report',lambda: {'inference_installed':True})
    with TestClient(create_app(manager=manager)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/static/app.js').status_code == 200
        assert client.post('/api/jobs',json={'duration':2}).status_code == 422
        assert client.post('/api/jobs',json={'reference_audio':'/etc/passwd'}).status_code == 422
        assert client.post('/api/jobs',json={},headers={'Origin':'https://evil.example'}).status_code == 403
        assert client.get('/',headers={'Host':'evil.example'}).status_code == 400
        assert client.post('/api/uploads',files={'file':('fake.wav',b'not audio','audio/wav')}).status_code == 422
        assert not list((manager.output_dir/'uploads').iterdir())
        result = client.post('/api/compare',json={'seed':-1})
        assert result.status_code == 202
        jobs = result.json()
        assert jobs[0]['request']['seed'] == jobs[1]['request']['seed']
        assert {j['request']['model'] for j in jobs} == {'turbo','xl'}
        for job in jobs: wait(manager,job)
        assert client.get(f"/api/jobs/{jobs[0]['id']}/files/request.json").status_code == 200
        assert client.get(f"/api/jobs/{jobs[0]['id']}/files/secrets.txt").status_code == 404


def test_cli_dry_run_override_and_comparison():
    runner=CliRunner()
    result=runner.invoke(app,['compare','--preset','neon','--seed','9','--duration','20','--set','offload=false','--dry-run'])
    assert result.exit_code == 0, result.output
    requests=json.loads(result.output)
    assert all(item['request']['seed']==9 and item['request']['offload'] is False for item in requests)
    assert requests[0]['prompt']['lyrics']=='[Instrumental]'
    assert runner.invoke(app,['generate','--duration','1','--dry-run']).exit_code != 0


def test_queue_limit_and_session_cancellation(manager):
    session = 'a' * 32
    jobs = [manager.submit(GenerationRequest(title='slow', session_id=session)) for _ in range(32)]
    with pytest.raises(ValueError, match='full'):
        manager.submit(GenerationRequest())
    manager.cancel_session(session)
    assert all(manager.get(job['id'])['stage'] == 'cancelled' for job in jobs)


def test_mp3_export_preserves_wav(tmp_path):
    import shutil
    import soundfile as sf
    import numpy as np
    from ace_studio.audio import ensure_mp3
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg unavailable')
    path=tmp_path/'audio.wav'
    sf.write(path, np.sin(np.arange(4800)*2*np.pi*440/48000)*.1, 48000)
    original=path.read_bytes()
    mp3=ensure_mp3(path)
    assert mp3.stat().st_size > 100
    assert path.read_bytes() == original
    assert ensure_mp3(path) == mp3


def test_compare_preflight_does_not_queue_half_a_pair(manager, monkeypatch):
    monkeypatch.setattr(manager.paths,'check',lambda model,*a: ['XL download incomplete'] if model=='xl' else [])
    with TestClient(create_app(manager=manager)) as client:
        response=client.post('/api/compare',json={})
        assert response.status_code == 409
        assert manager.list() == []
