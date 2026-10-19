// Extract only completed units. The unfinished suffix stays editable.
export function splitCompleted(text, boundary = 'word', flush = false) {
  if (flush) return {units: text.trim() ? [text.trim()] : [], consumed: text.length};
  const expression = boundary === 'sentence' ? /^\s*[\s\S]*?[.!?](?:\s+|$)/ : /^\s*\S+?(?:\s+|[.!?](?=\s|$))/;
  let rest = text, consumed = 0; const units = [];
  while (rest) {
    const match = rest.match(expression);
    if (!match) break;
    if (match[0].trim()) units.push(match[0].trim());
    consumed += match[0].length;
    rest = rest.slice(match[0].length);
  }
  return {units, consumed};
}

export class SpeechSession {
  constructor({api, showError, getModel, selectJob}) {
    Object.assign(this, {api, showError, getModel, selectJob});
    this.running = false; this.items = []; this.pending = []; this.dispatch = null;
    this.$ = id => document.getElementById(id);
    this.$('start-live').onclick = () => this.start().catch(showError);
    this.$('stop-live').onclick = () => this.stop().catch(showError);
    this.$('flush-speech').onclick = () => { if (this.running) this.input(true); else showError(new Error('Start a live session first.')); };
    this.$('speech-text').addEventListener('input', event => { if (!event.isComposing) this.input(); });
    this.$('speech-text').addEventListener('compositionend', () => this.input());
  }
  async start() {
    if (this.running) return;
    const duration = Number(this.$('speech-duration').value);
    const seed = Number(this.$('speech-seed').value);
    if (!Number.isFinite(duration) || duration < 0.5 || duration > 30 || !Number.isInteger(seed) || seed < 0 || seed > 2147483647) throw new Error('Use a clip duration from 0.5 to 30 seconds and a whole-number seed from 0 to 2147483647.');
    this.context = new AudioContext(); await this.context.resume(); this.nextStart = 0;
    this.session = crypto.randomUUID().replaceAll('-', ''); this.committed = '';
    this.items = []; this.pending = []; this.running = true; this.playChain = Promise.resolve();
    this.settings = {model: this.getModel(), voice_style: this.$('voice-style').value,
      duration: Number(this.$('speech-duration').value), seed: Number(this.$('speech-seed').value),
      offload: this.$('speech-offload').checked, boundary: this.$('speech-boundary').value};
    this.$('start-live').disabled = true; this.$('stop-live').disabled = false;
    for (const id of ['voice-style','speech-duration','speech-seed','speech-offload','speech-boundary']) this.$(id).disabled = true;
    this.$('live-indicator').textContent = 'LIVE'; this.$('live-indicator').classList.add('live');
    this.render(); this.input(); this.$('speech-text').focus();
  }
  input(flush = false) {
    if (!this.running) return;
    const text = this.$('speech-text').value;
    if (!text.startsWith(this.committed)) {
      this.showError(new Error('Already submitted words were edited. Stop and start a new session to render the revised text.'));
      return;
    }
    const {units, consumed} = splitCompleted(text.slice(this.committed.length), this.settings.boundary, flush);
    if (this.pending.length + units.length > 128) { this.showError(new Error('Typing buffer is full. Wait for the queue to drain.')); return; }
    this.committed = text.slice(0, this.committed.length + consumed);
    for (const unit of units) {
      const item = {text: unit, sequence: this.items.length, stage: 'buffered', submitted: performance.now()};
      this.items.push(item); this.pending.push(item);
    }
    this.pump(); this.render();
  }
  pump() {
    if (this.dispatch || !this.running) return;
    this.dispatch = this.send().catch(this.showError).finally(() => { this.dispatch = null; });
  }
  async send() {
    const session = this.session;
    while (this.running && this.session === session && this.pending.length && this.items.filter(i => ['queued','starting','loading','rendering'].includes(i.stage)).length < 8) {
      const item = this.pending.shift(); item.stage = 'submitting';
      try {
        const {boundary, ...settings} = this.settings;
        const job = await this.api('/api/jobs', {method: 'POST', body: { ...settings, mode:'speech', text:item.text, title:item.text.slice(0,160), session_id:session, sequence:item.sequence, thinking:false }});
        item.id = job.id; item.stage = job.stage;
      } catch (error) { item.stage = 'failed'; item.message = error.message; this.showError(error); }
      this.render();
    }
  }
  update(jobs) {
    if (!this.running) return;
    for (const item of this.items) {
      const job = jobs.find(j => j.id === item.id); if (!job) continue;
      item.stage = job.stage; item.message = job.message; item.metrics = job.metrics;
      if (job.stage === 'complete' && item.latency == null) item.latency = (performance.now()-item.submitted)/1000;
    }
    // Never let a later clip overtake an earlier, unfinished word.
    for (const item of this.items) {
      if (item.scheduled) continue;
      if (!['complete','failed','cancelled','interrupted'].includes(item.stage)) break;
      item.scheduled = true;
      if (item.stage === 'complete') {
        const session = this.session;
        this.playChain = this.playChain.then(async () => {
          const response = await fetch(`/api/jobs/${item.id}/files/audio.wav`);
          if (!response.ok) throw new Error('Could not load speech audio');
          const buffer = await this.context.decodeAudioData(await response.arrayBuffer());
          if (!this.running || session !== this.session) return;
          const source = this.context.createBufferSource(); source.buffer = buffer; source.connect(this.context.destination);
          const start = Math.max(this.context.currentTime + 0.03, this.nextStart);
          source.start(start); this.nextStart = start + buffer.duration;
        }).catch(error => { if (this.running) this.showError(error); });
      }
    }
    this.pump(); this.render();
  }
  async stop() {
    this.running = false;
    if (this.context && this.context.state !== 'closed') await this.context.close();
    if (this.dispatch) await this.dispatch; // Include an HTTP request already in flight in cancellation.
    await this.api(`/api/sessions/${this.session}/cancel`, {method:'POST'});
    this.pending = [];
    this.$('start-live').disabled = false; this.$('stop-live').disabled = true;
    for (const id of ['voice-style','speech-duration','speech-seed','speech-offload','speech-boundary']) this.$(id).disabled = false;
    this.$('live-indicator').textContent = 'STOPPED'; this.$('live-indicator').classList.remove('live');
    this.$('speech-status').textContent = 'Playback stopped and queued clips cancelled. Start again to render the text from the beginning.';
  }
  render() {
    const container = this.$('speech-clips'); container.replaceChildren();
    for (const item of this.items) {
      const row = document.createElement('div'); row.className = 'clip-row';
      const word = document.createElement('span'); word.textContent = item.text;
      const state = document.createElement('small'); state.textContent = item.latency != null ? `${item.stage} · ${item.latency.toFixed(1)}s to audio · ${item.metrics?.render_seconds ?? '?'}s render · ${item.metrics?.audio_seconds ?? '?'}s clip` : item.stage;
      row.append(word, state);
      if (item.id) { const button = document.createElement('button'); button.textContent = 'Inspect'; button.onclick = () => this.selectJob(item.id); row.append(button); }
      container.append(row);
    }
    const waiting = this.items.filter(i => !['complete','failed','cancelled','interrupted'].includes(i.stage)).length;
    this.$('queue-count').textContent = `${waiting} WAITING / ${this.items.length} CLIPS`;
    this.$('speech-status').textContent = this.running ? `${waiting} waiting · ${this.pending.length} buffered locally · ${Math.max(0, this.nextStart-(this.context?.currentTime||0)).toFixed(1)}s playback buffer. Submitted words stay fixed.` : 'Session stopped.';
  }
}
