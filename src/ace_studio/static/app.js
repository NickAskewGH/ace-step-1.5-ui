import {SpeechSession} from './speech.mjs';
const $ = id => document.getElementById(id), form = $('song-form');
let selectedModel = 'turbo', selectedJob = null, currentAudio = null, allJobs = [], comparisonIds = [], schema, defaults = {}, imported = {};
let sourceAudio = null, referenceAudio = null, pollBusy = false;
const terminal = new Set(['complete','failed','cancelled','interrupted']);
function showError(error) { $('error').textContent = error.message || String(error); $('error').hidden = false; }
function clearError() { $('error').hidden = true; }
async function api(path, options={}) {
  const init = {...options};
  if (init.body && !(init.body instanceof FormData)) { init.body = JSON.stringify(init.body); init.headers = {'Content-Type':'application/json'}; }
  const response = await fetch(path, init); const data = await response.json();
  if (!response.ok) { const detail = Array.isArray(data.detail) ? data.detail.map(item => `${item.loc?.slice(1).join('.') || 'Request'}: ${item.msg}`).join('\n') : data.detail; throw new Error(typeof detail === 'string' ? detail : JSON.stringify(data)); }
  return data;
}
function showDialog(title, value) { $('dialog-title').textContent = title; $('dialog-content').textContent = typeof value === 'string' ? value : JSON.stringify(value,null,2); $('dialog').showModal(); }
$('close-dialog').onclick = () => $('dialog').close();
function setPage(page) {
  for (const name of ['music','speech','library']) $('page-'+name).hidden = name !== page;
  document.querySelectorAll('[data-page]').forEach(button => {button.classList.toggle('active', button.dataset.page===page); button.setAttribute('aria-current', button.dataset.page===page ? 'page' : 'false');});
  const headings = {music:['MUSIC STUDIO','An idea. A whole new sound.','Give your next song somewhere to begin.'],speech:['SPEECH LAB','Let the words find a voice.','An experiment in speech, one completed word at a time.'],library:['YOUR LIBRARY','Every take has a place.','Listen back, compare, and make another version.']};
  [$('page-label').textContent,$('heading').textContent,$('subtitle').textContent] = headings[page];
}
document.querySelectorAll('[data-page]').forEach(button => button.onclick = () => setPage(button.dataset.page));
function setModel(model) {
  selectedModel = model;
  for (const name of ['turbo','xl']) { $('model-'+name).classList.toggle('selected',name===model); $('model-'+name).setAttribute('aria-pressed',String(name===model)); }
  saveDraft();
}
$('model-turbo').onclick = () => setModel('turbo'); $('model-xl').onclick = () => setModel('xl');
function field(name,label,type='text',options=null) {
  const wrapper = document.createElement('label'); wrapper.textContent = label;
  const input = document.createElement(options ? 'select' : 'input'); input.name=name;
  if (options) for (const [value,title] of options) {const option=document.createElement('option'); option.value=value; option.textContent=title; input.append(option);}
  else input.type=type;
  const spec=schema.properties[name];
  if(type==='number') {input.step=['shift','lm_temperature','lm_top_p'].includes(name)?'0.05':'1'; if(spec.minimum!=null)input.min=spec.minimum; if(spec.maximum!=null)input.max=spec.maximum;}
  if(type==='checkbox') { wrapper.className='check'; input.checked=!!spec.default; } else input.value=spec.default??'';
  wrapper.append(input); return wrapper;
}
function readForm() {
  const request={...defaults,...imported,model:selectedModel,mode:'music',source_audio:sourceAudio,reference_audio:referenceAudio};
  for (const input of form.elements) {
    if (!input.name) continue;
    request[input.name] = input.type==='checkbox' ? input.checked : ['number','range'].includes(input.type) ? (input.value==='' ? null : Number(input.value)) : input.value;
  }
  return request;
}
function loadForm(data) {
  imported={...data};
  for (const input of form.elements) {
    if (!input.name || !(input.name in data)) continue;
    if (input.type==='checkbox') input.checked=data[input.name]; else input.value=data[input.name]??'';
  }
  // Saved API requests contain local paths; retain only upload tokens from this server.
  function uploadToken(value) { if (!value) return null; const name=value.split('/').pop(); return /^[a-f0-9]{32}\.(wav|flac|mp3|ogg)$/.test(name) ? name : null; }
  sourceAudio=uploadToken(data.source_audio); referenceAudio=uploadToken(data.reference_audio);
  $('source-label').textContent=sourceAudio || 'Required for cover and repaint.';
  $('reference-label').textContent=referenceAudio || 'Optional: audio to guide the style.';
  setModel(data.model||'turbo'); updateDuration();
}
function saveDraft() { if (!schema) return; try {localStorage.setItem('ace-studio-draft-v1',JSON.stringify(readForm()));} catch {} }
function updateDuration() { $('duration-value').textContent=form.elements.duration.value+'s'; }
form.addEventListener('input',()=>{updateDuration();saveDraft();});
for (const button of document.querySelectorAll('[data-lyric]')) button.onclick=()=>{form.elements.lyrics.value+='\n\n['+button.dataset.lyric+']\n';form.elements.lyrics.focus();saveDraft();};
$('preview').onclick=async()=>{try{showDialog('Your generation prompt',await api('/api/preview',{method:'POST',body:readForm()}));}catch(e){showError(e);}};
$('export').onclick=()=>{const blob=new Blob([JSON.stringify(readForm(),null,2)],{type:'application/json'}); const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='ace-session.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};
$('import').onclick=()=>$('import-file').click();
$('import-file').onchange=async event=>{try{const data=JSON.parse(await event.target.files[0].text());await api('/api/preview',{method:'POST',body:data});loadForm(data);clearError();}catch(e){showError(e);}event.target.value='';};
for (const [kind,fieldName] of [['source','source_audio'],['reference','reference_audio']]) {
  $(kind+'-file').onchange=async event=>{try{const data=new FormData();data.append('file',event.target.files[0]);const result=await api('/api/uploads',{method:'POST',body:data});if(kind==='source')sourceAudio=result.id;else referenceAudio=result.id;$(kind+'-label').textContent=`${result.name} · ${result.duration.toFixed(1)}s`;saveDraft();}catch(e){showError(e);}};
  $('clear-'+kind).onclick=()=>{if(kind==='source')sourceAudio=null;else referenceAudio=null;$(kind+'-file').value='';$(kind+'-label').textContent='No audio selected';saveDraft();};
}
async function generate(compare=false) {
  clearError(); const button=$(compare?'compare':'generate'); button.disabled=true;
  try {
    const result=await api(compare?'/api/compare':'/api/jobs',{method:'POST',body:readForm()});
    if(compare){comparisonIds=result.map(j=>j.id);$('comparison').hidden=false;selectedJob=result[0].id;}
    else selectedJob=result.id;
    await poll();
  } catch(e){showError(e);} finally{button.disabled=false;}
}
form.onsubmit=event=>{event.preventDefault();generate();}; $('compare').onclick=()=>generate(true);
$('cancel').onclick=async()=>{try{await api(`/api/jobs/${selectedJob}/cancel`,{method:'POST'});await poll();}catch(e){showError(e);}};
async function system() {
  const data=await api('/api/system');
  $('system-label').textContent=data.gpu.split(',')[0];
  $('system').onclick=()=>showDialog('Local runtime',data);
  const ready=[];
  for(const model of ['turbo','xl']) {const info=data.models[model];$('model-'+model).disabled=!info.ready;$('model-'+model).title=info.ready?info.path:info.errors.join('\n');if(info.ready)ready.push(model==='xl'?'XL Turbo':'Turbo');}
  $('model-note').textContent=ready.length?`${ready.join(' + ')} ready${!data.models.xl.ready?' · XL download incomplete':''}`:'Waiting for complete model downloads';
  $('compare').disabled=!data.models.turbo.ready||!data.models.xl.ready;
  if(!data.inference_installed)$('model-note').textContent='Install inference dependencies: uv sync --locked --extra inference';
}
$('refresh-models').onclick=()=>system().catch(showError);
$('release').onclick=async()=>{try{await api('/api/release',{method:'POST'});$('model-note').textContent='GPU released. Next generation will load the selected model.';}catch(e){showError(e);}};
function metricNodes(metrics) {
  const container=document.createElement('div');container.className='metrics';if(!metrics)return container;
  for(const [label,value] of [['RENDER',metrics.render_seconds+'s'],['LOAD',metrics.load_seconds+'s'],['AUDIO',metrics.audio_seconds+'s'],['PEAK VRAM',metrics.peak_vram_gb+' GB'],['SEED',metrics.seed]]) {const node=document.createElement('span');node.className='metric';const strong=document.createElement('b');strong.textContent=value;node.append(strong,document.createTextNode(label));container.append(node);}return container;
}
function linkFor(job,name,label) {const a=document.createElement('a');a.href=`/api/jobs/${job.id}/files/${name}`;a.textContent=label;a.download=name;return a;}
function selectJob(id) {selectedJob=id;renderPlayer();}
function renderPlayer() {
  const job=allJobs.find(j=>j.id===selectedJob);if(!job)return;
  $('track-title').textContent=job.title;$('track-model').textContent=job.request.model==='xl'?'XL TURBO':'TURBO';
  $('job-message').textContent=job.message;$('cancel').hidden=terminal.has(job.stage);
  $('job-progress').hidden=terminal.has(job.stage);$('job-progress').value=job.progress||0;
  if(job.stage==='complete') {
    if(currentAudio!==job.id){$('audio').src=`/api/jobs/${job.id}/files/audio.wav`;currentAudio=job.id;}
    $('audio').hidden=false;$('metrics').replaceChildren(metricNodes(job.metrics));
  } else {$('audio').pause();$('audio').hidden=true;currentAudio=null;$('metrics').replaceChildren();}
  const links=$('job-links');links.replaceChildren();
  if(job.stage==='complete') links.append(linkFor(job,'audio.wav','Download WAV'),linkFor(job,'audio.mp3','Download MP3'),linkFor(job,'metrics.json','Timing & metrics'));
  links.append(linkFor(job,'request.json','Request JSON'),linkFor(job,'prompt.txt','Prompt'));
  const reuse=document.createElement('button');reuse.textContent='Use these settings';reuse.onclick=()=>{if(job.request.mode==='speech'){setPage('speech');$('voice-style').value=job.request.voice_style;$('speech-text').value=job.request.text;}else{loadForm(job.request);setPage('music');}};links.append(reuse);
}
function renderHistory() {
  $('history').replaceChildren();$('library-grid').replaceChildren();
  for(const job of allJobs.slice(0,8)){const button=document.createElement('button');button.className='history-item';button.textContent=job.title;const sub=document.createElement('span');sub.textContent=`${job.request.model.toUpperCase()} · ${job.stage}`;button.append(sub);button.onclick=()=>selectJob(job.id);$('history').append(button);}
  if(!allJobs.length){const p=document.createElement('p');p.className='hint';p.textContent='Your songs and speech experiments will appear here.';$('library-grid').append(p);}
  for(const job of allJobs){const card=document.createElement('article');card.className='panel library-card';const tag=document.createElement('div');tag.className='tag';tag.textContent=`${job.request.mode.toUpperCase()} / ${job.request.model.toUpperCase()} / ${job.stage.toUpperCase()}`;const title=document.createElement('h3');title.textContent=job.title;const p=document.createElement('p');p.textContent=new Date(job.created_at).toLocaleString();const button=document.createElement('button');button.textContent='Open session ↗';button.onclick=()=>{selectJob(job.id);$('track-title').scrollIntoView({behavior:'smooth',block:'center'});};card.append(tag,title,p,metricNodes(job.metrics),button);$('library-grid').append(card);}
}
let comparisonSignature='';
function renderComparison(){
  if(!comparisonIds.length)return;
  const jobs=comparisonIds.map(id=>allJobs.find(j=>j.id===id)).filter(Boolean);const signature=JSON.stringify(jobs.map(j=>[j.id,j.stage,j.message]));if(signature===comparisonSignature)return;comparisonSignature=signature;
  const container=$('comparison-results');container.replaceChildren();
  for(const job of jobs){const card=document.createElement('div');card.className='comparison-card';const heading=document.createElement('h3');heading.textContent=job.request.model==='xl'?'XL Turbo · 4B':'Turbo · 2B';const status=document.createElement('p');status.textContent=job.message;card.append(heading,status,metricNodes(job.metrics));if(job.stage==='complete'){const audio=document.createElement('audio');audio.controls=true;audio.preload='metadata';audio.src=`/api/jobs/${job.id}/files/audio.wav`;card.append(audio,linkFor(job,'audio.wav','Download WAV'));}container.append(card);}
}
const speech=new SpeechSession({api,showError,getModel:()=>selectedModel,selectJob});
let historySignature='';
async function poll(){if(pollBusy)return;pollBusy=true;try{allJobs=await api('/api/jobs');const sig=JSON.stringify(allJobs.map(j=>[j.id,j.stage]));if(sig!==historySignature){historySignature=sig;renderHistory();}renderPlayer();renderComparison();speech.update(allJobs);}finally{pollBusy=false;}}
$('refresh-library').onclick=()=>poll().catch(showError);
async function init(){
  schema=await api('/api/schema');for(const [name,spec] of Object.entries(schema.properties))if('default'in spec)defaults[name]=spec.default;
  for(const args of [['bpm','Tempo (BPM)','number'],['keyscale','Key / scale'],['timesignature','Time signature','text',[['','Automatic'],['2','2/4'],['3','3/4'],['4','4/4'],['6','6/8']]],['language','Vocal language']])$('music-fields').append(field(...args));
  for(const args of [['seed','Seed (−1 = random)','number'],['steps','Diffusion steps','number'],['shift','Timestep shift','number'],['lm_temperature','LM temperature','number'],['lm_top_p','LM top-p','number'],['thinking','Plan with the language model','checkbox'],['offload','CPU offload','checkbox']])$('advanced-fields').append(field(...args));
  const presets=await api('/api/presets');for(const [name,data] of Object.entries(presets)){const button=document.createElement('button');button.className='preset';button.textContent=data.title;const sub=document.createElement('small');sub.textContent=name==='neon'?'Analog synths / 112 BPM':name==='daybreak'?'Acoustic folk / 96 BPM':'Neo-soul / 82 BPM';button.append(sub);button.onclick=()=>{loadForm({...data,model:selectedModel});setPage('music');};$('presets').append(button);}
  try{const saved=localStorage.getItem('ace-studio-draft-v1');if(saved)loadForm(JSON.parse(saved));}catch{}
  await system();await poll();setInterval(()=>poll().catch(showError),900);
}
init().catch(showError);
