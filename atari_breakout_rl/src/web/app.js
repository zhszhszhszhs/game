'use strict';
const $ = id => document.getElementById(id);
const statuses = {idle:'待开始',loading:'加载模型',playing:'正在演示',paused:'已暂停',episode_end:'本局结束',stopped:'已停止',error:'运行错误'};
let current = null, commandBusy = false, refreshing = false, initialized = false;
let previousFrame = null, previousTime = null, previousHistory = '', streamConnected = false;
function showError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
function modelLabel(path) {
  if (path === 'random') return 'Random · 随机策略';
  const parts = path.split('/'), run = parts[0], file = parts[parts.length - 1];
  let kind = file === 'best_model.zip' ? '最佳模型' : file === 'final_model.zip' ? '最终模型' : file;
  const match = file.match(/_(\d+)_steps/);
  if (match) kind = `${Number(match[1]) / 1000}k checkpoint`;
  return `${run} · ${kind}`;
}
function connectStream() {
  if (!streamConnected && !document.hidden) {
    streamConnected = true;
    $('game').src = '/api/game/stream?t=' + Date.now();
  }
}
$('game').onerror = () => { streamConnected = false; };
function render(s) {
  current = s;
  $('status').textContent = statuses[s.status] || s.status;
  $('arena-mode').textContent = statuses[s.status] || s.status;
  $('reward').textContent = s.reward;
  $('episode').textContent = s.episode ? `EPISODE ${String(s.episode).padStart(2,'0')}` : 'EPISODE —';
  $('length').textContent = s.length.toLocaleString();
  $('last-reward').textContent = s.last_reward == null ? '—' : s.last_reward;
  const seconds = Math.floor(s.game_seconds || 0);
  $('game-time').textContent = `${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
  $('lives').replaceChildren();
  if (s.lives == null) $('lives').textContent = '—';
  else for (let i=0; i<Math.max(5,s.lives); i++) { const heart=document.createElement('span'); heart.textContent='♥'; if(i>=s.lives) heart.className='life-lost'; $('lives').appendChild(heart); }
  $('lives').setAttribute('aria-label', `剩余 ${s.lives == null ? '未知' : s.lives} 条生命`);
  const action = s.action.split(' + ')[0];
  document.querySelectorAll('[data-action]').forEach(el => el.classList.toggle('active',el.dataset.action===action));
  $('auto-fire').textContent = s.auto_fire_count ? `本局自动发球 ${s.auto_fire_count} 次` : '';
  $('action-description').textContent = s.action.includes('自动发球') ? '丢失生命后，环境自动重新发球' : `当前动作：${s.action} · 每个决策最多重复 4 帧`;
  $('agent-label').textContent = s.status==='idle' ? '尚未启动' : s.agent;
  $('pause').textContent = s.status==='paused' ? '继续' : '暂停';
  $('pause').disabled = commandBusy || !['playing','paused'].includes(s.status);
  $('stop').disabled = commandBusy || ['idle','stopped','error'].includes(s.status);
  $('start').disabled = commandBusy || s.status==='loading';
  $('start').textContent = s.status==='loading' ? '正在加载…' : s.episode ? '↻ 重新开始' : '▶ 开始演示';
  $('speed').value = String(s.speed);
  $('playback').textContent = `${s.speed}× 播放`;
  const options = ['random',...s.models], selected = $('agent').value;
  if (JSON.stringify(options)!==JSON.stringify([...$('agent').options].map(o=>o.value))) {
    $('agent').replaceChildren(...options.map(v=>{const o=document.createElement('option');o.value=v;o.textContent=modelLabel(v);return o;}));
    $('agent').value = options.includes(selected) ? selected : s.recommended_model;
  }
  if (!initialized) {
    $('agent').value = ['idle','stopped'].includes(s.status) ? s.recommended_model : s.agent;
    $('seed').value = s.seed;
    initialized = true;
  }
  updateSelection();
  $('game').hidden = !s.frame_ready;
  $('placeholder').hidden = s.frame_ready;
  $('placeholder-title').textContent = s.status==='loading' ? '正在加载模型与游戏环境' : '准备好，看它开始学习的成果';
  $('placeholder-copy').textContent = s.status==='loading' ? '首次加载可能需要几秒钟' : '选择模型，点击「开始演示」';
  const overlay = ['paused','stopped','episode_end','error'].includes(s.status) && s.frame_ready;
  $('overlay').hidden = !overlay;
  $('overlay-text').textContent = s.status==='episode_end' ? `本局 ${s.reward} 分 · 即将开始下一局` : statuses[s.status];
  const historyJSON = JSON.stringify(s.history);
  if (historyJSON !== previousHistory) {
    previousHistory = historyJSON;
    $('history-empty').hidden = s.history.length > 0;
    $('history').replaceChildren(...s.history.map(h=>{const item=document.createElement('div');item.className='history-item';item.title=`第 ${h.episode} 局 · 种子 ${h.seed} · ${h.length} 步${h.truncated?' · 超时':''}`;const label=document.createElement('small');label.textContent=`#${h.episode}`;const score=document.createElement('strong');score.textContent=h.reward;item.append(label,score);return item;}));
  }
  const now=performance.now();
  if (previousFrame!==null && now>previousTime && s.frame_id>=previousFrame) {
    const fps = Math.round((s.frame_id-previousFrame)*1000/(now-previousTime));
    $('fps').textContent = ['playing','loading'].includes(s.status) ? `源帧率 ${fps} fps` : '画面静止';
  }
  previousFrame=s.frame_id; previousTime=now;
  if (s.frame_ready) connectStream();
  if (s.error) showError(s.error);
}
function updateSelection() {
  const pending = current && current.episode && $('agent').value!==current.agent;
  $('model-note').textContent = pending ? '已更换选择，点击「重新开始」后生效。' : $('agent').value==='random' ? '随机选择动作，直观对比训练前后的游戏表现。' : '使用已保存模型的贪心策略；可切换不同训练阶段。';
}
async function api(path,body) {
  const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),25000);
  try {
    const response=await fetch('/api/game/'+path, {signal:controller.signal,...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});
    const data=await response.json(); if(!response.ok) throw Error(data.error || '请求失败');return data;
  } finally { clearTimeout(timer); }
}
async function refresh() {
  if(refreshing || document.hidden) return;
  refreshing=true;
  try { const s=await api('state');$('connection').className='connection online';$('connection-label').textContent='服务器已连接';render(s); }
  catch(e) { $('connection').className='connection offline';$('connection-label').textContent='连接中断 · 自动重试'; }
  finally { refreshing=false; }
}
async function command(path,body) {
  if(commandBusy) return;
  commandBusy=true;showError('');if(current) render(current);
  try { const s=await api(path,body);render(s); }
  catch(e) { showError(e.name==='AbortError'?'请求超时，请检查服务器连接。':e.message); }
  finally { commandBusy=false;if(current) render(current);refresh(); }
}
function start() { if(!$('seed').reportValidity())return;command('start',{agent:$('agent').value,seed:Number($('seed').value)}); }
function pause() { if(current && ['playing','paused'].includes(current.status)) command('pause',{paused:current.status!=='paused'}); }
async function fullscreen() {
  try { if(document.fullscreenElement) await document.exitFullscreen();else await $('arena').requestFullscreen(); }
  catch(e) { showError('此浏览器不支持全屏，请使用浏览器全屏功能。'); }
}
$('start').onclick=start;$('pause').onclick=pause;$('stop').onclick=()=>command('stop',{});
$('fullscreen').onclick=fullscreen;$('agent').onchange=updateSelection;
$('speed').onchange=()=>command('speed',{speed:Number($('speed').value)});
document.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey||e.repeat)return;if(e.code==='Space'){e.preventDefault();pause();}else if(e.key.toLowerCase()==='f')fullscreen();else if(e.key.toLowerCase()==='r')start();});
document.addEventListener('visibilitychange',()=>{if(document.hidden){$('game').removeAttribute('src');streamConnected=false;}else refresh();});
refresh();setInterval(refresh,250);
