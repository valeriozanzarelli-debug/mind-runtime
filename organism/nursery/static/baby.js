/**
 * Baby UI — una sola fonte di verità:
 * - Tu parli/scrivi → trascrizione nel dialogo → risposta solo via TTS (testo = audio)
 * - Flusso continuo silenzioso (pensa/sogna/percepisce) senza speech casuale
 */

const BASE = (window.ORGANISM_BASE || "").replace(/\/$/, "");
const VISION_W = 320;
const VISION_H = 256;
const HOLD_MS = 380;
const FLOW_MS = 2500;
const HEAR_DEBOUNCE_MS = 2000;
const SELF_SPEAK_GUARD_MS = 2800;

const orb = document.getElementById("orb");
const synCount = document.getElementById("syn-count");
const synGrown = document.getElementById("syn-grown");
const motorBar = document.getElementById("motor-bar");
const inhibBar = document.getElementById("inhib-bar");
const consciousBar = document.getElementById("conscious-bar");
const voiceHint = document.getElementById("voice-hint");
const thoughtLine = document.getElementById("thought-line");
const consciousnessStream = document.getElementById("consciousness-stream");
const dialogueLog = document.getElementById("dialogue-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const speechLine = document.getElementById("speech-line");
const statusEl = document.getElementById("status");
const cam = document.getElementById("cam");
const frameCanvas = document.getElementById("frame-canvas");
const btnHand = document.getElementById("btn-hand");
const btnMic = document.getElementById("btn-mic");

function handBtn(fn) {
  if (btnHand) fn(btnHand);
}

let born = false;
let dormant = true;
let mediaStream = null;
let micRecognition = null;
let micActive = false;
let lastSpoke = "";
let lastHeardPhrase = "";
let lastHeardAt = 0;
let holdTimer = null;
let isHolding = false;
let holdStarted = false;
let teachingFocus = false;
let flowBusy = false;
let phraseBusy = false;
let flowTimer = null;
let voiceUnlocked = false;
let utterQueue = [];
let utterRunning = false;
let italianVoice = null;
let isSelfSpeaking = false;
let selfSpeakGuardUntil = 0;
let pendingSelfHear = "";
let consciousnessSeq = 0;
const seenMindSeq = new Set();

const synth = window.speechSynthesis;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const LOCALE = "it-IT";

async function api(path, body) {
  const r = await fetch(`${BASE}${path}`, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function setStatus(text) {
  statusEl.textContent = text;
}

function isQuestionText(text) {
  const t = text.trim();
  return t.includes("?") || /^(cosa|chi|come|dove|quando|perché|perche|non so)/i.test(t);
}

function isTeachingPhrase(text) {
  const t = text.trim().toLowerCase();
  return /^(quest[ao]|quell[ao]|è |e un|e una|vedo |guarda |si chiama )/.test(t);
}

function normalizePhrase(text) {
  return text.toLowerCase().replace(/[.?!,]/g, "").replace(/\s+/g, " ").trim();
}

function phraseEchoesSelf(phrase) {
  const p = normalizePhrase(phrase);
  const own = normalizePhrase(lastSpoke);
  if (!p || !own) return false;
  if (p === own || p.includes(own) || own.includes(p)) return true;
  const pw = p.split(" ").filter((w) => w.length > 2);
  const ow = own.split(" ").filter((w) => w.length > 2);
  if (!pw.length || !ow.length) return false;
  const inter = pw.filter((w) => ow.includes(w)).length;
  return inter / Math.max(pw.length, ow.length) >= 0.55;
}

function pickItalianVoice() {
  if (italianVoice) return italianVoice;
  const voices = synth.getVoices();
  italianVoice = voices.find((v) => v.lang && v.lang.startsWith("it")) || voices[0] || null;
  return italianVoice;
}

if (typeof synth !== "undefined" && synth.onvoiceschanged !== undefined) {
  synth.onvoiceschanged = () => pickItalianVoice();
}

function unlockVoice() {
  if (voiceUnlocked) return;
  voiceUnlocked = true;
  try {
    if (synth.paused) synth.resume();
    const warm = new SpeechSynthesisUtterance(" ");
    warm.volume = 0.01;
    warm.lang = LOCALE;
    synth.speak(warm);
  } catch (_) { /* iOS */ }
  pickItalianVoice();
  drainSpeakQueue();
}

async function reportSelfHear(text) {
  if (!text || !born) return;
  try {
    const r = await api("/api/baby/self-hear", { text });
    if (r.feedback?.self_heard) {
      voiceHint.textContent = "sente la propria voce — si corregge";
    }
    if (r.moment?.consciousness_stream?.length) {
      renderConsciousnessStream(r.moment.consciousness_stream);
    }
  } catch (_) { /* offline */ }
}

function enqueueSpeak(text) {
  const t = (text || "").trim();
  if (!t) return;
  if (utterQueue[utterQueue.length - 1] === t) return;
  utterQueue.push(t);
  drainSpeakQueue();
}

function drainSpeakQueue() {
  if (utterRunning || !utterQueue.length) return;
  if (!voiceUnlocked) return;
  const text = utterQueue.shift();
  utterRunning = true;
  doSpeak(text, () => {
    utterRunning = false;
    drainSpeakQueue();
  });
}

function doSpeak(text, onDone) {
  appendDialogueBubble("organism", text);
  if (speechLine) {
    speechLine.hidden = false;
    speechLine.textContent = text;
  }
  orb.classList.add("speaking");
  orb.classList.remove("wants-voice");
  isSelfSpeaking = true;
  endMicListen();
  pendingSelfHear = text;
  lastSpoke = text;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = LOCALE;
  const vp = emotionVoiceParams();
  u.rate   = vp.rate;
  u.pitch  = vp.pitch;
  u.volume = vp.volume;
  const voice = pickItalianVoice();
  if (voice) u.voice = voice;
  const finish = () => {
    orb.classList.remove("speaking");
    isSelfSpeaking = false;
    selfSpeakGuardUntil = Date.now() + SELF_SPEAK_GUARD_MS;
    const spoke = pendingSelfHear;
    pendingSelfHear = "";
    if (speechLine) speechLine.hidden = true;
    reportSelfHear(spoke);
    if (!micActive && !isHolding) idleHint();
    onDone?.();
  };
  u.onend = finish;
  u.onerror = () => {
    pendingSelfHear = "";
    finish();
  };
  synth.speak(u);
}

function speak(text) {
  if (!text) return;
  if (!voiceUnlocked) {
    enqueueSpeak(text);
    voiceHint.textContent = "";
    return;
  }
  enqueueSpeak(text);
}

function idleHint() {
  if (micActive) {
    voiceHint.textContent = "parla ora… rilascia 🎤 quando hai finito";
  } else if (isSelfSpeaking) {
    voiceHint.textContent = "sta parlando…";
  } else if (dormant) {
    voiceHint.textContent = "dorme — consolidamento · scrivi o parla per svegliarlo";
  } else {
    voiceHint.textContent = "tieni 🎤 per parlare · scrivi sotto";
  }
}

// Stato emotivo corrente — aggiornato ad ogni momento
let currentEmotion = { dominant: "curiosity", joy: 0.3, fear: 0.0, sadness: 0.0, shame: 0.0, trust: 0.5, curiosity: 0.6, anger: 0.0 };

function applyBrain(brain, moment) {
  if (!brain) return;
  synCount.textContent = (brain.synapses ?? 0).toLocaleString("it-IT");
  const grown = brain.synapses_grown ?? 0;
  synGrown.textContent = grown > 0 ? `+${grown} da quando è nato` : "";
  motorBar.style.width = `${Math.min(100, (brain.motor_pressure ?? 0) * 100)}%`;
  inhibBar.style.width = `${Math.min(100, (brain.inhibition ?? 0) * 100)}%`;
  if (consciousBar && moment?.consciousness) {
    consciousBar.style.width = `${Math.min(100, (moment.consciousness.ignition ?? 0) * 100)}%`;
  }
  orb.classList.remove("wants-voice", "conscious", "afraid", "happy", "sleeping", "sad", "stressed");
  if (dormant) orb.classList.add("sleeping");
  const emo = moment?.emotion;
  const tone = moment?.social_tone;
  if (emo) Object.assign(currentEmotion, emo);
  if (emo?.dominant === "fear" || tone?.is_angry) orb.classList.add("afraid");
  else if (emo?.dominant === "joy") orb.classList.add("happy");
  else if (emo?.dominant === "sadness" || (emo?.sadness > 0.5)) orb.classList.add("sad");
  const stress = (emo?.shame ?? 0) + (emo?.fear ?? 0) + (emo?.sadness ?? 0);
  if (stress > 1.2) orb.classList.add("stressed");
  if (moment?.consciousness?.conscious) {
    orb.classList.add("conscious");
    if (!isSelfSpeaking) {
      voiceHint.textContent = `coscienza · ${moment.consciousness.focus || "attivo"}`;
    }
  } else if (!isHolding && !holdStarted && !phraseBusy && !micActive && !isSelfSpeaking) {
    idleHint();
  }
}

/** Parametri voce in base allo stato emotivo corrente.
 *  La voce riflette l'interno — non è recitazione, è lo stato che parla. */
function emotionVoiceParams() {
  const e = currentEmotion;
  const joy     = e.joy     ?? 0;
  const fear    = e.fear    ?? 0;
  const sadness = e.sadness ?? 0;
  const shame   = e.shame   ?? 0;
  const trust   = e.trust   ?? 0.5;
  const curiosity = e.curiosity ?? 0.4;
  const stress  = shame + fear + sadness;

  let pitch  = 1.0;
  let rate   = 0.88;
  let volume = 1.0;

  if (stress > 1.3) {
    // Sotto forte stress / quasi pianto — voce spezzata, lenta, bassa
    pitch  = 0.78 + Math.random() * 0.08;  // trema leggermente
    rate   = 0.72;
    volume = 0.75;
  } else if (sadness > 0.5 || shame > 0.4) {
    // Triste o in imbarazzo — voce più lenta e bassa
    pitch  = 0.88;
    rate   = 0.80;
    volume = 0.85;
  } else if (fear > 0.4) {
    // Spaventato — voce più alta e rapida
    pitch  = 1.15;
    rate   = 1.05;
    volume = 0.9;
  } else if (joy > 0.6) {
    // Felice/eccitato — voce più alta e vivace
    pitch  = 1.12;
    rate   = 0.95;
    volume = 1.0;
  } else if (curiosity > 0.65) {
    // Curioso — tono leggermente su, ritmo normale
    pitch  = 1.05;
    rate   = 0.90;
    volume = 1.0;
  } else if (trust > 0.7) {
    // Calmo/fiducioso — tono basso, voce lenta e misurata
    pitch  = 0.95;
    rate   = 0.85;
    volume = 1.0;
  }

  return { pitch, rate, volume };
}

function formatTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" });
}

function renderDialogue(entries) {
  if (!dialogueLog || !entries?.length) return;
  dialogueLog.innerHTML = entries
    .map((row) => {
      const role = row.role === "organism" ? "organism" : "tu";
      const label = role === "organism" ? "Lui" : "Tu";
      const text = escapeHtml(row.text || "");
      const when = formatTime(row.t);
      return `<div class="dialogue-bubble ${role}"><span class="bubble-meta">${label}${when ? ` · ${when}` : ""}</span><p class="bubble-text">${text}</p></div>`;
    })
    .join("");
  dialogueLog.scrollTop = dialogueLog.scrollHeight;
}

function appendDialogueBubble(role, text) {
  if (!dialogueLog || !text) return;
  const cls = role === "organism" ? "organism" : "tu";
  const label = cls === "organism" ? "Lui" : "Tu";
  const when = formatTime(Date.now() / 1000);
  const el = document.createElement("div");
  el.className = `dialogue-bubble ${cls}`;
  el.innerHTML = `<span class="bubble-meta">${label} · ${when}</span><p class="bubble-text">${escapeHtml(text)}</p>`;
  dialogueLog.appendChild(el);
  while (dialogueLog.children.length > 80) dialogueLog.removeChild(dialogueLog.firstChild);
  dialogueLog.scrollTop = dialogueLog.scrollHeight;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function mindEventHtml(ev) {
  const kind = ev.kind || "note";
  const title = escapeHtml(ev.title || kind);
  const detail = escapeHtml(ev.detail || "");
  return `<article class="mind-event mind-${kind}" data-seq="${ev.seq || 0}"><span class="mind-icon">${ev.icon || "·"}</span><div class="mind-body"><strong>${title}</strong>${detail ? `<p>${detail}</p>` : ""}</div></article>`;
}

function appendMindEvents(events) {
  if (!consciousnessStream || !events?.length) return;
  let added = 0;
  for (const ev of events) {
    const seq = ev.seq || 0;
    if (seq && seenMindSeq.has(seq)) continue;
    if (seq) seenMindSeq.add(seq);
    consciousnessStream.insertAdjacentHTML("beforeend", mindEventHtml(ev));
    added += 1;
  }
  while (consciousnessStream.children.length > 48) {
    const first = consciousnessStream.firstElementChild;
    const rm = first?.dataset?.seq;
    if (rm) seenMindSeq.delete(Number(rm));
    first?.remove();
  }
  if (added) consciousnessStream.scrollTop = consciousnessStream.scrollHeight;
}

function renderConsciousnessStream(lines) {
  if (!lines?.length) return;
  const events = lines.map((ln, i) => ({
    seq: consciousnessSeq - lines.length + i + 1,
    kind: "note",
    icon: "·",
    title: "Nota",
    detail: ln,
  }));
  appendMindEvents(events);
}

async function sendPhrase(phrase, source = "caregiver") {
  const text = (phrase || "").trim();
  if (!text || phraseBusy) return;
  if (!born) {
    setStatus("baby non ancora nato — attendi...");
    return;
  }
  phraseBusy = true;
  chatSend?.setAttribute("disabled", "true");
  appendDialogueBubble("tu", text);
  if (chatInput) chatInput.value = "";
  voiceHint.textContent = `sente · «${text.slice(0, 40)}»`;
  try {
    if (dormant) {
      try {
        await api("/api/baby/wake", {});
        dormant = false;
        orb.classList.remove("sleeping");
      } catch (_) { /* */ }
    }
    const r = await api("/api/baby/hear", visionBody({ phrase: text, source }));
    handleHearResponse(r);
  } catch (err) {
    console.error("sendPhrase", err);
    setStatus("errore rete — riprova");
  } finally {
    phraseBusy = false;
    chatSend?.removeAttribute("disabled");
  }
}

function handleHearResponse(r) {
  if (r.mode === "self_feedback") {
    voiceHint.textContent = "è la sua voce — si ascolta";
    if (r.moment) applyMoment(r.moment);
  } else if (r.mode === "vision_object") {
    const p = r.parsed || {};
    voiceHint.textContent = r.consolidated
      ? `impara · ${p.object || r.name} ✓`
      : `${p.object || r.name} · ${r.trials ?? 1}/3`;
    const spoken = r.moment?.spoke || r.parsed?.phrase || "";
    if (spoken) applyMoment({ ...r.moment, spoke: spoken });
  } else if (r.moment) {
    applyMoment(r.moment);
  }
  if (r.dialogue?.length) renderDialogue(r.dialogue);
  refreshState();
}

async function sendChat(text) {
  await sendPhrase(text || chatInput?.value || "", "caregiver");
}

async function pollMind() {
  if (!born) return;
  try {
    const r = await api(`/api/baby/consciousness?n=20&since=${consciousnessSeq}`);
    if (r.seq > consciousnessSeq) consciousnessSeq = r.seq;
    if (r.events?.length) appendMindEvents(r.events);
  } catch (_) { /* offline */ }
}

function applyMoment(moment) {
  if (!moment) return;
  applyBrain(moment.brain, moment);
  if (moment.consciousness_stream?.length) renderConsciousnessStream(moment.consciousness_stream);
  const th = moment.thought;
  const dream = moment.dream;
  if (dream?.active && dream.content) {
    thoughtLine.hidden = false;
    thoughtLine.textContent = `sogna · ${dream.content.slice(0, 80)}`;
  } else if (th?.themes?.length) {
    thoughtLine.hidden = false;
    const p = moment.from_thought ? "pensa · " : moment.understood ? "capisce · " : "";
    thoughtLine.textContent = p + th.themes.slice(0, 6).join(" · ");
  } else {
    thoughtLine.hidden = true;
  }
  if (moment.spoke) speak(moment.spoke);
}

async function stabilizeBaby() {
  try {
    const r = await api("/api/baby/stabilize", { aggressive: true });
    dormant = true;
    orb.classList.add("sleeping");
    const syn = r.synapses_after ?? r.synapses ?? 0;
    const pruned = r.pruned_synapses ?? 0;
    setStatus(`sonno · ${Number(syn).toLocaleString("it-IT")} sinapsi (−${pruned})`);
    idleHint();
    return r;
  } catch (err) {
    console.warn("stabilize", err);
    return null;
  }
}

async function wakeOrBirth() {
  setStatus("sta caricando…");
  try {
    const ready = await api("/api/baby/ready");
    const hasLife = ready.born || ready.resumed_from_disk || ready.state_file_exists;
    if (hasLife) {
      born = true;
      orb.classList.add("awake");
      setStatus("caricamento memoria…");
      const existing = await api("/api/baby/state?lite=1");
      dormant = Boolean(existing.dormant ?? true);
      if (dormant) orb.classList.add("sleeping");
      applyBrain(existing.brain, existing.last_moment);
      if (existing.dialogue?.length) renderDialogue(existing.dialogue);
      try {
        const mind = await api("/api/baby/consciousness?n=24&since=0");
        consciousnessSeq = mind.seq || 0;
        if (mind.events?.length) {
          consciousnessStream.innerHTML = "";
          seenMindSeq.clear();
          appendMindEvents(mind.events);
        }
      } catch (_) { /* */ }
      await stabilizeBaby();
      await refreshState();
      return;
    }
    setStatus("sta nascendo…");
    await api("/api/baby/birth", {});
    born = true;
    orb.classList.remove("sleeping");
    orb.classList.add("awake");
    dormant = false;
    await refreshState();
  } catch (err) {
    console.error("wakeOrBirth", err);
    setStatus("errore di caricamento — riprova");
  }
}

function captureFrame() {
  const ctx = frameCanvas.getContext("2d");
  frameCanvas.width = VISION_W;
  frameCanvas.height = VISION_H;
  ctx.drawImage(cam, 0, 0, VISION_W, VISION_H);
  const img = ctx.getImageData(0, 0, VISION_W, VISION_H);
  const gray = new Array(VISION_W * VISION_H);
  let rSum = 0, gSum = 0, bSum = 0, wSum = 0;
  const cx = VISION_W / 2, cy = VISION_H / 2;
  let lumSum = 0;
  for (let i = 0, j = 0; i < img.data.length; i += 4, j++) {
    lumSum += 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
  }
  const lumMean = lumSum / (VISION_W * VISION_H);
  const threshold = lumMean + 16;
  for (let y = 0, j = 0; y < VISION_H; y++) {
    for (let x = 0; x < VISION_W; x++, j++) {
      const i = j * 4;
      const r = img.data[i], g = img.data[i + 1], b = img.data[i + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      gray[j] = Math.round(lum);
      const dist = Math.hypot(x - cx, y - cy);
      let w = 1.2 - Math.min(0.8, dist / (Math.min(VISION_W, VISION_H) * 0.55));
      if (lum >= threshold) w += 1.5 + (lum - threshold) / 60;
      rSum += r * w; gSum += g * w; bSum += b * w; wSum += w;
    }
  }
  return {
    gray,
    rgba: Array.from(img.data),
    color_rgb: { r: rSum / wSum, g: gSum / wSum, b: bSum / wSum },
  };
}

function visionBody(extra = {}) {
  const body = { image_w: VISION_W, image_h: VISION_H, ...extra };
  if (mediaStream && cam.videoWidth) {
    const f = captureFrame();
    body.image_gray = f.gray;
    body.image_rgba = f.rgba;
    body.color_rgb = f.color_rgb;
  }
  return body;
}

function cameraReady() {
  return Boolean(mediaStream && cam.videoWidth > 0 && cam.videoHeight > 0);
}

async function refreshState() {
  const s = await api("/api/baby/state?lite=1");
  dormant = Boolean(s.dormant ?? dormant);
  applyBrain(s.brain, s.last_moment);
  if (s.consciousness_stream?.length) renderConsciousnessStream(s.consciousness_stream);
  const objs = Object.keys(s.visual_binder?.object_names || {}).length;
  const dlg = s.dialogue_count ?? (s.dialogue_pairs || []).length;
  const mode = dormant ? "sonno" : "veglia";
  setStatus(`${mode} · ${dlg} dialoghi · ${objs} oggetti · ${s.syllables_known ?? 0} sillabe`);
  if (s.dialogue?.length) renderDialogue(s.dialogue);
}

async function startSenses() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    cam.srcObject = mediaStream;
    await cam.play();
  } catch (_) {
    statusEl.textContent = "camera non disponibile — concedi permesso";
  }
}

async function ensureMicPermission() {
  try {
    const a = await navigator.mediaDevices.getUserMedia({ audio: true });
    a.getTracks().forEach((t) => t.stop());
  } catch (_) {
    voiceHint.textContent = "concedi accesso al microfono";
  }
}

function setupMicRecognition() {
  if (!SpeechRecognition || micRecognition) return;
  micRecognition = new SpeechRecognition();
  micRecognition.lang = LOCALE;
  micRecognition.continuous = false;
  micRecognition.interimResults = false;
  micRecognition.onresult = (ev) => {
    const last = ev.results[ev.results.length - 1];
    if (!last?.isFinal) return;
    const phrase = last[0].transcript.trim();
    if (phrase) sendPhrase(phrase, "caregiver");
  };
  micRecognition.onend = () => {
    micActive = false;
    btnMic?.classList.remove("listening");
    // Auto-riavvia dopo ogni utterance — sempre in ascolto quando non parla
    if (born && userActivated && !isSelfSpeaking) {
      setTimeout(() => beginMicListen(), 400);
    }
  };
  micRecognition.onerror = (e) => {
    micActive = false;
    btnMic?.classList.remove("listening");
    if (e.error === "not-allowed") {
      voiceHint.textContent = "microfono negato — controlla permessi";
    } else if (e.error !== "aborted") {
      idleHint();
    }
  };
}

function beginMicListen() {
  if (!born || isSelfSpeaking || Date.now() < selfSpeakGuardUntil) return;
  setupMicRecognition();
  if (!micRecognition) {
    voiceHint.textContent = "riconoscimento vocale non supportato";
    return;
  }
  ensureMicPermission();
  micActive = true;
  btnMic?.classList.add("listening");
  voiceHint.textContent = "parla ora…";
  try {
    micRecognition.start();
  } catch (_) {
    micActive = false;
    btnMic?.classList.remove("listening");
  }
}

function endMicListen() {
  if (!micRecognition || !micActive) return;
  try {
    micRecognition.stop();
  } catch (_) { /* */ }
  micActive = false;
  btnMic?.classList.remove("listening");
}

async function look() {
  if (!born || !cameraReady()) return;
  voiceHint.textContent = "guarda…";
  handBtn((b) => b.classList.add("looking"));
  const r = await api("/api/baby/look", visionBody());
  handBtn((b) => b.classList.remove("looking"));
  if (r.recognized) {
    voiceHint.textContent = `riconosce · ${r.recognized} (${Math.round((r.confidence || 0) * 100)}%)`;
  } else {
    voiceHint.textContent = "osserva in silenzio";
  }
  applyMoment(r.moment);
  refreshState();
}

async function continuousFlow() {
  if (!born || flowBusy || phraseBusy || micActive || isSelfSpeaking) return;
  flowBusy = true;
  try {
    const r = await api("/api/baby/flow", visionBody());
    dormant = Boolean(r.dormant ?? dormant);
    if (r.brain || r.moment) applyBrain(r.brain || r.moment?.brain, r.moment);
    if (r.moment) {
      const th = r.moment.thought;
      const dream = r.moment.dream;
      if (dream?.active && dream.content) {
        thoughtLine.hidden = false;
        thoughtLine.textContent = `sogna · ${dream.content.slice(0, 80)}`;
      } else if (th?.themes?.length) {
        thoughtLine.hidden = false;
        thoughtLine.textContent = `pensa · ${th.themes.slice(0, 6).join(" · ")}`;
      }
    }
    if (r.consciousness?.events?.length) {
      appendMindEvents(r.consciousness.events);
      consciousnessSeq = Math.max(consciousnessSeq, r.consciousness.seq || 0);
    }
    if (r.moment?.spoke) applyMoment(r.moment);
  } catch (_) { /* offline */ }
  finally {
    flowBusy = false;
  }
}

function startContinuousFlow() {
  if (flowTimer) clearInterval(flowTimer);
  flowTimer = setInterval(continuousFlow, FLOW_MS);
}

function onUserActivate() {
  unlockVoice();
  // Al primo tocco/click, avvia subito il microfono se Baby è pronto
  if (born && !micActive && !isSelfSpeaking) {
    setTimeout(() => beginMicListen(), 300);
  }
}

function onHandDown(e) {
  e.preventDefault();
  unlockVoice();
  isHolding = true;
  holdStarted = false;
  holdTimer = setTimeout(() => {
    if (isHolding) {
      holdStarted = true;
      teachingFocus = true;
      handBtn((b) => b.classList.add("teaching"));
      voiceHint.textContent = "nomina ciò che vedi…";
    }
  }, HOLD_MS);
}

function onHandUp(e) {
  e.preventDefault();
  isHolding = false;
  clearTimeout(holdTimer);
  if (holdStarted) {
    holdStarted = false;
    teachingFocus = false;
    handBtn((b) => b.classList.remove("teaching"));
    idleHint();
  } else {
    look();
  }
}

// Mic: click-to-toggle (non più push-to-talk).
// Baby è un organismo — ascolta quando parli, non ha bisogno di un bottone da tenere premuto.
function onMicDown(e) {
  e.preventDefault();
  unlockVoice();
  if (micActive) {
    endMicListen();
  } else {
    beginMicListen();
  }
}

// Auto-riavvia il microfono dopo che Baby ha finito di parlare
function restartMicAfterSpeak() {
  if (!born || !userActivated) return;
  setTimeout(() => {
    if (!micActive && !isSelfSpeaking && Date.now() > selfSpeakGuardUntil) {
      beginMicListen();
    }
  }, 600);
}

if (btnHand) {
  btnHand.addEventListener("mousedown", onHandDown);
  btnHand.addEventListener("mouseup", onHandUp);
  btnHand.addEventListener("mouseleave", () => { if (isHolding) onHandUp({ preventDefault() {} }); });
  btnHand.addEventListener("touchstart", onHandDown, { passive: false });
  btnHand.addEventListener("touchend", onHandUp, { passive: false });
  btnHand.addEventListener("touchcancel", onHandUp, { passive: false });
}

if (btnMic) {
  btnMic.addEventListener("mousedown", onMicDown);
  btnMic.addEventListener("touchstart", onMicDown, { passive: false });
  // Click-to-toggle: no mouseup/touchend stop
}

document.body.addEventListener("touchstart", onUserActivate, { passive: true });
document.body.addEventListener("click", onUserActivate);

setInterval(pollMind, 4000);

if (chatForm) {
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    unlockVoice();
    sendChat();
  });
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    if (flowTimer) clearInterval(flowTimer);
    endMicListen();
  } else if (born) {
    startContinuousFlow();
    setTimeout(continuousFlow, 800);
  }
});

(async () => {
  setStatus("sta caricando…");
  const senses = startSenses();
  await wakeOrBirth();
  idleHint();
  const boot = () => {
    if (!born) return;
    startContinuousFlow();
    setTimeout(continuousFlow, 800);
  };
  cam.addEventListener("loadeddata", boot, { once: true });
  if (cameraReady()) boot();
  await senses;
  if (born && cameraReady()) boot();
})();
