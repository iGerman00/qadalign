const SR = 16000, NFFT = 512, WIN = 400, HOP = 160, NMEL = 80, FRAME = 0.04;

const PHONES = ["AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY","F","G","HH",
  "IH","IY","JH","K","L","M","N","NG","OW","OY","P","R","S","SH","T","TH","UH","UW","V","W","Y","Z","ZH"];
const P2ID = {};
PHONES.forEach((p, i) => P2ID[p] = i + 1);

const state = { session: null, dict: null };
const $ = id => document.getElementById(id);

// ---------------------------------------------------------------- dictionary
// Order the probes by where the page is being served from, so the first one
// usually hits and the console stays clean.
const DICT_PATHS = (() => {
  const dir = location.pathname.replace(/[^/]*$/, "");
  const nested = dir.endsWith("/web/");                 // repo root, page in web/
  const local = ["data/cmudict.txt.gz", "cmudict.txt.gz"];   // web/ as document root
  const up = ["../qadalign/data/cmudict.txt.gz", "qadalign/data/cmudict.txt.gz"];
  return nested ? [...up, ...local] : [...local, ...up];
})();

async function loadDict() {
  if (state.dict) return state.dict;
  let res = null;
  for (const p of DICT_PATHS) {
    try {
      const r = await fetch(p);
      if (r.ok) { res = r; break; }
    } catch (e) { /* try the next one */ }
  }
  if (!res) {
    throw new Error("pronunciation dictionary not found. Tried " +
      DICT_PATHS.join(", ") + ". Serve the repository root: python3 -m http.server 8000, " +
      "then open http://localhost:8000/web/");
  }
  const text = await new Response(res.body.pipeThrough(new DecompressionStream("gzip"))).text();
  const map = new Map();
  for (const line of text.split("\n")) {
    const tab = line.indexOf("\t");
    if (tab > 0) map.set(line.slice(0, tab), line.slice(tab + 1).split(" "));
  }
  state.dict = map;
  return map;
}

// ---------------------------------------------------------------- lyrics
// The text is kept exactly as typed: word tokens keep their case and
// punctuation, and every line keeps its original string. Section tags are not
// sung, so they contribute no phones, but they stay in the output.
function parseLyrics(text, dict) {
  const lines = [];
  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim()) continue;
    const sung = raw.replace(/\[[^\]]*\]/g, " ");
    const words = [];
    for (const tok of sung.split(/\s+/)) {
      if (!tok || !/[A-Za-z0-9]/.test(tok) || /^\*+$/.test(tok)) continue;
      const key = tok.toLowerCase().replace(/[()]/g, "").replace(/[^a-z']/g, "").replace(/’/g, "'");
      const ph = dict.get(key) || dict.get(key.split("'")[0]);
      if (ph) words.push({ text: tok, phones: ph });
    }
    lines.push({ text: raw.trim(), words });
  }
  return lines;
}

// ---------------------------------------------------------------- features
const hann = (() => {
  const w = new Float32Array(WIN);
  for (let i = 0; i < WIN; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / WIN);
  return w;
})();

const MELFB = (() => {
  const hz2mel = f => 2595 * Math.log10(1 + f / 700);
  const mel2hz = m => 700 * (Math.pow(10, m / 2595) - 1);
  const nf = NFFT / 2 + 1;
  const freqs = new Float32Array(nf);
  for (let i = 0; i < nf; i++) freqs[i] = i * (SR / 2) / (nf - 1);
  const lo = hz2mel(20), hi = hz2mel(8000);
  const edges = [];
  for (let i = 0; i < NMEL + 2; i++) edges.push(mel2hz(lo + (hi - lo) * i / (NMEL + 1)));
  const fb = [];
  for (let m = 0; m < NMEL; m++) {
    const a = edges[m], c = edges[m + 1], b = edges[m + 2];
    const row = new Float32Array(nf);
    let sum = 0;
    for (let j = 0; j < nf; j++) {
      const v = Math.max(0, Math.min((freqs[j] - a) / (c - a), (b - freqs[j]) / (b - c)));
      row[j] = v; sum += v;
    }
    if (sum === 0) {
      let best = 0, bd = Infinity;
      for (let j = 0; j < nf; j++) {
        const d = Math.abs(freqs[j] - c);
        if (d < bd) { bd = d; best = j; }
      }
      row[best] = 1;
    }
    fb.push(row);
  }
  return fb;
})();

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k], ai = im[i + k];
        const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ar + br; im[i + k] = ai + bi;
        re[i + k + len / 2] = ar - br; im[i + k + len / 2] = ai - bi;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

function logmel(x) {
  const n = x.length, T = 1 + Math.floor(n / HOP), pad = NFFT >> 1, off = (NFFT - WIN) >> 1;
  const re = new Float32Array(NFFT), im = new Float32Array(NFFT);
  const feats = new Float32Array(T * NMEL);
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < NFFT; i++) {
      let p = t * HOP + i - pad;
      p = p < 0 ? -p : (p >= n ? 2 * n - 2 - p : p);
      const wi = i - off;
      re[i] = (wi >= 0 && wi < WIN) ? x[p] * hann[wi] : 0;
      im[i] = 0;
    }
    fft(re, im);
    for (let m = 0; m < NMEL; m++) {
      const row = MELFB[m];
      let s = 0;
      for (let j = 0; j <= NFFT / 2; j++) s += row[j] * (re[j] * re[j] + im[j] * im[j]);
      feats[t * NMEL + m] = Math.log(s + 1e-10);
    }
  }
  for (let m = 0; m < NMEL; m++) {
    let sum = 0;
    for (let t = 0; t < T; t++) sum += feats[t * NMEL + m];
    const mean = sum / T;
    let v = 0;
    for (let t = 0; t < T; t++) { const d = feats[t * NMEL + m] - mean; v += d * d; }
    const sd = Math.sqrt(v / Math.max(1, T - 1)) + 1e-5;
    for (let t = 0; t < T; t++) feats[t * NMEL + m] = (feats[t * NMEL + m] - mean) / sd;
  }
  return { feats, T };
}

// ---------------------------------------------------------------- inference
async function posteriors(x, onProgress) {
  const total = x.length / SR;
  const nOut = Math.floor(total / FRAME) + 2;
  const acc = new Float32Array(nOut * 40), sacc = new Float32Array(nOut), wsum = new Float32Array(nOut);
  let t = 0, done = 0, windows = 1;
  for (let a = 0; a + 30 < total; a += 15) windows++;
  while (t < total - 1) {
    const b = Math.min(total, t + 30);
    const chunk = x.subarray(Math.floor(t * SR), Math.floor(b * SR));
    if (chunk.length < 4000) break;
    const { feats, T } = logmel(chunk);
    const mask = new BigInt64Array(T).fill(1n);
    const out = await state.session.run({
      features: new ort.Tensor("float32", feats, [1, T, NMEL]),
      mask: new ort.Tensor("int64", mask, [1, T]),
    });
    const logits = out.logits.data, sung = out.sung.data, F = out.logits.dims[1];
    const lp = new Float32Array(F * 40);
    for (let f = 0; f < F; f++) {
      let mx = -Infinity;
      for (let c = 0; c < 40; c++) mx = Math.max(mx, logits[f * 40 + c]);
      let sum = 0;
      for (let c = 0; c < 40; c++) sum += Math.exp(logits[f * 40 + c] - mx);
      for (let c = 0; c < 40; c++) lp[f * 40 + c] = logits[f * 40 + c] - mx - Math.log(sum);
    }
    const base = Math.round(t / FRAME);
    const w = new Float32Array(F);
    for (let f = 0; f < F; f++) w[f] = 0.5 - 0.5 * Math.cos(2 * Math.PI * (f + 1) / (F + 1));
    for (let f = 0; f < F; f++) {
      const idx = base + f;
      if (idx >= nOut) break;
      for (let c = 0; c < 40; c++) acc[idx * 40 + c] += lp[f * 40 + c] * w[f];
      sacc[idx] += sung[f] * w[f];
      wsum[idx] += w[f];
    }
    done++;
    if (onProgress) onProgress(done, windows);
    if (b >= total) break;
    t += 15;
  }
  const den = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) den[i] = Math.max(wsum[i], 1e-6);
  for (let i = 0; i < nOut; i++) {
    for (let c = 0; c < 40; c++) acc[i * 40 + c] /= den[i];
    sacc[i] /= den[i];
  }
  return { lp: acc, sung: sacc, n: nOut };
}

// ---------------------------------------------------------------- decode
function boxSmooth(a, k) {
  const n = a.length, out = new Float32Array(n), half = (k - 1) >> 1;
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half), hi = Math.min(n - 1, i + k - 1 - half);
    let s = 0;
    for (let j = lo; j <= hi; j++) s += a[j];
    out[i] = s / k;
  }
  return out;
}

function percentile(a, p) {
  const s = Float32Array.from(a).sort();
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
}

function sustainedOnset(sm, gate, minRun) {
  let run = 0;
  for (let i = 0; i < sm.length; i++) {
    run = sm[i] > gate ? run + 1 : 0;
    if (run >= minRun) return i - minRun + 1;
  }
  return -1;
}

function viterbi(lp, T, targets, skipCost, delCost) {
  const L = targets.length, S = 2 * L + 1;
  const ext = new Int32Array(S);
  for (let i = 0; i < L; i++) ext[1 + 2 * i] = targets[i];
  const NEG = -1e30;
  let alpha = new Float64Array(S).fill(NEG), next = new Float64Array(S);
  alpha[0] = lp[0];
  if (L) alpha[1] = lp[ext[1]];
  const repeatOk = new Uint8Array(S);
  for (let s = 2; s < S; s++) repeatOk[s] = (ext[s] !== 0 && ext[s] !== ext[s - 2]) ? 1 : 0;
  const back = new Int8Array(T * S);
  const a1 = new Float64Array(S), a2 = new Float64Array(S), a3 = new Float64Array(S);
  for (let t = 1; t < T; t++) {
    const row = t * 40, brow = t * S;
    for (let s = 0; s < S; s++) {
      a1[s] = s > 0 ? alpha[s - 1] : NEG;
      a2[s] = (s > 1 && repeatOk[s]) ? alpha[s - 2] : NEG;
      a3[s] = (s > 1 && L > 1) ? alpha[s - 2] - delCost : NEG;
    }
    for (let s = 0; s < S; s++) {
      let best = alpha[s], code = 0;
      if (a1[s] > best) { best = a1[s]; code = 1; }
      if (a2[s] > best) { best = a2[s]; code = 2; }
      if (a3[s] > best) { best = a3[s]; code = 3; }
      const emit = best + lp[row + ext[s]];
      const skip = alpha[s] - skipCost;
      if (skip > emit) { next[s] = skip; back[brow + s] = 4; }
      else { next[s] = emit; back[brow + s] = code; }
    }
    const tmp = alpha; alpha = next; next = tmp;
  }
  let s = S - 1;
  if (L && alpha[S - 2] > alpha[s]) s = S - 2;
  const path = new Int32Array(T), skipped = new Uint8Array(T);
  path[T - 1] = s;
  for (let t = T - 1; t > 0; t--) {
    const code = back[t * S + s];
    if (code === 4) skipped[t] = 1; else s -= code;
    path[t - 1] = s;
  }
  return { path, ext, skipped };
}

function wordTimes(path, ext, skipped, words) {
  const spans = [], S = ext.length;
  for (let s = 1; s < S; s += 2) {
    let first = -1, last = -1;
    for (let t = 0; t < path.length; t++) {
      if (path[t] === s && !skipped[t]) { if (first < 0) first = t; last = t; }
    }
    spans.push(first < 0 ? null : [first, last + 1]);
  }
  const out = [];
  let pi = 0;
  for (const w of words) {
    const chunk = spans.slice(pi, pi + w.phones.length);
    pi += w.phones.length;
    const f = chunk.filter(Boolean);
    if (!f.length) out.push({ text: w.text, start: null, end: null });
    else out.push({ text: w.text,
      start: Math.min(...f.map(x => x[0])) * FRAME + 0.02,
      end: Math.max(...f.map(x => x[1])) * FRAME + 0.02 });
  }
  for (let i = 0; i < out.length; i++) {
    if (out[i].start === null) {
      const nxt = out.slice(i + 1).find(x => x.start !== null);
      const prv = out.slice(0, i).reverse().find(x => x.end !== null);
      out[i].start = nxt ? nxt.start : (prv ? prv.end : 0);
      out[i].end = prv ? prv.end : (nxt ? nxt.start : 0);
    }
  }
  return out;
}

// ---------------------------------------------------------------- run
async function run(file, lyricsText, onProgress) {
  const notify = onProgress || (() => {});
  if (!state.session) {
    notify("model");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
    state.session = await ort.InferenceSession.create("qadalign.onnx", { executionProviders: ["wasm"] });
  }
  notify("dictionary");
  const dict = await loadDict();
  const parsed = parseLyrics(lyricsText, dict);
  const flat = parsed.flatMap(l => l.words);
  if (!flat.length) throw new Error("no pronounceable words in the lyrics");

  notify("audio");
  const x = await toMono16k(file);
  const { lp, sung, n } = await posteriors(x, (i, t) => notify("inference", i, t));

  const targets = [];
  for (const w of flat) for (const p of w.phones) targets.push(P2ID[p]);

  const sm = boxSmooth(sung, 25);
  const gate = Math.max(0.3, 0.5 * percentile(sm, 0.9));
  const onset = sustainedOnset(sm, gate, 25);
  if (onset > 0) for (let t = 0; t < onset; t++) for (let c = 1; c < 40; c++) lp[t * 40 + c] = -1e9;

  const { path, ext, skipped } = viterbi(lp, n, targets, 1.8, 4.0);
  const words = wordTimes(path, ext, skipped, flat);

  const out = [];
  let k = 0, prevEnd = 0;
  for (const ln of parsed) {
    const chunk = words.slice(k, k + ln.words.length);
    k += ln.words.length;
    if (!chunk.length) {
      // a line with nothing singable, such as a section tag: keep it, untimed
      out.push({ text: ln.text, start: prevEnd, end: prevEnd, words: [] });
      continue;
    }
    out.push({ text: ln.text, start: chunk[0].start, end: chunk[chunk.length - 1].end,
      words: chunk.map(w => ({ text: w.text, start: w.start, end: w.end })) });
    prevEnd = out[out.length - 1].end;
  }

  for (const ln of out) {
    for (let j = 0; j < ln.words.length - 1; j++) {
      ln.words[j].end = Math.max(ln.words[j].end, ln.words[j + 1].start);
    }
    if (ln.words.length) ln.end = Math.max(ln.end, ln.words[ln.words.length - 1].end);
  }
  for (let i = 0; i < out.length; i++) {
    const ln = out[i];
    const limit = i + 1 < out.length ? out[i + 1].start - 0.05 : Infinity;
    let end = ln.end;
    if (ln.words.length && limit > end) {
      const a = Math.floor(end / FRAME), b = Math.min(Math.floor(Math.min(limit, end + 1.0) / FRAME), sm.length);
      let last = -1;
      for (let t = a; t < b; t++) if (sm[t] > gate) last = t;
      if (last >= 0) end = Math.max(end, last * FRAME);
    }
    ln.end = ln.words.length ? Math.min(end + 0.10, limit) : end;
  }
  for (const ln of out) {
    ln.start = Math.round(ln.start * 100) / 100;
    ln.end = Math.round(Math.max(ln.end, ln.start) * 100) / 100;
    for (const w of ln.words) {
      w.start = Math.round(w.start * 100) / 100;
      w.end = Math.round(Math.max(w.end, w.start + 0.02) * 100) / 100;
    }
  }
  return { lines: out };
}

async function toMono16k(file) {
  const buf = await file.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audio = await ctx.decodeAudioData(buf);
  await ctx.close();
  const frames = Math.ceil(audio.duration * SR);
  const off = new OfflineAudioContext(1, frames, SR);
  const src = off.createBufferSource();
  src.buffer = audio;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

window.qad = { run };

// ---------------------------------------------------------------- player
let playerUrl = null;

function setPlayer(file) {
  if (playerUrl) URL.revokeObjectURL(playerUrl);
  playerUrl = URL.createObjectURL(file);
  const p = $("player");
  p.src = playerUrl;
  p.load();
  $("player-wrap").hidden = false;
}

// ---------------------------------------------------------------- preview
let preview = [];

function showPreview(res) {
  const p = $("player");
  const box = $("lines");
  box.textContent = "";
  preview = res.lines.map(ln => {
    const el = document.createElement("div");
    el.className = "line";
    el.dataset.s = ln.start;
    const words = ln.words.map(w => {
      const s = document.createElement("span");
      s.className = "w";
      s.textContent = w.text + " ";
      s.dataset.s = w.start;
      s.dataset.e = w.end;
      el.appendChild(s);
      return s;
    });
    if (!words.length) el.textContent = ln.text;
    box.appendChild(el);
    return { el, words, start: ln.start };
  });

  $("lines-wrap").hidden = false;
  p.ontimeupdate = syncPreview;
  syncPreview();
}

function syncPreview() {
  const t = $("player").currentTime;
  let cur = -1;
  for (let i = 0; i < preview.length; i++) {
    if (t >= preview[i].start) cur = i; else break;
  }
  for (let i = 0; i < preview.length; i++) {
    const ln = preview[i];
    ln.el.classList.toggle("cur", i === cur);
    for (const w of ln.words) {
      w.classList.toggle("on", i === cur && t >= +w.dataset.s && t < +w.dataset.e);
    }
  }
  if (cur >= 0) {
    const box = $("lines"), el = preview[cur].el;
    if (el.offsetTop < box.scrollTop || el.offsetTop > box.scrollTop + box.clientHeight - 40) {
      box.scrollTop = el.offsetTop - box.clientHeight / 2;
    }
  }
}

// ---------------------------------------------------------------- downloads
let last = null;

function vizzy(kind) {
  if (!last) return;
  const rows = kind === "words"
    ? last.res.lines.flatMap(l => l.words.map(w => ({ text: w.text, start: w.start, end: w.end })))
    : last.res.lines.map(l => ({ text: l.text, start: l.start, end: l.end }));
  const suffix = kind === "words" ? "_lyrics_wordlevel.json" : "_lyrics_linelevel.json";
  const blob = new Blob([JSON.stringify(rows, null, 1)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = last.base + suffix;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------- ui
$("go").onclick = async () => {
  const file = $("audio").files[0];
  if (!file) { $("status").textContent = "choose an audio file first"; return; }
  const t0 = performance.now();
  $("go").disabled = true;
  $("out").hidden = true;
  $("out").textContent = "";
  $("downloads").hidden = true;
  $("status").className = "";
  try {
    if (!$("player").src) setPlayer(file);
    const res = await run(file, $("lyrics").value, (stage, i, n) => {
      if (stage === "model") $("status").textContent = "loading model";
      else if (stage === "dictionary") $("status").textContent = "loading dictionary";
      else if (stage === "audio") $("status").textContent = "decoding audio";
      else if (stage === "inference") {
        const pct = n ? Math.round((100 * i) / n) : 0;
        $("status").textContent = `aligning ${pct}%, window ${i} of ${n}`;
      }
    });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    $("status").textContent = `${res.lines.length} lines in ${secs}s`;
    last = { res, base: file.name.replace(/\.[^.]+$/, "") };
    $("downloads").hidden = false;
    showPreview(res);
    $("out").hidden = false;
    $("out").textContent = JSON.stringify(res, null, 1);
  } catch (e) {
    $("status").className = "error";
    $("status").textContent = "error: " + e.message;
    console.error(e);
  } finally {
    $("go").disabled = false;
  }
};

$("dl-words").onclick = () => vizzy("words");
$("dl-lines").onclick = () => vizzy("lines");

$("audio").onchange = () => {
  const f = $("audio").files[0];
  if (f) setPlayer(f);
};

// ---------------------------------------------------------------- lyric tools
function editSelection(fn) {
  const ta = $("lyrics");
  const a = ta.selectionStart, b = ta.selectionEnd;
  const whole = a === b;
  const from = whole ? 0 : a, to = whole ? ta.value.length : b;
  const text = fn(ta.value.slice(from, to));
  ta.value = ta.value.slice(0, from) + text + ta.value.slice(to);
  ta.focus();
  ta.setSelectionRange(from, from + text.length);
}

$("tool-split").onclick = () => editSelection(t => t.replace(/[ \t]+/g, "\n"));
$("tool-join").onclick = () => editSelection(t => t.replace(/\s*\n\s*/g, " "));
$("tool-tags").onclick = () => {
  const ta = $("lyrics");
  ta.value = ta.value.split(/\r?\n/).filter(l => !/\[[^\]]*\]/.test(l)).join("\n");
  ta.focus();
};
