const SR = 16000, NFFT = 512, WIN = 400, HOP = 160, NMEL = 80, FRAME = 0.04;

const PHONES = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
  "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"];
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

// ---------------------------------------------------------------- g2p
// Dictionary -> compounds, digits, morphology -> letter to sound.
const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const ORD = {"zero": "zeroth", "one": "first", "two": "second", "three": "third", "four": "fourth", "five": "fifth", "six": "sixth", "seven": "seventh", "eight": "eighth", "nine": "ninth", "ten": "tenth", "eleven": "eleventh", "twelve": "twelfth", "thirteen": "thirteenth", "fourteen": "fourteenth", "fifteen": "fifteenth", "sixteen": "sixteenth", "seventeen": "seventeenth", "eighteen": "eighteenth", "nineteen": "nineteenth", "twenty": "twentieth", "thirty": "thirtieth", "forty": "fortieth", "fifty": "fiftieth", "sixty": "sixtieth", "seventy": "seventieth", "eighty": "eightieth", "ninety": "ninetieth", "hundred": "hundredth", "thousand": "thousandth"};
const SIBILANT = new Set(["CH", "JH", "S", "SH", "Z", "ZH"]);
const VOICELESS = new Set(["CH", "F", "K", "P", "S", "SH", "T", "TH"]);
const VOICED = new Set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "D", "DH", "EH", "ER", "EY", "G", "IH", "IY", "L", "M", "N", "NG", "OW", "OY", "R", "UH", "UW", "V", "W", "Y", "Z"]);
const VOWELS = new Set(["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]);
const C_SOFT = "eiy";
const MAGIC = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW"};
const ADD = [["n't", ["AH", "N", "T"]], ["'ll", ["L"]], ["'re", ["R"]], ["'ve", ["V"]], ["'m", ["M"]], ["'s", "plural"], ["'d", "past"], ["'", []], ["ings", ["IH", "NG", "Z"]], ["less", ["L", "AH", "S"]], ["ness", ["N", "AH", "S"]], ["ment", ["M", "AH", "N", "T"]], ["able", ["AH", "B", "AH", "L"]], ["ible", ["AH", "B", "AH", "L"]], ["est", ["IH", "S", "T"]], ["ing", ["IH", "NG"]], ["in'", ["IH", "NG"]], ["ed", "past"], ["es", "plural"], ["er", ["ER"]], ["ly", ["L", "IY"]], ["s", "plural"], ["in", ["IH", "NG"]]];
const PREFIX = ["un", "re", "dis", "mis", "pre", "non", "over", "under", "out", "sub", "super", "semi", "anti", "multi", "micro", "post", "co"];
const LTS = [["ough", ["AO"]], ["augh", ["AO"]], ["eigh", ["EY"]], ["tion", ["SH", "AH", "N"]], ["sion", ["ZH", "AH", "N"]], ["ight", ["AY", "T"]], ["cial", ["SH", "AH", "L"]], ["tial", ["SH", "AH", "L"]], ["ture", ["CH", "ER"]], ["sure", ["ZH", "ER"]], ["kn", ["N"]], ["wr", ["R"]], ["igh", ["AY"]], ["air", ["EH", "R"]], ["eer", ["IH", "R"]], ["ear", ["IH", "R"]], ["oor", ["AO", "R"]], ["our", ["AW", "ER"]], ["ore", ["AO", "R"]], ["are", ["EH", "R"]], ["ere", ["IH", "R"]], ["ire", ["AY", "R"]], ["ure", ["Y", "UH", "R"]], ["dge", ["JH"]], ["tch", ["CH"]], ["sch", ["S", "K"]], ["ous", ["AH", "S"]], ["ble", ["B", "AH", "L"]], ["tle", ["T", "AH", "L"]], ["cle", ["K", "AH", "L"]], ["gle", ["G", "AH", "L"]], ["ple", ["P", "AH", "L"]], ["dle", ["D", "AH", "L"]], ["th", ["TH"]], ["sh", ["SH"]], ["ch", ["CH"]], ["ph", ["F"]], ["wh", ["W"]], ["ck", ["K"]], ["ng", ["NG"]], ["qu", ["K", "W"]], ["gh", []], ["ee", ["IY"]], ["ea", ["IY"]], ["ei", ["EY"]], ["ey", ["EY"]], ["ie", ["IY"]], ["ai", ["EY"]], ["ay", ["EY"]], ["oa", ["OW"]], ["oe", ["OW"]], ["oi", ["OY"]], ["oy", ["OY"]], ["oo", ["UW"]], ["ou", ["AW"]], ["ow", ["OW"]], ["au", ["AO"]], ["aw", ["AO"]], ["ue", ["UW"]], ["ui", ["UW"]], ["ew", ["UW"]], ["eu", ["UW"]], ["ar", ["AA", "R"]], ["or", ["AO", "R"]], ["er", ["ER"]], ["ir", ["ER"]], ["ur", ["ER"]], ["ss", ["S"]], ["ll", ["L"]], ["tt", ["T"]], ["pp", ["P"]], ["mm", ["M"]], ["nn", ["N"]], ["dd", ["D"]], ["ff", ["F"]], ["rr", ["R"]], ["bb", ["B"]], ["gg", ["G"]], ["cc", ["K"]], ["zz", ["Z"]], ["a", ["AE"]], ["e", ["EH"]], ["i", ["IH"]], ["o", ["AA"]], ["u", ["AH"]], ["y", ["IY"]], ["b", ["B"]], ["c", ["K"]], ["d", ["D"]], ["f", ["F"]], ["g", ["G"]], ["h", ["HH"]], ["j", ["JH"]], ["k", ["K"]], ["l", ["L"]], ["m", ["M"]], ["n", ["N"]], ["p", ["P"]], ["q", ["K"]], ["r", ["R"]], ["s", ["S"]], ["t", ["T"]], ["v", ["V"]], ["w", ["W"]], ["x", ["K", "S"]], ["z", ["Z"]]];

function numberWords(digits) {
  const n = digits.length;
  if (!n) return [];
  if (n <= 2) {
    const v = parseInt(digits, 10);
    if (v < 20) return [ONES[v]];
    const t = Math.floor(v / 10), u = v % 10;
    return u === 0 ? [TENS[t]] : [TENS[t], ONES[u]];
  }
  if (n === 4 && digits.slice(2) === "00") return [ONES[+digits[0]], "thousand"];
  if (n === 4 && (digits.slice(0, 2) === "19" || digits.slice(0, 2) === "20")) {
    const first = numberWords(digits.slice(0, 2)), rest = digits.slice(2);
    if (rest[0] === "0") return first.concat(["oh", ONES[+rest[1]]]);
    return first.concat(numberWords(rest));
  }
  return [...digits].map(c => (c === "0" ? "oh" : ONES[+c]));
}

function ordinal(words) {
  if (!words.length || !(words[words.length - 1] in ORD)) return words;
  return words.slice(0, -1).concat([ORD[words[words.length - 1]]]);
}

function dictLookup(w, dict) {
  const e = dict.get(w);
  if (!e) return null;
  const ph = e.filter(p => P2ID[p]);
  return ph.length ? ph : null;
}

function plural(ph) {
  const last = ph.length ? ph[ph.length - 1] : "S";
  if (SIBILANT.has(last)) return ph.concat(["IH", "Z"]);
  return ph.concat([VOICED.has(last) ? "Z" : "S"]);
}

function past(ph) {
  const last = ph.length ? ph[ph.length - 1] : "T";
  if (last === "T" || last === "D") return ph.concat(["IH", "D"]);
  return ph.concat([VOICELESS.has(last) ? "T" : "D"]);
}

function applyAdd(base, add) {
  if (add === "plural") return plural(base);
  if (add === "past") return past(base);
  return base.concat(add || []);
}

function stemForms(stem) {
  const out = [stem + "e", stem];
  if (stem.length > 2) {
    if (stem[stem.length - 1] === stem[stem.length - 2]) out.push(stem.slice(0, -1));
    out.push(stem.slice(0, -1));
  }
  return out;
}

function resolve(w, dict, depth) {
  const ph = dictLookup(w, dict);
  if (ph) return ph;
  if (depth >= 3 || w.length < 4) return null;
  for (const [suf, add] of ADD) {
    if (!w.endsWith(suf) || w.length - suf.length < 2) continue;
    for (const cand of stemForms(w.slice(0, w.length - suf.length))) {
      const base = resolve(cand, dict, depth + 1);
      if (base) return applyAdd(base, add);
    }
  }
  for (const pre of PREFIX) {
    const rest = w.slice(pre.length);
    if (!w.startsWith(pre) || rest.length < 3) continue;
    const base = resolve(rest, dict, depth + 1);
    if (base) return base;
  }
  return null;
}

// Two dictionary words written as one like lovestruck, raindrop, homesick.
function glued(w, dict) {
  if (w.length < 6) return null;
  for (let cut = w.length - 3; cut > 2; cut--) {
    const a = dictLookup(w.slice(0, cut), dict), b = dictLookup(w.slice(cut), dict);
    if (a && b) return a.concat(b);
  }
  return null;
}

// Letter to sound
function lts(w) {
  const out = [];
  let i = 0;
  const n = w.length;
  while (i < n) {
    const ch = w[i];
    if (ch === "'") { i++; continue; }
    // a final e is silent, and it lengthens the vowel two letters back
    if (ch === "e" && i === n - 1 && n > 2) { i++; continue; }
    if (MAGIC[ch] && i + 2 === n - 1 && w[i + 2] === "e" && !"hrw".includes(w[i + 1])) {
      out.push(MAGIC[ch]); i++; continue;
    }
    let hit = false;
    for (const [src, ph] of LTS) {
      if (!w.startsWith(src, i)) continue;
      let emit = ph;
      if (src === "c" && i + 1 < n && C_SOFT.includes(w[i + 1])) emit = ["S"];
      else if (src === "g" && i + 1 < n && C_SOFT.includes(w[i + 1])) emit = ["JH"];
      out.push(...emit);
      i += src.length;
      hit = true;
      break;
    }
    if (!hit) i++;
  }
  if (out.length && !out.some(p => VOWELS.has(p))) out.push("AH");
  return out;
}

// Phones for a token whose pronunciation comes from digits.
function digitPhones(tok, dict) {
  const digits = tok.replace(/[^0-9A-Za-z]/g, "");
  let words;
  if (/^\d+$/.test(digits)) {
    words = numberWords(digits);
  } else {
    const m = /^(\d+)(st|nd|rd|th)$/.exec(digits.toLowerCase());
    if (!m) return null;
    words = ordinal(numberWords(m[1]));
  }
  const out = [];
  for (const word of words) {
    const ph = dictLookup(word, dict) || lts(word);
    if (!ph.length) return null;
    out.push(...ph);
  }
  return out;
}

function resolvePhones(word, dict, depth) {
  const raw = word.trim();
  depth = depth || 0;
  // hyphenated, slashed and ampersand compounds, part by part
  if (/[-/&+]/.test(raw)) {
    const parts = raw.split(/[-/&+]/).filter(p => /[A-Za-z0-9]/.test(p));
    if (parts.length > 1) {
      const out = [];
      let ok = true;
      for (const part of parts) {
        const sub = resolvePhones(part, dict, depth);
        if (!sub) { ok = false; break; }
        out.push(...sub);
      }
      if (ok && out.length) return out;
    }
  }
  if (/\d/.test(raw)) {
    const ph = digitPhones(raw, dict);
    if (ph) return ph;
    if (/[A-Za-z]/.test(raw)) {
      const out = [];
      for (const part of raw.match(/\d+|[A-Za-z']+/g) || []) {
        const sub = part[0] >= "0" && part[0] <= "9"
          ? digitPhones(part, dict) : resolvePhones(part, dict, depth);
        if (!sub) return null;
        out.push(...sub);
      }
      return out.length ? out : null;
    }
  }
  const w = raw.toLowerCase().replace(/’/g, "'").replace(/[^a-z']/g, "");
  if (!w) return null;
  // exact first, so leading-apostrophe entries ('em, 'til) still match; then
  // without surrounding quotes, for words written as 'good'; then the part
  // before an internal apostrophe, for rock'n'roll
  const bare = w.replace(/^'+|'+$/g, "");
  const cands = bare !== w ? [w, bare] : [w];
  if (bare.includes("'")) cands.push(bare.split("'")[0]);
  for (const cand of cands) {
    const ph = dictLookup(cand, dict);
    if (ph) return ph;
  }
  const ph = resolve(w, dict, 0)
    || (w.includes("'") ? resolve(w.replace(/'/g, ""), dict, 0) : null)
    || glued(w, dict)
    || lts(w);
  return ph && ph.length ? ph : null;
}

// Placeholder phones for a word no step can pronounce, one per syllable. The
// CTC path still needs something to stretch over the sung syllables, or the
// word has no time of its own.
const VOWEL_LETTERS = new Set("aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿаеёиоуыэюяіїє"
  .split("").concat("αεηιουωάέήίόύώ".split("")));

// Cyrillic read letter by letter. Not Russian phonology, but the consonants
// delimit words for the CTC path and the vowels give it length.
const CYRILLIC = {"а": ["AA"], "б": ["B"], "в": ["V"], "г": ["G"], "д": ["D"], "е": ["EH"], "ё": ["OW"], "ж": ["ZH"], "з": ["Z"], "и": ["IY"], "й": ["Y"], "к": ["K"], "л": ["L"], "м": ["M"], "н": ["N"], "о": ["AA"], "п": ["P"], "р": ["R"], "с": ["S"], "т": ["T"], "у": ["UW"], "ф": ["F"], "х": ["HH"], "ц": ["T", "S"], "ч": ["CH"], "ш": ["SH"], "щ": ["SH"], "ъ": [], "ы": ["IH"], "ь": [], "э": ["EH"], "ю": ["UW"], "я": ["AA"], "і": ["IY"], "ї": ["IY"], "є": ["EH"], "ґ": ["G"]};
const CYR_IOTATED = new Set(["е", "ю", "я", "ё", "є", "ї"]);
const CYR_VOWELS = new Set(["а", "е", "и", "о", "у", "ы", "э", "ю", "я", "ё", "є", "і", "ї"]);

function cyrillicPhones(w) {
  const cs = [...w];
  if (!cs.some(c => c in CYRILLIC)) return null;
  const out = [];
  for (let i = 0; i < cs.length; i++) {
    const ch = cs[i], ph = CYRILLIC[ch];
    if (!ph) continue;                      // ь, ъ and anything else silent
    if (CYR_IOTATED.has(ch) &&
        (i === 0 || CYR_VOWELS.has(cs[i - 1]) || "ьъй".includes(cs[i - 1]))) out.push("Y");
    out.push(...ph);
  }
  if (out.length && !out.some(p => VOWELS.has(p))) out.push("AH");
  return out.length ? out : null;
}

function skeleton(word) {
  const w = word.toLowerCase();
  const cyr = cyrillicPhones(w);
  if (cyr) return cyr;
  const letters = [...w].filter(c => /\p{L}/u.test(c));
  let k = letters.filter(c => VOWEL_LETTERS.has(c)).length;
  if (!k) k = Math.ceil(letters.length / 3);   // no vowel letters we know
  return new Array(Math.max(1, Math.min(k, 6))).fill("AH");
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
      // any script counts: a Cyrillic or CJK word is still a word
      if (!tok || !/[\p{L}\p{N}]/u.test(tok) || /^\*+$/.test(tok)) continue;
      // a word nothing can pronounce gets placeholder vowels, one per
      // syllable, so it keeps its place and still takes audio
      const ph = resolvePhones(tok, dict);
      words.push({ text: tok, phones: ph || skeleton(tok), skel: !ph });
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
    if (i < j) { [re[i], re[j]] = [re[j], re[i]];[im[i], im[j]] = [im[j], im[i]]; }
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
    else out.push({
      text: w.text,
      start: Math.min(...f.map(x => x[0])) * FRAME + 0.02,
      end: Math.max(...f.map(x => x[1])) * FRAME + 0.02
    });
  }
  return out;
}

// ---------------------------------------------------------------- placement
// A word the path skipped, or a whole line it skipped, has no measured time.
// Those words share the silence between their neighbours in proportion to
// their syllables, and such a line is spread over the gap between the lines
// around it. Mirrors place_words in qadalign/align.py.
const SYL_DUR = 0.30;

function syllables(text) {
  const m = text.toLowerCase().replace(/[^a-z]/g, "").match(/[aeiouy]+/g);
  return Math.max(1, m ? m.length : 0);
}

function fillGap(words, lo, hi) {
  const n = words.length;
  let i = 0;
  while (i < n) {
    if (words[i].start !== null) { i++; continue; }
    let j = i;
    while (j + 1 < n && words[j + 1].start === null) j++;
    let a = i > 0 ? words[i - 1].end : lo;
    let b = j + 1 < n ? words[j + 1].start : hi;
    if (a === null || a === undefined) a = 0;
    if (b === null || b === undefined || b < a) b = a;
    let total = 0;
    for (let k = i; k <= j; k++) total += syllables(words[k].text);
    if (!total) total = 1;
    let t = a;
    const span = b - a;
    for (let k = i; k <= j; k++) {
      const d = span * syllables(words[k].text) / total;
      words[k].start = t;
      words[k].end = t + d;
      t += d;
    }
    i = j + 1;
  }
}

function placeWords(lines) {
  const anchors = lines.map(ln => {
    const w = ln.words.find(w => w.start !== null);
    return w ? w.start : null;
  });
  let prevEnd = 0;
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i], words = ln.words;
    if (!words.length) { ln.start = ln.end = prevEnd; continue; }
    let nxt = null;
    for (let j = i + 1; j < lines.length; j++) if (anchors[j] !== null) { nxt = anchors[j]; break; }
    const here = anchors[i];
    if (here === null) {
      const lo = prevEnd + 0.02;
      let hi = nxt !== null ? nxt - 0.05
        : lo + SYL_DUR * words.reduce((s, w) => s + syllables(w.text), 0);
      fillGap(words, lo, Math.max(hi, lo));
    } else {
      const lead = words.findIndex(w => w.start !== null);
      let last = words.length - 1;
      while (last >= 0 && words[last].start === null) last--;
      let lo = here;
      if (lead) {
        const want = SYL_DUR * words.slice(0, lead).reduce((s, w) => s + syllables(w.text), 0);
        lo = Math.max(prevEnd + 0.02, here - want);
      }
      let hi = words[last].end;
      if (last < words.length - 1) {
        hi += SYL_DUR * words.slice(last + 1).reduce((s, w) => s + syllables(w.text), 0);
        if (nxt !== null) hi = Math.min(hi, nxt - 0.05);
        hi = Math.max(hi, words[last].end);
      }
      fillGap(words, lo, hi);
    }
    ln.start = words[0].start;
    ln.end = words[words.length - 1].end;
    prevEnd = ln.end;
  }
}

// ---------------------------------------------------------------- run
async function run(file, lyricsText, onProgress) {
  const notify = onProgress || (() => { });
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
  if (!flat.length) throw new Error("no lyric words found");

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
  let k = 0;
  for (const ln of parsed) {
    out.push({ text: ln.text, words: words.slice(k, k + ln.words.length) });
    k += ln.words.length;
  }
  placeWords(out);
  for (const ln of out) {
    ln.words = ln.words.map(w => ({ text: w.text, start: w.start, end: w.end }));
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
let previewRaf = 0, previewCur = -1;

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
  cancelAnimationFrame(previewRaf);
  previewRaf = 0;
  previewCur = -1;
  p.onplay = startPreviewSync;
  p.onpause = p.onseeked = p.onended = syncPreview;
  syncPreview();
  startPreviewSync();
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
  if (cur >= 0 && cur !== previewCur) {
    const box = $("lines"), el = preview[cur].el;
    if (el.offsetTop < box.scrollTop || el.offsetTop > box.scrollTop + box.clientHeight - 40) {
      box.scrollTop = el.offsetTop - box.clientHeight / 2;
    }
  }
  previewCur = cur;
}

function startPreviewSync() {
  if (previewRaf) return;
  const step = () => {
    syncPreview();
    previewRaf = $("player").paused ? 0 : requestAnimationFrame(step);
  };
  previewRaf = requestAnimationFrame(step);
}

// ---------------------------------------------------------------- downloads
let last = null;

function withLineBoundaries(res) {
  if (!$("match-line").checked) return res;
  return { lines: res.lines.map(ln => {
    if (!ln.words.length) return ln;
    const words = ln.words.slice();
    words[0] = { ...words[0], start: ln.start };
    const tail = words[words.length - 1];
    words[words.length - 1] = { ...tail, end: Math.max(ln.end, tail.end) };
    return { ...ln, words };
  }) };
}

function applyBoundaries() {
  if (!last) return;
  last.res = withLineBoundaries(last.raw);
  showPreview(last.res);
  $("out").textContent = JSON.stringify(last.res, null, 1);
}

function cumulativeRows(lines) {
  const rows = [];
  for (const l of lines) {
    let prev = l.start;
    for (let i = 0; i < l.words.length; i++) {
      const end = Math.max(prev, l.words[i].end);
      rows.push({ text: l.words.slice(0, i + 1).map(w => w.text).join(" "), start: prev, end });
      prev = end;
    }
  }
  return rows;
}

function vizzy(kind) {
  if (!last) return;
  const rows = kind === "words"
    ? last.res.lines.flatMap(l => l.words.map(w => ({ text: w.text, start: w.start, end: w.end })))
    : kind === "cumulative"
      ? cumulativeRows(last.res.lines)
      : last.res.lines.map(l => ({ text: l.text, start: l.start, end: l.end }));
  const suffix = kind === "words" ? "_lyrics_wordlevel.json"
    : kind === "cumulative" ? "_lyrics_cumulative.json" : "_lyrics_linelevel.json";
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
  $("match-line-wrap").hidden = true;
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
    last = { raw: res, res, base: file.name.replace(/\.[^.]+$/, "") };
    $("downloads").hidden = false;
    $("match-line-wrap").hidden = false;
    $("out").hidden = false;
    applyBoundaries();
  } catch (e) {
    $("status").className = "error";
    $("status").textContent = "error: " + e.message;
    console.error(e);
  } finally {
    $("go").disabled = false;
  }
};

$("dl-words").onclick = () => vizzy("words");
$("dl-cumulative").onclick = () => vizzy("cumulative");
$("dl-lines").onclick = () => vizzy("lines");
$("match-line").onchange = applyBoundaries;

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
  const out = [];
  for (const line of ta.value.split(/\r?\n/)) {
    const tagged = /\[[^\]]*\]/.test(line);
    const text = line.replace(/\[[^\]]*\]/g, "").trim();
    if (tagged && !text) continue;      // nothing but tags, e.g. [Verse 1]
    out.push(tagged ? text : line);     // otherwise keep the words
  }
  ta.value = out.join("\n");
  ta.focus();
};
