"""
Build `docs/index.html`: the complete app as one static file.

Both engines, one file. The offline engine (BM25 + extractive QA + classical-CV
image analysis) is ported to the browser and needs nothing at all. The online
engine is optional: it calls Claude directly from the page, using a key the
visitor pastes in, so your own key is never exposed by the hosted site.

Drop `docs/` on GitHub Pages and the offline demo runs forever, free.

The JS mirrors bm25.py / extractive.py exactly; tests/test_static_parity.mjs
checks that both implementations agree on real queries.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
KB = ROOT / "knowledge_base"

OFFLINE_JS = r"""
/* ============================================================================
   Offline adapter — a faithful port of bm25.py, extractive.py and
   imageanalysis.py. No network, no API key, no model.
   ========================================================================== */

const DOCS = __DOCS__;
const CHUNK = 500, OVERLAP = 100, TOP_K = 3, MIN_SCORE = 0.1;
const K1 = 1.5, B = 0.75;

const STOPWORDS = new Set(["a","an","and","are","as","at","be","by","do","does",
  "for","from","how","i","in","is","it","many","me","my","of","on","or","the",
  "to","what","when","where","which","with","you","your"]);

export function tokenize(text) {
  return (text.toLowerCase().match(/[a-z0-9]+/g) || [])
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

function chunkText(text) {
  const out = [];
  for (let start = 0; start < text.length; start += CHUNK - OVERLAP) {
    const piece = text.slice(start, start + CHUNK).trim();
    if (piece) out.push(piece);
  }
  return out;
}

/* ---------------------------- BM25 Okapi ---------------------------------- */

export class BM25Index {
  fit(chunks) {
    this.chunks = chunks;
    this.tf = [];
    this.lengths = [];
    this.df = new Map();
    for (const c of chunks) {
      const toks = tokenize(c.text);
      const counts = new Map();
      toks.forEach((t) => counts.set(t, (counts.get(t) || 0) + 1));
      this.tf.push(counts);
      this.lengths.push(toks.length);
      for (const t of counts.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
    }
    const sum = this.lengths.reduce((a, b) => a + b, 0);
    this.avgdl = this.lengths.length ? sum / this.lengths.length : 0;
    return this;
  }

  idf(term) {
    const n = this.chunks.length;
    const df = this.df.get(term) || 0;
    if (df === 0) return 0;
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
  }

  score(qt, i) {
    const tf = this.tf[i], dl = this.lengths[i];
    let total = 0;
    for (const term of qt) {
      const f = tf.get(term) || 0;
      if (!f) continue;
      const denom = f + K1 * (1 - B + B * (this.avgdl ? dl / this.avgdl : 1));
      total += this.idf(term) * (f * (K1 + 1)) / denom;
    }
    return total;
  }

  search(query, topK = TOP_K, minScore = MIN_SCORE) {
    const qt = tokenize(query);
    if (!qt.length || !this.chunks.length) return [];
    return this.chunks
      .map((c, i) => ({ source: c.source, text: c.text,
                        score: Math.round(this.score(qt, i) * 1e4) / 1e4 }))
      .sort((a, b) => b.score - a.score)
      .filter((r) => r.score > minScore)
      .slice(0, topK);
  }
}

/* -------------------------- Extractive answers ---------------------------- */

const CUES = {
  how: ["by","via","through","visit","open","submit","use","select"],
  when: ["within","before","after","during","day","week","month"],
  where: ["in","at","on","portal","desk","floor"],
  who: ["manager","lead","team","hr","head"],
  why: ["because","so","since"],
};

export function splitSentences(text) {
  const out = [];
  for (const block of text.split("\n")) {
    const b = block.trim();
    if (!b || b.startsWith("#")) continue;
    for (const s of b.split(/(?<=[.!?])\s+(?=[A-Z#])/)) {
      const t = s.trim();
      if (t.length > 25) out.push(t);
    }
  }
  return out;
}

export function extractiveAnswer(query, snippets, maxSentences = 2) {
  if (!snippets.length)
    return "Nothing in the knowledge base is relevant to that question.";

  const candidates = [];
  snippets.forEach((sn) => splitSentences(sn.text).forEach((text, ) =>
    candidates.push({ text, source: sn.source, idx: candidates.length })));
  if (!candidates.length) return snippets[0].text.slice(0, 300);

  const idx = new BM25Index().fit(
    candidates.map((c) => ({ source: c.source, text: c.text })));
  const qt = tokenize(query);
  const firstWord = (query.trim().toLowerCase().split(/\s+/)[0]) || "";
  const cues = CUES[firstWord] || [];

  candidates.forEach((c, i) => {
    let s = idx.score(qt, i);
    const low = c.text.toLowerCase();
    if (cues.length && cues.some((k) => low.includes(k))) s *= 1.15;
    c.score = s;
  });

  const ranked = [...candidates].sort((a, b) => b.score - a.score);
  if (ranked[0].score <= 0)
    return "The knowledge base was searched but no passage answers that question directly.";

  const picked = [], seen = new Set();
  for (const c of ranked) {
    if (c.score <= 0 || picked.length >= maxSentences) break;
    const key = c.text.slice(0, 60);
    if (seen.has(key)) continue;
    seen.add(key);
    picked.push(c);
  }
  picked.sort((a, b) => a.idx - b.idx);

  const body = picked.map((c) => c.text).join(" ");
  const sources = [...new Set(picked.map((c) => c.source))].sort();
  return `${body}\n\n— extracted verbatim from ${sources.join(", ")}`;
}

/* --------------------- Classical CV image analysis ------------------------ */

const NAMED = { black:[0,0,0], white:[255,255,255], grey:[128,128,128],
  red:[220,30,30], maroon:[128,0,0], orange:[240,140,20], amber:[200,150,0],
  yellow:[240,230,60], olive:[128,128,0], green:[40,170,70], teal:[0,128,128],
  cyan:[60,200,220], blue:[40,90,200], navy:[10,30,80], purple:[120,60,180],
  magenta:[200,60,170], pink:[240,160,190], brown:[120,80,40], beige:[225,210,180] };

const nameColour = ([r,g,b]) => Object.keys(NAMED).reduce((best, n) => {
  const [R,G,B] = NAMED[n];
  const d = (r-R)**2 + (g-G)**2 + (b-B)**2;
  return d < best.d ? { n, d } : best;
}, { n: "grey", d: Infinity }).n;

/** Pure function so it can be unit-tested outside a browser. */
export function analysePixels(data, w, h) {
  const n = w * h;
  const lum = new Float32Array(n);
  let rg_sum = 0, yb_sum = 0, rg_sq = 0, yb_sq = 0, spread = 0;
  const buckets = new Map();

  for (let i = 0; i < n; i++) {
    const r = data[i*4], g = data[i*4+1], b = data[i*4+2];
    lum[i] = 0.299*r + 0.587*g + 0.114*b;
    const rg = r - g, yb = 0.5*(r+g) - b;
    rg_sum += rg; yb_sum += yb; rg_sq += rg*rg; yb_sq += yb*yb;
    spread += Math.abs(r - b);
    // 5-bit-per-channel quantisation as a stand-in for median cut
    const key = (r>>5)*1024 + (g>>5)*32 + (b>>5);
    const e = buckets.get(key) || { c: 0, r: 0, g: 0, b: 0 };
    e.c++; e.r += r; e.g += g; e.b += b;
    buckets.set(key, e);
  }

  const mean = lum.reduce((a,v) => a+v, 0) / n;
  const variance = lum.reduce((a,v) => a + (v-mean)**2, 0) / n;
  const brightness = mean / 255, contrast = Math.sqrt(variance) / 255;

  const rgStd = Math.sqrt(rg_sq/n - (rg_sum/n)**2);
  const ybStd = Math.sqrt(yb_sq/n - (yb_sum/n)**2);
  const colourfulness = Math.sqrt(rgStd**2 + ybStd**2) +
    0.3 * Math.sqrt((rg_sum/n)**2 + (yb_sum/n)**2);
  const isGrey = colourfulness < 8 && spread/n < 6;

  // Edge density: gradient magnitude on the luminance plane.
  let edges = 0;
  for (let y = 1; y < h-1; y++)
    for (let x = 1; x < w-1; x++) {
      const i = y*w + x;
      const gx = lum[i+1] - lum[i-1], gy = lum[i+w] - lum[i-w];
      if (Math.hypot(gx, gy) > 40) edges++;
    }
  const edgeDensity = edges / Math.max(1, (w-2)*(h-2));

  const dominant = [...buckets.entries()]
    .sort((a,b) => b[1].c - a[1].c).slice(0, 3)
    .map(([, e]) => {
      const rgb = [Math.round(e.r/e.c), Math.round(e.g/e.c), Math.round(e.b/e.c)];
      return { rgb, hex: "#" + rgb.map((v) => v.toString(16).padStart(2,"0")).join("").toUpperCase(),
               name: nameColour(rgb), share: Math.round(e.c/n * 1000)/1000 };
    });

  const orientation = w > h*1.15 ? "landscape" : h > w*1.15 ? "portrait" : "square";
  return { width: w, height: h, orientation, brightness: +brightness.toFixed(3),
           contrast: +contrast.toFixed(3), edge_density: +edgeDensity.toFixed(3),
           colourfulness: +colourfulness.toFixed(1), is_greyscale: isGrey,
           dominant_colours: dominant };
}

const exposure = (b) => b < 0.2 ? "very dark" : b < 0.4 ? "dark"
  : b < 0.62 ? "evenly exposed" : b < 0.82 ? "bright" : "very bright";
const detailOf = (e) => e < 0.02 ? "almost no internal detail — a flat or uniform image"
  : e < 0.08 ? "smooth, with few hard edges"
  : e < 0.2 ? "moderately detailed" : "densely detailed or textured";

export function describeMetrics(m, format = "PNG") {
  const p = m.dominant_colours, lead = p.length ? p[0].name : "neutral";
  const phrase = m.is_greyscale
    ? (["white","black","grey"].includes(lead) ? `${lead} tones` : "neutral greys")
    : (p.length > 1 && p[1].name !== lead ? `${lead} and ${p[1].name}` : lead);

  const caption = `A ${m.orientation} ${format.toUpperCase()} image ` +
    `(${m.width}×${m.height}), ${exposure(m.brightness)}, dominated by ${phrase}.`;

  const tags = [ m.is_greyscale ? "greyscale" : lead, m.orientation,
    m.edge_density < 0.02 ? "flat" : m.edge_density > 0.2 ? "detailed"
      : m.contrast > 0.25 ? "high-contrast" : "smooth" ];

  const share = p.length ? `${(p[0].share*100).toFixed(0)}%` : "n/a";
  const detailed = `The dominant colour is ${lead} (${p.length ? p[0].hex : "n/a"}), ` +
    `covering about ${share} of the frame. Mean brightness is ${m.brightness.toFixed(2)} ` +
    `and RMS contrast ${m.contrast.toFixed(2)}; the image is ${detailOf(m.edge_density)}. ` +
    `Measured with classical computer vision — colour quantisation, edge filtering and ` +
    `exposure statistics — not a vision model, so no objects or people are identified.`;

  return { caption, tags: tags.slice(0,3), detailed };
}

/* ============================ ONLINE ENGINE ================================
   Optional. Calls the Claude API with the SAME chunks the offline engine used,
   so the two answers differ only in how they are produced. Offline never
   touches this code path.
   ========================================================================== */

let USER_KEY = "";
let PROVIDER = "groq";   // default; user picks in the dropdown

// Browser-side provider registry. Every entry except Anthropic speaks the
// OpenAI chat-completions protocol, so one code path covers them all.
const WEB_PROVIDERS = {
  groq:     { label: "Groq (free)",     url: "https://api.groq.com/openai/v1/chat/completions",
              model: "llama-3.3-70b-versatile", kind: "openai" },
  gemini:   { label: "Google Gemini",   url: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
              model: "gemini-2.5-flash", kind: "openai" },
  openai:   { label: "OpenAI",          url: "https://api.openai.com/v1/chat/completions",
              model: "gpt-4o-mini", kind: "openai" },
  xai:      { label: "xAI (Grok)",      url: "https://api.x.ai/v1/chat/completions",
              model: "grok-3", kind: "openai" },
  deepseek: { label: "DeepSeek",        url: "https://api.deepseek.com/chat/completions",
              model: "deepseek-chat", kind: "openai" },
  mistral:  { label: "Mistral",         url: "https://api.mistral.ai/v1/chat/completions",
              model: "mistral-small-latest", kind: "openai" },
  anthropic:{ label: "Anthropic (Claude)", url: "https://api.anthropic.com/v1/messages",
              model: "claude-sonnet-4-6", kind: "anthropic" },
};

/** Exposed for tests and for the key field. */
export function setApiKey(key) { USER_KEY = (key || "").trim(); }
export function hasApiKey() { return Boolean(USER_KEY); }
export function setProvider(name) { if (WEB_PROVIDERS[name]) PROVIDER = name; }
export function getProvider() { return PROVIDER; }

const KEY_HINT =
  "Pick a provider and paste its API key at the top of this page to enable the " +
  "online engine. Groq and Gemini have generous free tiers. The offline engine " +
  "needs no key.";

async function chatComplete(system, messages) {
  // Without a key the browser cannot even reach an API: the request carries no
  // credentials, so it is blocked and surfaces as an opaque "Failed to fetch".
  if (!USER_KEY) throw new Error(KEY_HINT);

  const p = WEB_PROVIDERS[PROVIDER] || WEB_PROVIDERS.groq;
  let headers, body;

  if (p.kind === "anthropic") {
    headers = {
      "Content-Type": "application/json",
      "x-api-key": USER_KEY,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    };
    body = JSON.stringify({ model: p.model, max_tokens: 1000, system, messages });
  } else {
    // OpenAI-compatible: system prompt is just the first message.
    headers = { "Content-Type": "application/json", "Authorization": "Bearer " + USER_KEY };
    body = JSON.stringify({
      model: p.model, max_tokens: 1000, temperature: 0.3,
      messages: [{ role: "system", content: system }, ...messages],
    });
  }

  let r;
  try {
    r = await fetch(p.url, { method: "POST", headers, body });
  } catch (err) {
    throw new Error(
      "Could not reach " + p.label + " from the browser. This is usually a "
      + "network block or a CORS rejection, not a problem with the key.");
  }

  if (r.status === 401 || r.status === 403)
    throw new Error(p.label + " rejected that API key (" + r.status + ").");
  if (r.status === 429)
    throw new Error("Rate limited by " + p.label + " (429). Try again shortly.");
  if (!r.ok) throw new Error(p.label + " returned " + r.status + ".");

  const data = await r.json();
  if (p.kind === "anthropic")
    return data.content.filter((b) => b.type === "text").map((b) => b.text).join("\n").trim();
  return (data.choices?.[0]?.message?.content || "").trim();
}

const RAG_SYSTEM =
  "You are a helpful assistant. Answer the user's question using ONLY the provided " +
  "context. If the context does not contain enough information, say so plainly. " +
  "Be concise and name the source document you relied on.";

const VISION_SYSTEM =
  "You are an image analysis assistant. Respond in exactly this format:\n" +
  "CAPTION: <one-sentence description>\nTAGS: <tag1>, <tag2>, <tag3>\n" +
  "DETAILS: <2-3 sentence detailed description>";

export function parseVision(raw) {
  const out = { caption: "", tags: [], detailed: "" };
  raw.split("\n").forEach((line) => {
    const l = line.trim();
    if (/^caption:/i.test(l)) out.caption = l.split(":").slice(1).join(":").trim();
    else if (/^tags:/i.test(l))
      out.tags = l.split(":").slice(1).join(":").split(",").map((s) => s.trim()).filter(Boolean);
    else if (/^details:/i.test(l)) out.detailed = l.split(":").slice(1).join(":").trim();
  });
  if (!out.caption) out.caption = raw.slice(0, 120);
  if (!out.tags.length) out.tags = ["image", "uploaded", "analysis"];
  return out;
}

/* ------------------------------- adapter ---------------------------------- */

const CHUNKS = [];
for (const [source, text] of Object.entries(DOCS))
  for (const t of chunkText(text)) CHUNKS.push({ source, text: t });

const INDEX = new BM25Index().fit(CHUNKS);
const cache = new Map();

function keyField(available) {
  const banner = document.getElementById("banner");
  if (!banner) return;

  if (!document.getElementById("apikey")) {
    const options = Object.entries(WEB_PROVIDERS)
      .map(([id, p]) => '<option value="' + id + '"' +
        (id === PROVIDER ? " selected" : "") + '>' + p.label + '</option>')
      .join("");

    banner.innerHTML =
      '<div class="note"><strong>The offline engine needs nothing.</strong> ' +
      'BM25 retrieval, extractive answers and classical-CV image analysis all run ' +
      'in this page \u2014 no key, no server, no network.<br>' +
      'To also see the online engine, choose a provider and paste your own API key. ' +
      'It is kept in memory for this tab only: never stored, never sent anywhere but ' +
      'the provider you pick. Groq and Gemini have generous free tiers.' +
      '<br><select id="provider" style="margin-top:9px;padding:8px 10px;' +
      'border:1px solid var(--rule);border-radius:3px;font-family:var(--mono);' +
      'font-size:12px;background:var(--paper)">' + options + '</select> ' +
      '<input id="apikey" type="password" placeholder="paste API key\u2026" ' +
      'autocomplete="off" spellcheck="false" ' +
      'style="width:min(320px,100%);padding:8px 10px;' +
      'border:1px solid var(--rule);border-radius:3px;font-family:var(--mono);' +
      'font-size:12px">' +
      ' <span id="keystatus" style="font-family:var(--mono);font-size:11px;' +
      'letter-spacing:.06em;text-transform:uppercase;color:var(--ink-soft)"></span></div>';

    document.getElementById("apikey").addEventListener("input", (e) => {
      setApiKey(e.target.value);
      updateKeyStatus();
    });
    document.getElementById("provider").addEventListener("change", (e) => {
      setProvider(e.target.value);
      updateKeyStatus();
    });
  }
  updateKeyStatus(available);
}

function updateKeyStatus() {
  const el = document.getElementById("keystatus");
  if (!el) return;
  const label = WEB_PROVIDERS[PROVIDER] ? WEB_PROVIDERS[PROVIDER].label : PROVIDER;
  el.textContent = hasApiKey() ? (label + " armed") : "online engine idle";
  el.style.color = hasApiKey() ? "var(--azure)" : "var(--ink-soft)";
}

async function imageMetrics(file) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 256 / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, w, h);
  const m = analysePixels(ctx.getImageData(0, 0, w, h).data, w, h);
  m.width = bitmap.width; m.height = bitmap.height;
  m.orientation = bitmap.width > bitmap.height * 1.15 ? "landscape"
    : bitmap.height > bitmap.width * 1.15 ? "portrait" : "square";
  return m;
}

const fileToBase64 = (file) => new Promise((res, rej) => {
  const fr = new FileReader();
  fr.onload = () => res(fr.result.split(",")[1]);
  fr.onerror = () => rej(new Error("Could not read the file."));
  fr.readAsDataURL(file);
});

export const PREVIEW_ADAPTER = {
  setKey: setApiKey,

  /** The UI delegates banner rendering here so the key field survives reruns. */
  banner(available, provider) { keyField(available); },

  async stats() {
    const byDoc = {};
    CHUNKS.forEach((c) => (byDoc[c.source] = (byDoc[c.source] || 0) + 1));
    return { total_chunks: CHUNKS.length, cached_queries: cache.size,
             documents: Object.entries(byDoc).map(([source, chunks]) => ({ source, chunks })),
             // Honest: without a key the online engine genuinely cannot run.
             online_available: hasApiKey(), online_provider: PROVIDER,
             online_hint: KEY_HINT };
  },

  async ask(query, mode = "both") {
    const ck = `${mode}::${query.toLowerCase().trim()}`;
    if (cache.has(ck)) return { ...cache.get(ck), from_cache: true };

    const hits = INDEX.search(query);
    const payload = {
      metric: "bm25", from_cache: false,
      snippets: hits.map((h) => ({ source: h.source, text: h.text.slice(0, 220), score: h.score })),
      sources: [...new Set(hits.map((h) => h.source))],
      offline: null, online: null, online_error: null,
      online_available: hasApiKey(), online_provider: PROVIDER,
      online_hint: KEY_HINT,
    };

    if (!hits.length) {
      const nothing = "Nothing in the knowledge base is relevant to that question.";
      if (mode !== "online") payload.offline = nothing;
      if (mode !== "offline") payload.online = nothing;
      cache.set(ck, payload);
      return payload;
    }

    if (mode !== "online") payload.offline = extractiveAnswer(query, hits);

    if (mode !== "offline") {
      const context = hits.map((h) => `[Source: ${h.source}]\n${h.text}`).join("\n\n---\n\n");
      try {
        payload.online = await chatComplete(RAG_SYSTEM,
          [{ role: "user", content: `Context:\n${context}\n\nQuestion: ${query}` }]);
      } catch (err) {
        payload.online_error = err.message;
      }
    }

    if (!payload.online_error) cache.set(ck, payload);
    return payload;
  },

  async describe(file, mode = "both") {
    const out = { offline: null, online: null, online_error: null,
                  online_available: hasApiKey(), online_provider: PROVIDER,
                  online_hint: KEY_HINT };

    if (mode !== "online") {
      const m = await imageMetrics(file);
      out.offline = describeMetrics(m, file.type.split("/")[1] || "image");
    }
    if (mode !== "offline") {
      const p = WEB_PROVIDERS[PROVIDER] || WEB_PROVIDERS.groq;
      const VISION_CAPABLE = { openai: 1, gemini: 1, xai: 1, anthropic: 1 };
      if (!VISION_CAPABLE[PROVIDER]) {
        out.online_error = p.label + " does not support image input. " +
          "Switch to OpenAI, Gemini, xAI or Anthropic for online image analysis, " +
          "or use the offline analyser, which needs no key.";
      } else {
        try {
          const b64 = await fileToBase64(file);
          let content;
          if (p.kind === "anthropic") {
            content = [
              { type: "image", source: { type: "base64", media_type: file.type, data: b64 } },
              { type: "text", text: "Describe this image." }];
          } else {
            // OpenAI-compatible vision uses image_url with a data URI.
            content = [
              { type: "text", text: "Describe this image." },
              { type: "image_url", image_url: { url: "data:" + file.type + ";base64," + b64 } }];
          }
          out.online = parseVision(await chatComplete(VISION_SYSTEM,
            [{ role: "user", content }]));
        } catch (err) {
          out.online_error = err.message;
        }
      }
    }
    return out;
  },
};

if (typeof document !== "undefined") keyField(false);
"""


def build() -> Path:
    css = (STATIC / "style.css").read_text()
    html = (STATIC / "index.html").read_text()
    app_js = (STATIC / "app.js").read_text()

    # The static build never talks to a backend, so strip ServerAdapter entirely
    # rather than shipping unreachable fetch() calls in a file we call offline-safe.
    start = app_js.index("const ServerAdapter = {")
    end = app_js.index("const adapter = window.PREVIEW_ADAPTER || ServerAdapter;")
    app_js = (app_js[:start] + app_js[end:]).replace(
        "const adapter = window.PREVIEW_ADAPTER || ServerAdapter;",
        "const adapter = window.PREVIEW_ADAPTER;")
    docs = {p.name: p.read_text() for p in sorted(KB.glob("*.md"))}

    offline_js = OFFLINE_JS.replace("__DOCS__", json.dumps(docs, indent=0))
    # `export` keywords exist for the Node parity test; strip them for the browser.
    browser_js = (offline_js.replace("export class ", "class ")
                            .replace("export function ", "function ")
                            .replace("export const PREVIEW_ADAPTER",
                                     "window.PREVIEW_ADAPTER"))

    body = html.replace('<link rel="stylesheet" href="/static/style.css">',
                        f"<style>\n{css}\n</style>")
    body = body.replace('<script src="/static/app.js"></script>',
                        f"<script>\n{browser_js}\n</script>\n<script>\n{app_js}\n</script>")

    out_dir = ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "index.html").write_text(body)
    (out_dir / ".nojekyll").write_text("")

    # Emit the raw module so Node can import and test the pure functions.
    (ROOT / "tests" / "offline_module.mjs").write_text(offline_js)

    assert "/static/" not in body, "external asset reference survived inlining"
    assert "window.PREVIEW_ADAPTER" in body
    # Exactly one network call site exists, and it lives in the online engine.
    # The offline engine (BM25, extractive QA, classical CV) must never fetch.
    assert browser_js.count("fetch(") == 1, "offline engine must not fetch anything"
    assert body.count("fetch(") == 1, \
        f"expected exactly one network call site, found {body.count('fetch(')}"
    assert "ServerAdapter" not in body, "dead backend code left in the static build"
    # All seven provider endpoints must be present in the registry.
    for host in ("api.groq.com", "generativelanguage.googleapis.com",
                 "api.openai.com", "api.x.ai", "api.deepseek.com",
                 "api.mistral.ai", "api.anthropic.com"):
        assert host in body, f"provider endpoint {host} missing from the build"
    for name in docs:
        assert name in body, f"{name} missing from inlined corpus"
    return out_dir / "index.html"


if __name__ == "__main__":
    p = build()
    print(f"{p} written: {p.stat().st_size:,} bytes\n"
          f"  offline engine: no key, no server, no network\n"
          f"  online engine : optional, browser-direct Claude call")
