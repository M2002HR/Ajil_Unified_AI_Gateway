(() => {
  const STORAGE_KEY = "uag_studio_settings_v2";

  const tabs = [
    { id: "chat", label: "Chat", desc: "Chat Completions with provider/model routing" },
    { id: "vision", label: "Vision", desc: "Image understanding using chat + image input" },
    { id: "responses", label: "Responses", desc: "OpenAI-style Responses endpoint" },
    { id: "image", label: "Image", desc: "Image generation (Pollinations models)" },
    { id: "tts", label: "Text to Speech", desc: "Generate audio from text" },
    { id: "stt", label: "Speech to Text", desc: "Transcribe audio files" },
    { id: "embeddings", label: "Embeddings", desc: "Generate embedding vectors" },
    { id: "orchestrate", label: "Orchestrate", desc: "Advanced raw capability routing" },
    { id: "models", label: "Model Explorer", desc: "Live catalog from all providers" },
  ];

  const state = {
    activeTab: "chat",
    catalogItems: [],
    providerRows: [],
    loadingModels: false,
    visionImageDataUrl: "",
    imageOptionCache: {},
  };

  const qs = (s) => document.querySelector(s);
  const qsa = (s) => Array.from(document.querySelectorAll(s));

  const refs = {
    navTabs: qs("#navTabs"),
    viewTitle: qs("#viewTitle"),
    viewDesc: qs("#viewDesc"),
    tokenInput: qs("#tokenInput"),
    tokenHeaderInput: qs("#tokenHeaderInput"),
    routerStrategy: qs("#routerStrategy"),
    routerMode: qs("#routerMode"),
    routerTimeout: qs("#routerTimeout"),
    routerAttempts: qs("#routerAttempts"),
    saveSettingsBtn: qs("#saveSettingsBtn"),
    loadModelsBtn: qs("#loadModelsBtn"),
    connStatus: qs("#connStatus"),
    modelCountBadge: qs("#modelCountBadge"),
    providerCountBadge: qs("#providerCountBadge"),
    toastRoot: qs("#toastRoot"),
  };

  function showToast(text) {
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = text;
    refs.toastRoot.appendChild(el);
    setTimeout(() => el.remove(), 2400);
  }

  function setStatus(text, kind = "") {
    refs.connStatus.textContent = text;
    refs.connStatus.className = `hint ${kind}`.trim();
  }

  function getSettings() {
    return {
      token: refs.tokenInput.value.trim(),
      tokenHeader: refs.tokenHeaderInput.value.trim() || "x-api-token",
      strategy: refs.routerStrategy.value,
      mode: refs.routerMode.value,
      timeoutSec: Math.max(3, Number(refs.routerTimeout.value) || 25),
      maxAttempts: Math.max(1, Number(refs.routerAttempts.value) || 6),
    };
  }

  function saveSettings() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(getSettings()));
  }

  function loadSettings() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      refs.tokenInput.value = parsed.token || "";
      refs.tokenHeaderInput.value = parsed.tokenHeader || "x-api-token";
      refs.routerStrategy.value = parsed.strategy || "fallback_chain";
      refs.routerMode.value = parsed.mode || "limit_safe";
      refs.routerTimeout.value = String(parsed.timeoutSec || 25);
      refs.routerAttempts.value = String(parsed.maxAttempts || 6);
    } catch (_) {
      // ignore invalid storage
    }
  }

  function authHeaders(json = true) {
    const cfg = getSettings();
    const headers = {};
    if (cfg.token) {
      headers[cfg.tokenHeader] = cfg.token;
    }
    if (json) {
      headers["content-type"] = "application/json";
    }
    return headers;
  }

  function defaultRouterConfig(provider) {
    const cfg = getSettings();
    const out = {
      strategy: cfg.strategy,
      mode: cfg.mode,
      timeout_sec: cfg.timeoutSec,
      max_attempts: cfg.maxAttempts,
    };
    if (provider) {
      out.providers = [provider];
    }
    return out;
  }

  async function apiJson(path, body, opts = {}) {
    const resp = await fetch(path, {
      method: opts.method || "POST",
      headers: opts.headers || authHeaders(true),
      body: body != null ? JSON.stringify(body) : undefined,
    });
    let payload = null;
    const text = await resp.text();
    try {
      payload = text ? JSON.parse(text) : null;
    } catch (_) {
      payload = { raw: text };
    }
    return { ok: resp.ok, status: resp.status, data: payload };
  }

  async function apiStreamChat(path, body, handlers = {}) {
    const onDelta = typeof handlers.onDelta === "function" ? handlers.onDelta : () => {};
    const onEvent = typeof handlers.onEvent === "function" ? handlers.onEvent : () => {};
    const resp = await fetch(path, {
      method: "POST",
      headers: authHeaders(true),
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      const text = await resp.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch (_) {
        data = { raw: text };
      }
      return { ok: false, status: resp.status, data };
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";
    const events = [];

    function consumeBlock(block) {
      const lines = String(block || "").split("\n");
      const dataLines = lines
        .map((line) => String(line || "").trim())
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim());
      if (!dataLines.length) return;
      const raw = dataLines.join("\n");
      if (raw === "[DONE]") return;
      let parsed = null;
      try {
        parsed = JSON.parse(raw);
      } catch (_) {
        return;
      }
      events.push(parsed);
      onEvent(parsed);
      const choice = Array.isArray(parsed.choices) ? parsed.choices[0] : null;
      const delta = choice && typeof choice === "object" ? (choice.delta || {}) : {};
      const deltaText = typeof delta.content === "string" ? delta.content : "";
      if (deltaText) {
        fullText += deltaText;
        onDelta(deltaText, parsed);
      }
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      for (const block of blocks) consumeBlock(block);
    }
    if (buffer.trim()) {
      consumeBlock(buffer);
    }

    const synthesized = {
      object: "chat.completion",
      choices: [{ index: 0, message: { role: "assistant", content: fullText }, finish_reason: "stop" }],
      stream_events: events,
    };
    return { ok: true, status: resp.status, data: synthesized };
  }

  function formatJson(value) {
    try {
      return JSON.stringify(value, null, 2);
    } catch (_) {
      return String(value);
    }
  }

  function firstWordToken(text) {
    const tokens = String(text || "").trim().split(/\s+/);
    for (const token of tokens) {
      const cleaned = token
        .replace(/^[^A-Za-z0-9\u0600-\u06FF]+/, "")
        .replace(/[^A-Za-z0-9\u0600-\u06FF]+$/, "");
      if (cleaned) return cleaned;
    }
    return "";
  }

  function directionByFirstWord(text) {
    const first = firstWordToken(text);
    if (/[\u0600-\u06FF]/.test(first)) return "rtl";
    return "ltr";
  }

  function escapeHtml(raw) {
    return String(raw || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('\"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function renderMarkdown(el, text) {
    const raw = String(text || "");
    const dir = directionByFirstWord(raw);
    el.setAttribute("dir", dir);
    el.classList.add("md-output");
    el.classList.toggle("rtl", dir === "rtl");
    el.classList.toggle("ltr", dir !== "rtl");
    if (!raw.trim()) {
      el.innerHTML = "";
      return;
    }

    if (window.marked && window.DOMPurify) {
      const html = window.marked.parse(raw, { gfm: true, breaks: true });
      el.innerHTML = window.DOMPurify.sanitize(html);
      return;
    }
    el.innerHTML = `<pre>${escapeHtml(raw)}</pre>`;
  }

  function uniqueProviders(items) {
    return Array.from(new Set(items.map((x) => x.provider))).sort();
  }

  function byCapability(capability) {
    return state.catalogItems.filter((row) => (row.capabilities || []).includes(capability));
  }

  function byProviderAndCapability(provider, capability) {
    return byCapability(capability).filter((row) => row.provider === provider);
  }

  function findCatalogModel(provider, modelId, capability) {
    const providerKey = String(provider || "").toLowerCase();
    const modelKey = String(modelId || "").toLowerCase();
    return state.catalogItems.find((row) => {
      if (String(row.provider || "").toLowerCase() !== providerKey) return false;
      if (String(row.id || "").toLowerCase() !== modelKey) return false;
      if (capability && !(row.capabilities || []).includes(capability)) return false;
      return true;
    }) || null;
  }

  function modelMaxOutputTokens(provider, modelId, capability) {
    const row = findCatalogModel(provider, modelId, capability);
    if (!row) return null;
    const value = Number(row.max_output_tokens);
    if (!Number.isFinite(value) || value <= 0) return null;
    return Math.floor(value);
  }

  function applyChatMaxTokensDefault(provider, model, forceValue = true) {
    const inputEl = qs("#chatMaxTokens");
    if (!inputEl) return;
    const limit = modelMaxOutputTokens(provider, model, "chat.completions");
    if (limit && limit > 0) {
      inputEl.max = String(limit);
      if (forceValue || !Number(inputEl.value)) {
        inputEl.value = String(limit);
      } else if (Number(inputEl.value) > limit) {
        inputEl.value = String(limit);
      }
      return;
    }
    inputEl.removeAttribute("max");
    if (forceValue && !Number(inputEl.value)) {
      inputEl.value = "1024";
    }
  }

  function resolveChatMaxTokens(provider, model) {
    const inputEl = qs("#chatMaxTokens");
    const selectedLimit = modelMaxOutputTokens(provider, model, "chat.completions");
    const userValue = Number(inputEl?.value || 0);
    if (Number.isFinite(userValue) && userValue > 0) {
      if (selectedLimit && userValue > selectedLimit) return selectedLimit;
      return Math.floor(userValue);
    }
    if (selectedLimit && selectedLimit > 0) return selectedLimit;
    return 1024;
  }

  function rowSupportsVision(row) {
    if (!row || typeof row !== "object") return false;
    const inMods = (row.input_modalities || []).map((v) => String(v).toLowerCase());
    if (inMods.includes("image")) return true;

    const provider = String(row.provider || "").toLowerCase();
    const modelId = String(row.id || "").toLowerCase();
    if (!modelId) return false;
    if (provider === "gemini") {
      if (modelId.startsWith("gemma-")) return false;
      if (modelId.includes("embedding")) return false;
      if (modelId.startsWith("gemini-2.") || modelId.startsWith("gemini-1.5-")) return true;
      if (modelId.includes("vision")) return true;
    }
    if (provider === "groq") {
      if (modelId.includes("vision") || modelId.includes("llava") || modelId.includes("vl")) return true;
    }
    return false;
  }

  function setSelectOptions(select, rows, valueKey = "id", labelKey = "label") {
    const prev = select.value;
    select.innerHTML = "";
    for (const row of rows) {
      const option = document.createElement("option");
      option.value = row[valueKey];
      option.textContent = `${row.provider}/${row[labelKey] || row[valueKey]}`;
      select.appendChild(option);
    }
    if (prev) {
      const found = rows.find((r) => r[valueKey] === prev);
      if (found) select.value = prev;
    }
  }

  function fillProviderSelect(select, capability, preferredProvider = "") {
    const providers = uniqueProviders(byCapability(capability));
    select.innerHTML = "";
    for (const provider of providers) {
      const option = document.createElement("option");
      option.value = provider;
      option.textContent = provider;
      select.appendChild(option);
    }
    if (preferredProvider && providers.includes(preferredProvider)) {
      select.value = preferredProvider;
    }
  }

  function fillVisionProviderSelect(select, preferredProvider = "") {
    const providers = uniqueProviders(byCapability("chat.completions").filter((row) => rowSupportsVision(row)));
    select.innerHTML = "";
    for (const provider of providers) {
      const option = document.createElement("option");
      option.value = provider;
      option.textContent = provider;
      select.appendChild(option);
    }
    if (preferredProvider && providers.includes(preferredProvider)) {
      select.value = preferredProvider;
    }
  }

  function fillModelSelect(providerSelect, modelSelect, capability, opts = {}) {
    const provider = providerSelect.value;
    let rows = byProviderAndCapability(provider, capability);
    if (opts.visionOnly) {
      rows = rows.filter((row) => rowSupportsVision(row));
    }
    setSelectOptions(modelSelect, rows, "id", "label");
  }

  async function refreshModels() {
    if (state.loadingModels) return;
    state.loadingModels = true;
    setStatus("Loading models...", "");

    try {
      const url = "/v1/models/catalog?include_raw=false&include_preview=true&include_paid=true";
      const resp = await fetch(url, { headers: authHeaders(false) });
      const text = await resp.text();
      const data = text ? JSON.parse(text) : {};

      if (!resp.ok) {
        const errText = (data && (data.detail || data.error)) || `HTTP ${resp.status}`;
        throw new Error(String(errText));
      }

      state.catalogItems = Array.isArray(data.items) ? data.items : [];
      state.providerRows = Array.isArray(data.providers_status) ? data.providers_status : [];
      const fallbackApplied = Boolean(data.fallback_applied);

      refs.modelCountBadge.textContent = `${state.catalogItems.length} models`;
      refs.providerCountBadge.textContent = `${uniqueProviders(state.catalogItems).length} providers`;

      hydrateAllSelectors();
      renderModelExplorer();
      if (fallbackApplied) {
        setStatus(`Loaded ${state.catalogItems.length} fallback models (upstream unavailable)`, "ok");
      } else {
        setStatus(`Loaded ${state.catalogItems.length} models`, "ok");
      }
      showToast("Model catalog refreshed");
    } catch (err) {
      setStatus(`Model load failed: ${err.message}`, "err");
      showToast("Model catalog request failed");
    } finally {
      state.loadingModels = false;
    }
  }

  function modelObject(provider, model) {
    return [{ provider, model, priority: 0 }];
  }

  function setSimpleOptions(selectEl, values, preferred = "") {
    const clean = Array.from(new Set((values || []).map((v) => String(v || "").trim()).filter(Boolean)));
    const prev = preferred || selectEl.value || "";
    selectEl.innerHTML = "";
    for (const value of clean) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value;
      selectEl.appendChild(opt);
    }
    if (clean.includes(prev)) {
      selectEl.value = prev;
    } else if (clean.length > 0) {
      selectEl.value = clean[0];
    }
  }

  async function fetchImageOptions(provider, model, refresh = false) {
    const key = `${String(provider || "").toLowerCase()}:${String(model || "")}`;
    if (!refresh && state.imageOptionCache[key]) return state.imageOptionCache[key];
    const params = new URLSearchParams({ provider, model, refresh: refresh ? "true" : "false" });
    const resp = await fetch(`/v1/images/options?${params.toString()}`, { headers: authHeaders(false) });
    const text = await resp.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (_) {
      data = {};
    }
    if (!resp.ok) {
      const detail = (data && (data.detail || data.error)) || `HTTP ${resp.status}`;
      throw new Error(String(detail));
    }
    state.imageOptionCache[key] = data;
    return data;
  }

  function extractTextFromChatPayload(payload) {
    if (!payload || typeof payload !== "object") return "";
    const choices = payload.choices;
    if (Array.isArray(choices) && choices[0] && choices[0].message) {
      return String(choices[0].message.content || "").trim();
    }
    return "";
  }

  function extractResponseWinner(data) {
    if (!data || typeof data !== "object") return null;
    if (data.winner && typeof data.winner === "object") return data.winner;
    return null;
  }

  function extractPayloadFromEnvelope(data) {
    const winner = extractResponseWinner(data);
    if (winner && typeof winner.payload !== "undefined") return winner.payload;
    return data;
  }

  function createConsole(root) {
    const tpl = qs("#sharedConsoleTpl");
    const node = tpl.content.cloneNode(true);
    root.appendChild(node);
    const blocks = root.querySelectorAll("pre.code");
    return {
      request: blocks[0],
      response: blocks[1],
    };
  }

  function renderChatView() {
    const el = qs("#view-chat");
    el.innerHTML = `
      <section class="card stack gap-sm">
        <h4>Chat Completions</h4>
        <div class="row two">
          <label>Provider
            <select id="chatProvider"></select>
          </label>
          <label>Model
            <select id="chatModel"></select>
          </label>
        </div>
        <label>System Prompt
          <textarea id="chatSystem" placeholder="Optional behavior instructions"></textarea>
        </label>
        <label>User Message
          <textarea id="chatUser" placeholder="Write your request"></textarea>
        </label>
        <div class="row two">
          <label>Temperature
            <input id="chatTemp" type="number" min="0" max="2" step="0.1" value="0.2" />
          </label>
          <label>Max Tokens
            <input id="chatMaxTokens" type="number" min="1" step="1" value="1024" />
          </label>
        </div>
        <label class="check">
          <input id="chatStream" type="checkbox" />
          Stream output (SSE)
        </label>
        <div class="row">
          <button id="chatRun" class="btn">Run Chat</button>
          <span id="chatStatus" class="hint">Idle</span>
        </div>
      </section>
      <section class="result">
        <h5>Assistant Output</h5>
        <div id="chatOutput" class="result-text"></div>
      </section>
    `;

    const consoleRefs = createConsole(el);
    const providerEl = qs("#chatProvider");
    const modelEl = qs("#chatModel");

    providerEl.addEventListener("change", () => {
      fillModelSelect(providerEl, modelEl, "chat.completions");
      applyChatMaxTokensDefault(providerEl.value, modelEl.value, true);
    });
    modelEl.addEventListener("change", () => {
      applyChatMaxTokensDefault(providerEl.value, modelEl.value, true);
    });

    qs("#chatRun").addEventListener("click", async () => {
      const statusEl = qs("#chatStatus");
      const provider = providerEl.value;
      const model = modelEl.value;
      const system = qs("#chatSystem").value.trim();
      const user = qs("#chatUser").value.trim();
      if (!provider || !model || !user) {
        statusEl.textContent = "Provider, model, and user message are required";
        statusEl.className = "hint err";
        return;
      }

      const messages = [];
      if (system) messages.push({ role: "system", content: system });
      messages.push({ role: "user", content: user });
      const body = {
        model: modelObject(provider, model),
        messages,
        temperature: Number(qs("#chatTemp").value || 0.2),
        max_tokens: resolveChatMaxTokens(provider, model),
        stream: Boolean(qs("#chatStream")?.checked),
        x_router: defaultRouterConfig(provider),
      };

      consoleRefs.request.textContent = formatJson(body);
      statusEl.textContent = "Running...";
      statusEl.className = "hint";
      qs("#chatOutput").textContent = "";

      const t0 = performance.now();
      let liveText = "";
      const resp = body.stream
        ? await apiStreamChat("/v1/chat/completions", body, {
            onDelta: (deltaText) => {
              liveText += String(deltaText || "");
              renderMarkdown(qs("#chatOutput"), liveText || "...");
              statusEl.textContent = "Streaming...";
              statusEl.className = "hint";
            },
          })
        : await apiJson("/v1/chat/completions", body);
      const dt = (performance.now() - t0).toFixed(1);
      consoleRefs.response.textContent = formatJson(resp.data);

      if (!resp.ok) {
        statusEl.textContent = `Failed (${resp.status}) in ${dt}ms`;
        statusEl.className = "hint err";
        qs("#chatOutput").textContent = "";
        return;
      }

      const text = extractTextFromChatPayload(resp.data);
      renderMarkdown(qs("#chatOutput"), text || "(No assistant text found in response payload)");
      statusEl.textContent = `Completed in ${dt}ms`;
      statusEl.className = "hint ok";
    });
  }

  function renderVisionView() {
    const el = qs("#view-vision");
    el.innerHTML = `
      <section class="card stack gap-sm">
        <h4>Vision (Image Description)</h4>
        <div class="row two">
          <label>Provider
            <select id="visionProvider"></select>
          </label>
          <label>Model
            <select id="visionModel"></select>
          </label>
        </div>
        <label>Prompt
          <textarea id="visionPrompt" placeholder="Describe what you want to extract from the image"></textarea>
        </label>
        <label>Image File
          <input id="visionFile" type="file" accept="image/*" />
        </label>
        <label class="check">
          <input id="visionStream" type="checkbox" />
          Stream output (SSE)
        </label>
        <div class="row">
          <button id="visionRun" class="btn">Analyze Image</button>
          <span id="visionStatus" class="hint">Idle</span>
        </div>
      </section>
      <section class="grid cols-2 gap-md">
        <section class="result">
          <h5>Preview</h5>
          <div id="visionPreview" class="result-text">No image selected</div>
        </section>
        <section class="result">
          <h5>Model Output</h5>
          <div id="visionOutput" class="result-text"></div>
        </section>
      </section>
    `;

    const consoleRefs = createConsole(el);
    const providerEl = qs("#visionProvider");
    const modelEl = qs("#visionModel");

    providerEl.addEventListener("change", () => fillModelSelect(providerEl, modelEl, "chat.completions", { visionOnly: true }));

    qs("#visionFile").addEventListener("change", async (evt) => {
      const file = evt.target.files?.[0];
      if (!file) return;
      const dataUrl = await new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(String(r.result || ""));
        r.onerror = () => reject(new Error("Could not read image"));
        r.readAsDataURL(file);
      });
      state.visionImageDataUrl = dataUrl;
      qs("#visionPreview").innerHTML = `<img src="${dataUrl}" alt="vision input" style="max-width:100%;border:1px solid var(--line);border-radius:10px;" />`;
    });

    qs("#visionRun").addEventListener("click", async () => {
      const statusEl = qs("#visionStatus");
      const provider = providerEl.value;
      const model = modelEl.value;
      const prompt = qs("#visionPrompt").value.trim() || "Describe this image in detail.";
      if (!provider || !model) {
        statusEl.textContent = "No vision-capable provider/model is available right now";
        statusEl.className = "hint err";
        return;
      }
      if (!state.visionImageDataUrl) {
        statusEl.textContent = "Please select an image file first";
        statusEl.className = "hint err";
        return;
      }

      const body = {
        model: modelObject(provider, model),
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              { type: "image_url", image_url: { url: state.visionImageDataUrl } },
            ],
          },
        ],
        stream: Boolean(qs("#visionStream")?.checked),
        x_router: defaultRouterConfig(provider),
      };

      consoleRefs.request.textContent = formatJson(body);
      statusEl.textContent = "Running...";
      statusEl.className = "hint";
      const t0 = performance.now();
      let liveText = "";
      const resp = body.stream
        ? await apiStreamChat("/v1/chat/completions", body, {
            onDelta: (deltaText) => {
              liveText += String(deltaText || "");
              renderMarkdown(qs("#visionOutput"), liveText || "...");
              statusEl.textContent = "Streaming...";
              statusEl.className = "hint";
            },
          })
        : await apiJson("/v1/chat/completions", body);
      const dt = (performance.now() - t0).toFixed(1);
      consoleRefs.response.textContent = formatJson(resp.data);

      if (!resp.ok) {
        statusEl.textContent = `Failed (${resp.status}) in ${dt}ms`;
        statusEl.className = "hint err";
        return;
      }

      const text = extractTextFromChatPayload(resp.data);
      renderMarkdown(qs("#visionOutput"), text || "No text answer found");
      statusEl.textContent = `Completed in ${dt}ms`;
      statusEl.className = "hint ok";
    });
  }

  function renderResponsesView() {
    const el = qs("#view-responses");
    el.innerHTML = `
      <section class="card stack gap-sm">
        <h4>Responses</h4>
        <div class="row two">
          <label>Provider
            <select id="respProvider"></select>
          </label>
          <label>Model
            <select id="respModel"></select>
          </label>
        </div>
        <label>Input
          <textarea id="respInput" placeholder="What should the model do?"></textarea>
        </label>
        <div class="row">
          <button id="respRun" class="btn">Run Response</button>
          <span id="respStatus" class="hint">Idle</span>
        </div>
      </section>
      <section class="result">
        <h5>Output</h5>
        <div id="respOutput" class="result-text"></div>
      </section>
    `;

    const consoleRefs = createConsole(el);
    const providerEl = qs("#respProvider");
    const modelEl = qs("#respModel");

    providerEl.addEventListener("change", () => fillModelSelect(providerEl, modelEl, "responses"));

    qs("#respRun").addEventListener("click", async () => {
      const statusEl = qs("#respStatus");
      const provider = providerEl.value;
      const model = modelEl.value;
      const input = qs("#respInput").value.trim();
      if (!input) {
        statusEl.textContent = "Input is required";
        statusEl.className = "hint err";
        return;
      }

      const body = {
        model: modelObject(provider, model),
        input,
        x_router: defaultRouterConfig(provider),
      };

      consoleRefs.request.textContent = formatJson(body);
      statusEl.textContent = "Running...";
      statusEl.className = "hint";
      const t0 = performance.now();
      const resp = await apiJson("/v1/responses", body);
      const dt = (performance.now() - t0).toFixed(1);
      consoleRefs.response.textContent = formatJson(resp.data);

      if (!resp.ok) {
        statusEl.textContent = `Failed (${resp.status}) in ${dt}ms`;
        statusEl.className = "hint err";
        return;
      }

      const winnerPayload = extractPayloadFromEnvelope(resp.data);
      let textOut = "";
      if (winnerPayload && typeof winnerPayload === "object") {
        const outputText = winnerPayload.output_text;
        if (typeof outputText === "string") {
          textOut = outputText;
        }
      }
      if (!textOut) {
        textOut = formatJson(winnerPayload || resp.data);
      }
      renderMarkdown(qs("#respOutput"), textOut);
      statusEl.textContent = `Completed in ${dt}ms`;
      statusEl.className = "hint ok";
    });
  }

  function renderImageView() {
    const el = qs("#view-image");
    el.innerHTML = `
      <section class="card stack gap-sm">
        <h4>Image Generation</h4>
        <div class="row two">
          <label>Provider
            <select id="imgProvider"></select>
          </label>
          <label>Model
            <select id="imgModel"></select>
          </label>
        </div>
        <label>Prompt
          <textarea id="imgPrompt" placeholder="Describe the image"></textarea>
        </label>
        <div class="row two">
          <label>Size
            <select id="imgSize"></select>
          </label>
          <label>Quality
            <select id="imgQuality"></select>
          </label>
        </div>
        <label>Count (n)
          <input id="imgN" type="number" min="1" max="4" step="1" value="1" />
        </label>
        <div class="row">
          <button id="imgRun" class="btn">Generate</button>
          <button id="imgRefreshOpts" class="btn btn-ghost">Refresh Options</button>
          <span id="imgStatus" class="hint">Idle</span>
        </div>
      </section>
      <section class="result">
        <h5>Generated Images</h5>
        <div id="imgOutput" class="image-grid"></div>
      </section>
    `;

    const consoleRefs = createConsole(el);
    const providerEl = qs("#imgProvider");
    const modelEl = qs("#imgModel");
    const sizeEl = qs("#imgSize");
    const qualityEl = qs("#imgQuality");

    async function loadImageOptions(refresh = false) {
      const provider = providerEl.value;
      const model = modelEl.value;
      if (!provider || !model) {
        setSimpleOptions(sizeEl, ["1024x1024"], "1024x1024");
        setSimpleOptions(qualityEl, ["medium"], "medium");
        return;
      }
      try {
        const data = await fetchImageOptions(provider, model, refresh);
        const sizes = Array.isArray(data.sizes) && data.sizes.length ? data.sizes : ["1024x1024"];
        const qualities = Array.isArray(data.qualities) && data.qualities.length ? data.qualities : ["medium"];
        setSimpleOptions(sizeEl, sizes, String(data.default_size || ""));
        setSimpleOptions(qualityEl, qualities, String(data.default_quality || ""));
      } catch (err) {
        setSimpleOptions(sizeEl, ["1024x1024"], "1024x1024");
        setSimpleOptions(qualityEl, ["medium"], "medium");
      }
    }

    providerEl.addEventListener("change", async () => {
      fillModelSelect(providerEl, modelEl, "images.generations");
      await loadImageOptions(false);
    });
    modelEl.addEventListener("change", async () => {
      await loadImageOptions(false);
    });
    qs("#imgRefreshOpts").addEventListener("click", async () => {
      await loadImageOptions(true);
      showToast("Image options refreshed");
    });

    qs("#imgRun").addEventListener("click", async () => {
      const statusEl = qs("#imgStatus");
      const provider = providerEl.value;
      const model = modelEl.value;
      const prompt = qs("#imgPrompt").value.trim();
      if (!prompt) {
        statusEl.textContent = "Prompt is required";
        statusEl.className = "hint err";
        return;
      }

      const body = {
        model: modelObject(provider, model),
        prompt,
        size: qs("#imgSize").value.trim() || "1024x1024",
        quality: qs("#imgQuality").value.trim() || "medium",
        n: Math.max(1, Number(qs("#imgN").value) || 1),
        response_format: "b64_json",
        x_router: defaultRouterConfig(provider),
      };

      consoleRefs.request.textContent = formatJson(body);
      statusEl.textContent = "Generating...";
      statusEl.className = "hint";
      const t0 = performance.now();
      const resp = await apiJson("/v1/images/generations", body);
      const dt = (performance.now() - t0).toFixed(1);
      consoleRefs.response.textContent = formatJson(resp.data);

      if (!resp.ok) {
        statusEl.textContent = `Failed (${resp.status}) in ${dt}ms`;
        statusEl.className = "hint err";
        return;
      }

      const payload = extractPayloadFromEnvelope(resp.data);
      const images = (payload && payload.data) || [];
      const holder = qs("#imgOutput");
      holder.innerHTML = "";
      for (const item of images) {
        const url = String(item.url || "").trim();
        const b64 = String(item.b64_json || "").trim();
        let src = url;
        if (!src && b64) {
          src = `data:image/png;base64,${b64}`;
        }
        if (!src) continue;
        const img = document.createElement("img");
        img.src = src;
        img.alt = "generated";
        holder.appendChild(img);
      }
      if (!holder.children.length) {
        holder.textContent = "No image payload found.";
      }

      statusEl.textContent = `Completed in ${dt}ms`;
      statusEl.className = "hint ok";
    });

    setSimpleOptions(sizeEl, ["1024x1024"], "1024x1024");
    setSimpleOptions(qualityEl, ["medium"], "medium");
    void loadImageOptions(false);
  }

  function renderTtsView() {
    const el = qs("#view-tts");
    el.innerHTML = `
      <section class="card stack gap-sm">
        <h4>Text to Speech</h4>
        <div class="row two">
          <label>Provider
            <select id="ttsProvider"></select>
          </label>
          <label>Model
            <select id="ttsModel"></select>
          </label>
        </div>
        <label>Text
          <textarea id="ttsInput" placeholder="Text to synthesize"></textarea>
        </label>
        <div class="row two">
          <label>Voice
            <input id="ttsVoice" type="text" value="diana" />
          </label>
          <label>Format
            <input id="ttsFormat" type="text" value="wav" />
          </label>
        </div>
        <div class="row">
          <button id="ttsRun" class="btn">Generate Speech</button>
          <span id="ttsStatus" class="hint">Idle</span>
        </div>
      </section>
      <section class="result audio-box">
        <h5>Audio Output</h5>
        <audio id="ttsAudio" controls></audio>
        <a id="ttsDownload" class="btn btn-ghost hidden" download="speech.wav">Download Audio</a>
      </section>
    `;

    const consoleRefs = createConsole(el);
    const providerEl = qs("#ttsProvider");
    const modelEl = qs("#ttsModel");
    providerEl.addEventListener("change", () => fillModelSelect(providerEl, modelEl, "audio.speech"));

    qs("#ttsRun").addEventListener("click", async () => {
      const statusEl = qs("#ttsStatus");
      const provider = providerEl.value;
      const model = modelEl.value;
      const text = qs("#ttsInput").value.trim();
      if (!text) {
        statusEl.textContent = "Text is required";
        statusEl.className = "hint err";
        return;
      }

      const body = {
        model: modelObject(provider, model),
        input: text,
        voice: qs("#ttsVoice").value.trim() || "diana",
        response_format: qs("#ttsFormat").value.trim() || "wav",
        x_router: defaultRouterConfig(provider),
      };
      consoleRefs.request.textContent = formatJson(body);
      statusEl.textContent = "Generating...";
      statusEl.className = "hint";

      const t0 = performance.now();
      const resp = await fetch("/v1/audio/speech", {
        method: "POST",
        headers: authHeaders(true),
        body: JSON.stringify(body),
      });
      const dt = (performance.now() - t0).toFixed(1);

      const ctype = String(resp.headers.get("content-type") || "").toLowerCase();
      if (!resp.ok || !ctype.startsWith("audio/")) {
        let errPayload = null;
        try {
          errPayload = await resp.json();
        } catch (_) {
          errPayload = { raw: await resp.text() };
        }
        consoleRefs.response.textContent = formatJson(errPayload);
        statusEl.textContent = `Failed (${resp.status}) in ${dt}ms`;
        statusEl.className = "hint err";
        return;
      }

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const audio = qs("#ttsAudio");
      const dl = qs("#ttsDownload");
      audio.src = url;
      dl.href = url;
      dl.classList.remove("hidden");

      consoleRefs.response.textContent = formatJson({
        ok: true,
        status: resp.status,
        content_type: ctype,
        size_bytes: blob.size,
      });

      statusEl.textContent = `Completed in ${dt}ms`;
      statusEl.className = "hint ok";
    });
  }

  function renderSttView() {
    const el = qs("#view-stt");
    el.innerHTML = `
      <section class="card stack gap-sm">
        <h4>Speech to Text</h4>
        <label>Audio File
          <input id="sttFile" type="file" accept="audio/*" />
        </label>
        <label>Language (optional)
          <input id="sttLang" type="text" value="fa" />
        </label>
        <div class="row">
          <button id="sttRun" class="btn">Transcribe</button>
          <span id="sttStatus" class="hint">Idle</span>
        </div>
      </section>
      <section class="result">
        <h5>Transcript</h5>
        <div id="sttOutput" class="result-text"></div>
      </section>
    `;

    const consoleRefs = createConsole(el);

    qs("#sttRun").addEventListener("click", async () => {
      const statusEl = qs("#sttStatus");
      const file = qs("#sttFile").files?.[0];
      if (!file) {
        statusEl.textContent = "Please select an audio file";
        statusEl.className = "hint err";
        return;
      }

      const lang = qs("#sttLang").value.trim();
      const form = new FormData();
      form.append("file", file);
      if (lang) {
        form.append("language", lang);
      }

      const reqMeta = { endpoint: "/v1/audio/transcriptions", filename: file.name, language: lang || null };
      consoleRefs.request.textContent = formatJson(reqMeta);
      statusEl.textContent = "Transcribing...";
      statusEl.className = "hint";

      const t0 = performance.now();
      const resp = await fetch("/v1/audio/transcriptions", {
        method: "POST",
        headers: authHeaders(false),
        body: form,
      });
      const dt = (performance.now() - t0).toFixed(1);

      let data = null;
      try {
        data = await resp.json();
      } catch (_) {
        data = { raw: await resp.text() };
      }
      consoleRefs.response.textContent = formatJson(data);

      if (!resp.ok) {
        statusEl.textContent = `Failed (${resp.status}) in ${dt}ms`;
        statusEl.className = "hint err";
        return;
      }

      const out = data?.payload?.text || data?.payload?.raw?.text || data?.text || "";
      renderMarkdown(qs("#sttOutput"), out || "No transcript text found");
      statusEl.textContent = `Completed in ${dt}ms`;
      statusEl.className = "hint ok";
    });
  }

  function renderEmbeddingsView() {
    const el = qs("#view-embeddings");
    el.innerHTML = `
      <section class="card stack gap-sm">
        <h4>Embeddings</h4>
        <div class="row two">
          <label>Provider
            <select id="embProvider"></select>
          </label>
          <label>Model
            <select id="embModel"></select>
          </label>
        </div>
        <label>Input Text
          <textarea id="embInput" placeholder="Text for vector embedding"></textarea>
        </label>
        <div class="row">
          <button id="embRun" class="btn">Generate Embedding</button>
          <span id="embStatus" class="hint">Idle</span>
        </div>
      </section>
      <section class="result">
        <h5>Embedding Summary</h5>
        <div id="embOutput" class="result-text"></div>
      </section>
    `;

    const consoleRefs = createConsole(el);
    const providerEl = qs("#embProvider");
    const modelEl = qs("#embModel");
    providerEl.addEventListener("change", () => fillModelSelect(providerEl, modelEl, "embeddings"));

    qs("#embRun").addEventListener("click", async () => {
      const statusEl = qs("#embStatus");
      const provider = providerEl.value;
      const model = modelEl.value;
      const input = qs("#embInput").value.trim();
      if (!input) {
        statusEl.textContent = "Input text is required";
        statusEl.className = "hint err";
        return;
      }

      const body = {
        model: modelObject(provider, model),
        input,
        x_router: defaultRouterConfig(provider),
      };

      consoleRefs.request.textContent = formatJson(body);
      statusEl.textContent = "Running...";
      statusEl.className = "hint";
      const t0 = performance.now();
      const resp = await apiJson("/v1/embeddings", body);
      const dt = (performance.now() - t0).toFixed(1);
      consoleRefs.response.textContent = formatJson(resp.data);

      if (!resp.ok) {
        statusEl.textContent = `Failed (${resp.status}) in ${dt}ms`;
        statusEl.className = "hint err";
        return;
      }

      const payload = extractPayloadFromEnvelope(resp.data) || {};
      let vector = null;
      if (Array.isArray(payload.data) && payload.data[0] && payload.data[0].embedding) {
        vector = payload.data[0].embedding;
      } else if (payload.embedding && Array.isArray(payload.embedding.values)) {
        vector = payload.embedding.values;
      }

      if (Array.isArray(vector)) {
        const first = vector.slice(0, 12).map((v) => Number(v).toFixed(6)).join(", ");
        qs("#embOutput").textContent = `Dimension: ${vector.length}\nFirst values: [${first}]`;
      } else {
        qs("#embOutput").textContent = "Could not infer vector from payload";
      }

      statusEl.textContent = `Completed in ${dt}ms`;
      statusEl.className = "hint ok";
    });
  }

  function renderOrchestrateView() {
    const el = qs("#view-orchestrate");
    el.innerHTML = `
      <section class="card stack gap-sm">
        <h4>Orchestrate (Advanced)</h4>
        <label>Capability
          <select id="orcCapability">
            <option value="chat.completions">chat.completions</option>
            <option value="responses">responses</option>
            <option value="embeddings">embeddings</option>
            <option value="images.generations">images.generations</option>
            <option value="audio.speech">audio.speech</option>
          </select>
        </label>
        <label>Payload JSON
          <textarea id="orcPayload">{
  "model": [{"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 0}],
  "messages": [{"role": "user", "content": "سلام"}]
}</textarea>
        </label>
        <label>x_router JSON (optional)
          <textarea id="orcRouter">{
  "strategy": "fallback_chain",
  "mode": "limit_safe",
  "timeout_sec": 25,
  "max_attempts": 6
}</textarea>
        </label>
        <div class="row">
          <button id="orcRun" class="btn">Run Orchestrate</button>
          <span id="orcStatus" class="hint">Idle</span>
        </div>
      </section>
    `;

    const consoleRefs = createConsole(el);

    qs("#orcRun").addEventListener("click", async () => {
      const statusEl = qs("#orcStatus");
      try {
        const capability = qs("#orcCapability").value;
        const payload = JSON.parse(qs("#orcPayload").value);
        const routerRaw = qs("#orcRouter").value.trim();
        const xRouter = routerRaw ? JSON.parse(routerRaw) : null;

        const body = {
          capability,
          payload,
          x_router: xRouter,
        };

        consoleRefs.request.textContent = formatJson(body);
        statusEl.textContent = "Running...";
        statusEl.className = "hint";
        const t0 = performance.now();
        const resp = await apiJson("/v1/orchestrate", body);
        const dt = (performance.now() - t0).toFixed(1);
        consoleRefs.response.textContent = formatJson(resp.data);

        if (!resp.ok) {
          statusEl.textContent = `Failed (${resp.status}) in ${dt}ms`;
          statusEl.className = "hint err";
          return;
        }

        statusEl.textContent = `Completed in ${dt}ms`;
        statusEl.className = "hint ok";
      } catch (err) {
        statusEl.textContent = `Invalid JSON: ${err.message}`;
        statusEl.className = "hint err";
      }
    });
  }

  function renderModelExplorer() {
    const el = qs("#view-models");
    if (!el) return;

    const rows = state.catalogItems.slice().sort((a, b) => {
      const left = `${a.provider}/${a.id}`;
      const right = `${b.provider}/${b.id}`;
      return left.localeCompare(right);
    });

    const providerOptions = ["<option value=''>all</option>"]
      .concat(uniqueProviders(rows).map((p) => `<option value="${p}">${p}</option>`))
      .join("");

    el.innerHTML = `
      <section class="card stack gap-sm">
        <h4>Catalog Filters</h4>
        <div class="row two">
          <label>Provider
            <select id="mdlProviderFilter">${providerOptions}</select>
          </label>
          <label>Capability
            <select id="mdlCapFilter">
              <option value="">all</option>
              <option value="chat.completions">chat.completions</option>
              <option value="responses">responses</option>
              <option value="embeddings">embeddings</option>
              <option value="images.generations">images.generations</option>
              <option value="audio.speech">audio.speech</option>
              <option value="audio.transcriptions">audio.transcriptions</option>
            </select>
          </label>
        </div>
      </section>
      <section class="model-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Provider</th>
              <th>Model</th>
              <th>Type</th>
              <th>Family</th>
              <th>Capabilities</th>
              <th>Input</th>
              <th>Output</th>
            </tr>
          </thead>
          <tbody id="mdlRows"></tbody>
        </table>
      </section>
    `;

    function drawRows() {
      const provider = qs("#mdlProviderFilter").value;
      const cap = qs("#mdlCapFilter").value;
      const target = qs("#mdlRows");
      target.innerHTML = "";

      const filtered = rows.filter((row) => {
        if (provider && row.provider !== provider) return false;
        if (cap && !(row.capabilities || []).includes(cap)) return false;
        return true;
      });

      for (const row of filtered) {
        const tr = document.createElement("tr");
        const caps = (row.capabilities || []).map((c) => `<span class="cap">${c}</span>`).join(" ");
        tr.innerHTML = `
          <td>${row.provider}</td>
          <td>${row.id}</td>
          <td>${row.model_type || ""}</td>
          <td>${row.family || ""}</td>
          <td><div class="cap-list">${caps}</div></td>
          <td>${(row.input_modalities || []).join(", ")}</td>
          <td>${(row.output_modalities || []).join(", ")}</td>
        `;
        target.appendChild(tr);
      }
    }

    qs("#mdlProviderFilter").addEventListener("change", drawRows);
    qs("#mdlCapFilter").addEventListener("change", drawRows);
    drawRows();
  }

  function hydrateAllSelectors() {
    const specs = [
      ["#chatProvider", "#chatModel", "chat.completions"],
      ["#respProvider", "#respModel", "responses"],
      ["#imgProvider", "#imgModel", "images.generations"],
      ["#ttsProvider", "#ttsModel", "audio.speech"],
      ["#embProvider", "#embModel", "embeddings"],
    ];

    for (const [providerSel, modelSel, capability] of specs) {
      const providerEl = qs(providerSel);
      const modelEl = qs(modelSel);
      if (!providerEl || !modelEl) continue;
      const preferred = providerEl.value || "";
      fillProviderSelect(providerEl, capability, preferred);
      fillModelSelect(providerEl, modelEl, capability);
      if (providerSel === "#imgProvider") {
        providerEl.dispatchEvent(new Event("change"));
      }
    }

    const visionProvider = qs("#visionProvider");
    const visionModel = qs("#visionModel");
    if (visionProvider && visionModel) {
      const preferredVisionProvider = visionProvider.value || "";
      fillVisionProviderSelect(visionProvider, preferredVisionProvider);
      fillModelSelect(visionProvider, visionModel, "chat.completions", { visionOnly: true });
    }

    const chatProvider = qs("#chatProvider");
    const chatModel = qs("#chatModel");
    if (chatProvider && chatModel) {
      applyChatMaxTokensDefault(chatProvider.value, chatModel.value, true);
    }
  }

  function renderTabMenu() {
    refs.navTabs.innerHTML = "";
    for (const tab of tabs) {
      const btn = document.createElement("button");
      btn.textContent = tab.label;
      btn.dataset.tab = tab.id;
      btn.className = tab.id === state.activeTab ? "active" : "";
      btn.addEventListener("click", () => switchTab(tab.id));
      refs.navTabs.appendChild(btn);
    }
  }

  function switchTab(tabId) {
    state.activeTab = tabId;
    qsa(".view").forEach((view) => view.classList.add("hidden"));
    const current = qs(`#view-${tabId}`);
    if (current) current.classList.remove("hidden");

    const meta = tabs.find((t) => t.id === tabId);
    if (meta) {
      refs.viewTitle.textContent = meta.label;
      refs.viewDesc.textContent = meta.desc;
    }
    renderTabMenu();
  }

  function renderAllViews() {
    renderChatView();
    renderVisionView();
    renderResponsesView();
    renderImageView();
    renderTtsView();
    renderSttView();
    renderEmbeddingsView();
    renderOrchestrateView();
    renderModelExplorer();
  }

  async function bootstrap() {
    loadSettings();
    renderAllViews();
    renderTabMenu();
    switchTab(state.activeTab);

    refs.saveSettingsBtn.addEventListener("click", () => {
      saveSettings();
      setStatus("Settings saved", "ok");
      showToast("Studio settings saved");
    });

    refs.loadModelsBtn.addEventListener("click", refreshModels);

    await refreshModels();
  }

  bootstrap();
})();
