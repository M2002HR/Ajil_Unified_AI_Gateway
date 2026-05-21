const PAGE_META = {
  dashboard: {
    title: "Overview",
    desc: "Realtime service health, success rate, latency and throughput",
  },
  events: {
    title: "Event Stream",
    desc: "Live event logs with server-side and client-side filtering",
  },
  usage: {
    title: "Usage Analytics",
    desc: "Per provider/model/key usage, token accounting and reliability",
  },
  http: {
    title: "HTTP Metrics",
    desc: "Request distribution, error codes and endpoint latency profiles",
  },
  router: {
    title: "Router Insights",
    desc: "Load-balancer scores, provider outcomes and candidate quality",
  },
  keys: {
    title: "Key Limits",
    desc: "Latest rate-limit headers and key-level pressure visibility",
  },
};

const MENU_ITEMS = [
  ["dashboard", "Overview"],
  ["events", "Event Stream"],
  ["usage", "Usage"],
  ["http", "HTTP"],
  ["router", "Router"],
  ["keys", "Key Limits"],
];

const state = {
  token: "",
  headerName: "x-admin-token",
  currentPage: "dashboard",
  sinceMinutes: 60,
  ws: null,
  wsPaused: false,
  wsStatus: "connecting",
  reconnectTimer: null,
  reconnectAttempt: 0,
  lastTick: "",
  maxEventRows: 5000,
  eventsBySeq: new Map(),
  events: [],
  charts: new Map(),
  layouts: {},
  live: {
    summaryEnvelope: null,
    summary: null,
    usageOverview: null,
    usageByProvider: null,
    usageByModel: null,
    usageByKey: null,
    httpStats: null,
    routerScores: null,
    keyLimits: null,
    health: null,
  },
  tabs: {
    usageAggregate: null,
    usageEvents: null,
    httpStats: null,
    keyLimits: null,
    eventSnapshot: null,
  },
  filters: {
    events: {
      level: "",
      eventType: "",
      requestId: "",
      search: "",
      limit: 300,
    },
    usage: {
      groupBy: "provider",
      provider: "",
      model: "",
      capability: "",
      limit: 200,
    },
    http: {
      groupBy: "path",
    },
    keys: {
      provider: "",
    },
  },
};

const els = {
  app: document.getElementById("app"),
  loginView: document.getElementById("loginView"),
  loginForm: document.getElementById("loginForm"),
  adminToken: document.getElementById("adminToken"),
  adminHeader: document.getElementById("adminHeader"),
  loginError: document.getElementById("loginError"),
  menu: document.getElementById("menu"),
  logoutBtn: document.getElementById("logoutBtn"),
  pageTitle: document.getElementById("pageTitle"),
  pageDesc: document.getElementById("pageDesc"),
  sinceSelect: document.getElementById("sinceSelect"),
  sinceLabel: document.getElementById("sinceLabel"),
  refreshBtn: document.getElementById("refreshBtn"),
  pauseWsBtn: document.getElementById("pauseWsBtn"),
  wsStatus: document.getElementById("wsStatus"),
  lastTick: document.getElementById("lastTick"),
  toastRoot: document.getElementById("toastRoot"),
  modalOverlay: document.getElementById("modalOverlay"),
  modalTitle: document.getElementById("modalTitle"),
  modalBody: document.getElementById("modalBody"),
  modalCloseBtn: document.getElementById("modalCloseBtn"),
};

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmtInt(value) {
  const n = Number(value || 0);
  if (!Number.isFinite(n)) return "0";
  return Math.round(n).toLocaleString();
}

function fmtFloat(value, digits = 2) {
  const n = Number(value || 0);
  if (!Number.isFinite(n)) return "0";
  return n.toFixed(digits);
}

function fmtPct(value) {
  const n = Number(value || 0) * 100;
  if (!Number.isFinite(n)) return "0.00%";
  return `${n.toFixed(2)}%`;
}

function fmtMs(value) {
  const n = Number(value || 0);
  if (!Number.isFinite(n)) return "0ms";
  return `${n.toFixed(2)}ms`;
}

function formatTime(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString();
}

function compactJson(value, maxLen = 180) {
  let out = "";
  try {
    out = JSON.stringify(value ?? {});
  } catch {
    out = String(value ?? "");
  }
  if (out.length <= maxLen) return out;
  return `${out.slice(0, maxLen)}...`;
}

function topN(items, n = 12) {
  return (items || []).slice(0, n);
}

function toQuery(params = {}) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    const text = String(v).trim();
    if (!text) continue;
    q.set(k, text);
  }
  return q.toString();
}

function toast(message, isError = false) {
  const node = document.createElement("div");
  node.className = `toast${isError ? " err" : ""}`;
  node.textContent = String(message || "");
  els.toastRoot.appendChild(node);
  requestAnimationFrame(() => node.classList.add("show"));
  setTimeout(() => {
    node.classList.remove("show");
    setTimeout(() => node.remove(), 240);
  }, 3200);
}

function openModal(title, contentObj) {
  els.modalTitle.textContent = title;
  els.modalBody.innerHTML = `<pre>${esc(JSON.stringify(contentObj, null, 2))}</pre>`;
  els.modalOverlay.classList.remove("hidden");
  els.modalOverlay.setAttribute("aria-hidden", "false");
}

function closeModal() {
  els.modalOverlay.classList.add("hidden");
  els.modalOverlay.setAttribute("aria-hidden", "true");
}

function setWsStatus(kind, text) {
  const cls = kind === "ok" ? "ok" : kind === "err" ? "err" : "warn";
  state.wsStatus = cls;
  els.wsStatus.className = `badge ${cls}`;
  els.wsStatus.textContent = text;
}

function storeAuth() {
  localStorage.setItem("uag_admin_token", state.token);
  localStorage.setItem("uag_admin_header", state.headerName);
}

function loadAuth() {
  state.token = localStorage.getItem("uag_admin_token") || "";
  state.headerName = localStorage.getItem("uag_admin_header") || "x-admin-token";
  if (state.token) {
    els.adminToken.value = state.token;
  }
  els.adminHeader.value = state.headerName;
}

function clearAuth() {
  localStorage.removeItem("uag_admin_token");
  localStorage.removeItem("uag_admin_header");
  state.token = "";
}

function apiHeaders() {
  return {
    [state.headerName]: state.token,
  };
}

async function apiGet(path, params = {}) {
  const query = toQuery(params);
  const url = query ? `${path}?${query}` : path;
  const resp = await fetch(url, {
    method: "GET",
    headers: apiHeaders(),
  });

  const text = await resp.text();
  let parsed;
  try {
    parsed = text ? JSON.parse(text) : {};
  } catch {
    parsed = { raw: text };
  }

  if (!resp.ok) {
    const message = parsed && typeof parsed === "object" ? parsed.detail || parsed.error || JSON.stringify(parsed) : String(text || "Request failed");
    throw new Error(`${resp.status} ${resp.statusText}: ${message}`);
  }
  return parsed;
}

function mergeEvents(items) {
  if (!Array.isArray(items)) return;
  for (const item of items) {
    const seq = Number(item?.seq || 0);
    if (!Number.isFinite(seq) || seq <= 0) continue;
    state.eventsBySeq.set(seq, item);
  }

  const rows = Array.from(state.eventsBySeq.entries())
    .sort((a, b) => b[0] - a[0])
    .map((entry) => entry[1]);
  if (rows.length > state.maxEventRows) {
    const keep = rows.slice(0, state.maxEventRows);
    state.eventsBySeq = new Map(keep.map((item) => [Number(item.seq || 0), item]));
    state.events = keep;
  } else {
    state.events = rows;
  }
}

function ensureChart(id) {
  const el = document.getElementById(id);
  if (!el || typeof echarts === "undefined") return null;

  const existing = state.charts.get(id);
  if (existing && existing.getDom() === el) return existing;

  if (existing) {
    try {
      existing.dispose();
    } catch {
      // no-op
    }
  }

  const next = echarts.init(el);
  state.charts.set(id, next);
  return next;
}

function setChartOption(id, option) {
  const chart = ensureChart(id);
  if (!chart) return;
  chart.setOption(option, { notMerge: true, lazyUpdate: true });
}

function resizeCharts() {
  for (const chart of state.charts.values()) {
    try {
      chart.resize();
    } catch {
      // no-op
    }
  }
}

function debounce(fn, waitMs = 220) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), waitMs);
  };
}

async function refreshCoreData() {
  const since = state.sinceMinutes;
  const [summaryEnvelope, eventSnapshot, health] = await Promise.all([
    apiGet("/admin/logs/summary", { since_minutes: since }),
    apiGet("/admin/logs/events", { since_minutes: since, limit: state.filters.events.limit }),
    apiGet("/health"),
  ]);

  state.live.summaryEnvelope = summaryEnvelope;
  state.live.summary = summaryEnvelope.events || null;
  state.live.usageOverview = summaryEnvelope.usage_overview || null;
  state.live.usageByProvider = summaryEnvelope.usage_by_provider || null;
  state.live.usageByModel = summaryEnvelope.usage_by_model || null;
  state.live.usageByKey = summaryEnvelope.usage_by_key || null;
  state.live.routerScores = summaryEnvelope.router_scores || null;
  state.live.health = health;

  state.tabs.eventSnapshot = eventSnapshot;
  mergeEvents(eventSnapshot.items || []);
}

async function refreshUsageTabData() {
  const since = state.sinceMinutes;
  const f = state.filters.usage;

  const [aggregate, events] = await Promise.all([
    apiGet("/admin/usage/aggregate", {
      group_by: f.groupBy,
      since_minutes: since,
      provider: f.provider,
      model: f.model,
      capability: f.capability,
    }),
    apiGet("/admin/usage/events", {
      since_minutes: since,
      provider: f.provider,
      model: f.model,
      capability: f.capability,
      limit: f.limit,
    }),
  ]);

  state.tabs.usageAggregate = aggregate;
  state.tabs.usageEvents = events;
}

async function refreshHttpTabData() {
  state.tabs.httpStats = await apiGet("/admin/logs/http-stats", {
    since_minutes: state.sinceMinutes,
    group_by: state.filters.http.groupBy,
  });
}

async function refreshKeysTabData() {
  const provider = state.filters.keys.provider;
  const [limits, keyUsage] = await Promise.all([
    apiGet("/admin/usage/key-limits/latest", { provider }),
    apiGet("/admin/usage/aggregate", {
      group_by: "key",
      since_minutes: state.sinceMinutes,
      provider,
    }),
  ]);
  state.tabs.keyLimits = limits;
  state.live.usageByKey = keyUsage;
}

async function refreshPageData(page) {
  if (page === "usage") {
    await refreshUsageTabData();
  }
  if (page === "http") {
    await refreshHttpTabData();
  }
  if (page === "keys") {
    await refreshKeysTabData();
  }
  if (page === "router") {
    state.live.routerScores = await apiGet("/admin/router/stats");
  }
  if (page === "events") {
    state.tabs.eventSnapshot = await apiGet("/admin/logs/events", {
      since_minutes: state.sinceMinutes,
      level: state.filters.events.level,
      event_type: state.filters.events.eventType,
      request_id: state.filters.events.requestId,
      limit: state.filters.events.limit,
    });
    mergeEvents(state.tabs.eventSnapshot.items || []);
  }
}

function wsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const query = new URLSearchParams({
    token: state.token,
    since_minutes: String(state.sinceMinutes),
    limit: String(state.filters.events.limit),
  });
  return `${protocol}://${window.location.host}/ws/admin?${query.toString()}`;
}

function clearReconnect() {
  if (state.reconnectTimer) {
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
  }
}

function scheduleReconnect() {
  clearReconnect();
  const wait = Math.min(15000, 1200 * (state.reconnectAttempt + 1));
  state.reconnectTimer = setTimeout(() => {
    connectWs();
  }, wait);
}

function sendWs(obj) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.ws.send(JSON.stringify(obj));
}

function applyWsFilters() {
  sendWs({
    type: "filters",
    since_minutes: state.sinceMinutes,
    limit: state.filters.events.limit,
    level: state.filters.events.level,
    event_type: state.filters.events.eventType,
    request_id_filter: state.filters.events.requestId,
    http_group_by: state.filters.http.groupBy,
  });
}

function applyWsScopes() {
  sendWs({
    type: "subscribe",
    scopes: ["summary", "events", "usage", "http", "router"],
  });
}

function connectWs() {
  if (state.ws) {
    try {
      state.ws.close();
    } catch {
      // no-op
    }
  }

  clearReconnect();
  setWsStatus("warn", "connecting");
  const socket = new WebSocket(wsUrl());
  state.ws = socket;

  socket.addEventListener("open", () => {
    state.reconnectAttempt = 0;
    setWsStatus("ok", "live");
    applyWsScopes();
    applyWsFilters();
  });

  socket.addEventListener("message", (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data || "{}");
    } catch {
      return;
    }

    if (payload?.type === "error") {
      toast(payload.error || "WebSocket error", true);
      return;
    }

    if (payload?.type === "hello" || payload?.type === "tick") {
      if (state.wsPaused) return;
      handleWsSnapshot(payload);
    }
  });

  socket.addEventListener("close", () => {
    setWsStatus("warn", "reconnecting");
    state.reconnectAttempt += 1;
    scheduleReconnect();
  });

  socket.addEventListener("error", () => {
    setWsStatus("err", "error");
  });
}

function handleWsSnapshot(payload) {
  state.lastTick = payload.server_time || "";
  els.lastTick.textContent = formatTime(state.lastTick);

  if (payload.summary) state.live.summary = payload.summary;
  if (payload.usage_overview) state.live.usageOverview = payload.usage_overview;
  if (payload.usage_by_provider) state.live.usageByProvider = payload.usage_by_provider;
  if (payload.usage_by_model) state.live.usageByModel = payload.usage_by_model;
  if (payload.usage_by_key) state.live.usageByKey = payload.usage_by_key;
  if (payload.http) state.live.httpStats = payload.http;
  if (payload.router_scores) state.live.routerScores = payload.router_scores;

  if (payload.events && Array.isArray(payload.events.items)) {
    mergeEvents(payload.events.items);
  }

  renderCurrentPage();
}

function buildMenu() {
  els.menu.innerHTML = MENU_ITEMS.map(([id, label]) => `<button data-page="${id}">${esc(label)}</button>`).join("");
  els.menu.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const page = btn.getAttribute("data-page") || "dashboard";
      await setPage(page);
    });
  });
}

function setMenuActive(page) {
  els.menu.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-page") === page);
  });
}

async function setPage(page) {
  if (!PAGE_META[page]) return;
  state.currentPage = page;
  setMenuActive(page);

  Object.keys(PAGE_META).forEach((name) => {
    const el = document.getElementById(`page-${name}`);
    if (!el) return;
    el.classList.toggle("hidden", name !== page);
  });

  const meta = PAGE_META[page];
  els.pageTitle.textContent = meta.title;
  els.pageDesc.textContent = meta.desc;

  await refreshPageData(page).catch((err) => {
    toast(String(err.message || err), true);
  });
  renderCurrentPage();
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = String(text ?? "");
}

function ensureDashboardLayout() {
  if (state.layouts.dashboard) return;
  const root = document.getElementById("page-dashboard");
  root.innerHTML = `
    <div class="grid cols-5">
      <div class="card"><h3>Total HTTP</h3><div id="kpiHttpTotal" class="stat">0</div><div class="sub">requests in window</div></div>
      <div class="card"><h3>HTTP Success</h3><div id="kpiHttpSuccessRate" class="stat">0%</div><div class="sub">2xx + 3xx success rate</div></div>
      <div class="card"><h3>Avg Latency</h3><div id="kpiHttpLatency" class="stat">0ms</div><div class="sub">across completed requests</div></div>
      <div class="card"><h3>Usage Requests</h3><div id="kpiUsageReq" class="stat">0</div><div class="sub">provider attempts</div></div>
      <div class="card"><h3>Provider Count</h3><div id="kpiProviderCount" class="stat">0</div><div class="sub">active providers seen</div></div>
    </div>

    <div class="grid cols-2" style="margin-top:10px">
      <div class="chart-card"><h3>HTTP Status Distribution</h3><div id="chartDashboardStatus" class="chart"></div></div>
      <div class="chart-card"><h3>Events By Level</h3><div id="chartDashboardLevels" class="chart"></div></div>
    </div>

    <div class="grid cols-2" style="margin-top:10px">
      <div class="chart-card"><h3>Top Providers By Requests</h3><div id="chartDashboardProviders" class="chart small"></div></div>
      <div class="chart-card"><h3>Top Models Avg Latency</h3><div id="chartDashboardModelLatency" class="chart small"></div></div>
    </div>

    <div class="card" style="margin-top:10px">
      <div class="row" style="justify-content:space-between">
        <h3 style="margin:0">Recent Events</h3>
        <span class="muted">live stream + snapshot cache</span>
      </div>
      <div class="table-wrap" style="margin-top:8px">
        <table>
          <thead><tr><th>Time</th><th>Seq</th><th>Level</th><th>Type</th><th>Message</th><th>Details</th></tr></thead>
          <tbody id="dashEventsBody"></tbody>
        </table>
      </div>
    </div>
  `;
  state.layouts.dashboard = true;
}

function renderDashboard() {
  ensureDashboardLayout();

  const summary = state.live.summary || {};
  const http = summary.http || {};
  const usageOverview = state.live.usageOverview || {};
  const overall = usageOverview.overall || {};

  setText("kpiHttpTotal", fmtInt(http.requests_total));
  setText("kpiHttpSuccessRate", fmtPct(http.success_rate));
  setText("kpiHttpLatency", fmtMs(http.latency_avg_ms));
  setText("kpiUsageReq", fmtInt(overall.requests_total));
  setText("kpiProviderCount", fmtInt(usageOverview.providers_count));

  const statusSeries = [
    { value: Number(http.status_2xx || 0), name: "2xx" },
    { value: Number(http.status_3xx || 0), name: "3xx" },
    { value: Number(http.status_4xx || 0), name: "4xx" },
    { value: Number(http.status_5xx || 0), name: "5xx" },
  ];

  setChartOption("chartDashboardStatus", {
    tooltip: { trigger: "item" },
    legend: { bottom: 0 },
    series: [
      {
        type: "pie",
        radius: ["42%", "74%"],
        center: ["50%", "45%"],
        data: statusSeries,
        label: { formatter: "{b}: {c}" },
      },
    ],
  });

  const byLevel = summary.by_level || {};
  const levelNames = Object.keys(byLevel);
  setChartOption("chartDashboardLevels", {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: levelNames },
    yAxis: { type: "value" },
    series: [
      {
        type: "bar",
        data: levelNames.map((key) => Number(byLevel[key] || 0)),
        itemStyle: { color: "#ff8f35", borderRadius: [6, 6, 0, 0] },
      },
    ],
  });

  const providerRows = topN(state.live.usageByProvider?.items || [], 10);
  setChartOption("chartDashboardProviders", {
    tooltip: { trigger: "axis" },
    grid: { left: 70, right: 20, top: 18, bottom: 24 },
    xAxis: { type: "value" },
    yAxis: { type: "category", data: providerRows.map((r) => r.group), inverse: true },
    series: [
      {
        type: "bar",
        data: providerRows.map((r) => Number(r.requests_total || 0)),
        itemStyle: { color: "#ff9f45", borderRadius: [0, 6, 6, 0] },
      },
    ],
  });

  const modelRows = topN(state.live.usageByModel?.items || [], 10);
  setChartOption("chartDashboardModelLatency", {
    tooltip: { trigger: "axis" },
    grid: { left: 70, right: 20, top: 18, bottom: 24 },
    xAxis: { type: "value" },
    yAxis: { type: "category", data: modelRows.map((r) => r.group), inverse: true },
    series: [
      {
        type: "bar",
        data: modelRows.map((r) => Number(r.latency_avg_ms || 0)),
        itemStyle: { color: "#f57838", borderRadius: [0, 6, 6, 0] },
      },
    ],
  });

  const rows = topN(state.events, 25)
    .map((event) => {
      const level = String(event.level || "").toUpperCase();
      const badgeCls = level === "ERROR" ? "err" : level === "WARNING" ? "warn" : "ok";
      return `
        <tr>
          <td>${esc(formatTime(event.ts))}</td>
          <td>${esc(event.seq)}</td>
          <td><span class="badge ${badgeCls}">${esc(level || "INFO")}</span></td>
          <td>${esc(event.event_type || "")}</td>
          <td>${esc(event.message || "")}</td>
          <td><button class="btn btn-ghost" data-detail-seq="${esc(event.seq)}">view</button></td>
        </tr>
      `;
    })
    .join("");
  const body = document.getElementById("dashEventsBody");
  body.innerHTML = rows || `<tr><td colspan="6" class="muted">No events</td></tr>`;
  body.querySelectorAll("button[data-detail-seq]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const seq = Number(btn.getAttribute("data-detail-seq") || 0);
      const item = state.eventsBySeq.get(seq);
      if (item) openModal(`Event #${seq}`, item);
    });
  });
}

function ensureEventsLayout() {
  if (state.layouts.events) return;
  const root = document.getElementById("page-events");
  root.innerHTML = `
    <div class="card">
      <div class="filters">
        <label>Level
          <select id="eventsLevel">
            <option value="">All</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
          </select>
        </label>
        <label>Event Type
          <input id="eventsType" type="text" placeholder="router.provider.attempt" />
        </label>
        <label>Request ID
          <input id="eventsRequestId" type="text" placeholder="request id" />
        </label>
        <label>Search
          <input id="eventsSearch" type="text" placeholder="text in message/data" />
        </label>
        <label>Max Rows
          <select id="eventsLimit">
            <option value="100">100</option>
            <option value="300" selected>300</option>
            <option value="700">700</option>
            <option value="1200">1200</option>
          </select>
        </label>
      </div>
      <div class="row" style="margin-top:10px">
        <button id="eventsApplyBtn" class="btn">Apply Filters</button>
        <button id="eventsClearBtn" class="btn btn-ghost">Clear</button>
        <span class="muted">Filters apply to websocket and manual snapshot endpoint</span>
      </div>
    </div>

    <div class="grid cols-2" style="margin-top:10px">
      <div class="chart-card"><h3>Top Event Types</h3><div id="chartEventsTypes" class="chart small"></div></div>
      <div class="chart-card"><h3>Event Ingress Per Minute</h3><div id="chartEventsTimeline" class="chart small"></div></div>
    </div>

    <div class="card" style="margin-top:10px">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Seq</th>
              <th>Level</th>
              <th>Type</th>
              <th>Request</th>
              <th>Message</th>
              <th>Data</th>
              <th></th>
            </tr>
          </thead>
          <tbody id="eventsBody"></tbody>
        </table>
      </div>
    </div>
  `;

  document.getElementById("eventsApplyBtn").addEventListener("click", async () => {
    state.filters.events.level = String(document.getElementById("eventsLevel").value || "").trim().toUpperCase();
    state.filters.events.eventType = String(document.getElementById("eventsType").value || "").trim();
    state.filters.events.requestId = String(document.getElementById("eventsRequestId").value || "").trim();
    state.filters.events.search = String(document.getElementById("eventsSearch").value || "").trim();
    state.filters.events.limit = Number(document.getElementById("eventsLimit").value || "300");

    applyWsFilters();
    await refreshPageData("events");
    renderEvents();
  });

  document.getElementById("eventsClearBtn").addEventListener("click", async () => {
    state.filters.events = { level: "", eventType: "", requestId: "", search: "", limit: 300 };
    document.getElementById("eventsLevel").value = "";
    document.getElementById("eventsType").value = "";
    document.getElementById("eventsRequestId").value = "";
    document.getElementById("eventsSearch").value = "";
    document.getElementById("eventsLimit").value = "300";

    applyWsFilters();
    await refreshPageData("events");
    renderEvents();
  });

  state.layouts.events = true;
}

function filteredEventsForView() {
  const f = state.filters.events;
  const search = String(f.search || "").toLowerCase();
  let items = state.events.slice();

  if (f.level) {
    items = items.filter((item) => String(item.level || "").toUpperCase() === f.level);
  }
  if (f.eventType) {
    items = items.filter((item) => String(item.event_type || "").includes(f.eventType));
  }
  if (f.requestId) {
    items = items.filter((item) => String(item.request_id || "") === f.requestId);
  }
  if (search) {
    items = items.filter((item) => {
      const text = `${item.message || ""} ${item.event_type || ""} ${item.request_id || ""} ${compactJson(item.data || {})}`.toLowerCase();
      return text.includes(search);
    });
  }

  return items.slice(0, Math.max(1, Number(f.limit || 300)));
}

function renderEvents() {
  ensureEventsLayout();

  const rows = filteredEventsForView();
  const body = document.getElementById("eventsBody");
  body.innerHTML = rows
    .map((event) => {
      const level = String(event.level || "INFO").toUpperCase();
      const cls = level === "ERROR" ? "err" : level === "WARNING" ? "warn" : "ok";
      return `
        <tr>
          <td>${esc(formatTime(event.ts))}</td>
          <td>${esc(event.seq)}</td>
          <td><span class="badge ${cls}">${esc(level)}</span></td>
          <td>${esc(event.event_type || "")}</td>
          <td><span class="code">${esc(event.request_id || "-")}</span></td>
          <td>${esc(event.message || "")}</td>
          <td><span class="code">${esc(compactJson(event.data || {}))}</span></td>
          <td><button class="btn btn-ghost" data-event-seq="${esc(event.seq)}">open</button></td>
        </tr>
      `;
    })
    .join("");

  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="8" class="muted">No events found for selected filters</td></tr>`;
  }

  body.querySelectorAll("button[data-event-seq]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const seq = Number(btn.getAttribute("data-event-seq") || 0);
      const item = state.eventsBySeq.get(seq);
      if (item) openModal(`Event #${seq}`, item);
    });
  });

  const byType = {};
  const perMinute = {};
  for (const row of rows) {
    const type = String(row.event_type || "unknown");
    byType[type] = (byType[type] || 0) + 1;
    const d = new Date(row.ts || "");
    if (!Number.isNaN(d.getTime())) {
      const key = `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
      perMinute[key] = (perMinute[key] || 0) + 1;
    }
  }

  const topTypes = Object.entries(byType)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12);

  setChartOption("chartEventsTypes", {
    tooltip: { trigger: "axis" },
    grid: { left: 70, right: 20, top: 18, bottom: 24 },
    xAxis: { type: "value" },
    yAxis: { type: "category", inverse: true, data: topTypes.map((item) => item[0]) },
    series: [
      {
        type: "bar",
        data: topTypes.map((item) => Number(item[1])),
        itemStyle: { color: "#ff9f45", borderRadius: [0, 6, 6, 0] },
      },
    ],
  });

  const timelineKeys = Object.keys(perMinute).sort();
  setChartOption("chartEventsTimeline", {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: timelineKeys },
    yAxis: { type: "value" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18 }],
    series: [
      {
        type: "line",
        smooth: true,
        areaStyle: { color: "rgba(255, 137, 38, 0.18)" },
        itemStyle: { color: "#f57838" },
        data: timelineKeys.map((k) => Number(perMinute[k] || 0)),
      },
    ],
  });
}

function ensureUsageLayout() {
  if (state.layouts.usage) return;
  const root = document.getElementById("page-usage");
  root.innerHTML = `
    <div class="card">
      <div class="filters">
        <label>Group By
          <select id="usageGroupBy">
            <option value="provider">provider</option>
            <option value="model">model</option>
            <option value="key">key</option>
            <option value="provider_model">provider_model</option>
            <option value="provider_model_key">provider_model_key</option>
            <option value="capability">capability</option>
          </select>
        </label>
        <label>Provider
          <input id="usageProvider" type="text" placeholder="optional" />
        </label>
        <label>Model
          <input id="usageModel" type="text" placeholder="optional" />
        </label>
        <label>Capability
          <input id="usageCapability" type="text" placeholder="chat.completions" />
        </label>
        <label>Event Limit
          <select id="usageLimit">
            <option value="100">100</option>
            <option value="200" selected>200</option>
            <option value="500">500</option>
            <option value="1000">1000</option>
          </select>
        </label>
      </div>
      <div class="row" style="margin-top:10px">
        <button id="usageApplyBtn" class="btn">Apply</button>
        <button id="usageResetBtn" class="btn btn-ghost">Reset</button>
      </div>
    </div>

    <div class="grid cols-4" style="margin-top:10px">
      <div class="card"><h3>Total Requests</h3><div id="usageKpiReq" class="stat">0</div><div class="sub">provider attempts</div></div>
      <div class="card"><h3>Success Rate</h3><div id="usageKpiRate" class="stat">0%</div><div class="sub">success / total</div></div>
      <div class="card"><h3>Average Latency</h3><div id="usageKpiLatency" class="stat">0ms</div><div class="sub">across matched rows</div></div>
      <div class="card"><h3>Total Tokens</h3><div id="usageKpiTokens" class="stat">0</div><div class="sub">prompt + completion</div></div>
    </div>

    <div class="grid cols-2" style="margin-top:10px">
      <div class="chart-card"><h3>Requests vs Errors</h3><div id="chartUsageVolume" class="chart small"></div></div>
      <div class="chart-card"><h3>Success Rate and Latency</h3><div id="chartUsageQuality" class="chart small"></div></div>
    </div>

    <div class="chart-card" style="margin-top:10px"><h3>Token Distribution</h3><div id="chartUsageTokens" class="chart small"></div></div>

    <div class="card" style="margin-top:10px">
      <h3 style="margin:0 0 8px">Aggregate Rows</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Group</th><th>Requests</th><th>Success</th><th>Errors</th><th>Rate</th><th>429</th><th>Avg Latency</th><th>P95</th><th>Tokens</th><th>Last Error</th>
            </tr>
          </thead>
          <tbody id="usageAggBody"></tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-top:10px">
      <h3 style="margin:0 0 8px">Recent Usage Events</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th><th>Provider</th><th>Model</th><th>Capability</th><th>Key</th><th>Status</th><th>Latency</th><th>Tokens</th><th>Error</th>
            </tr>
          </thead>
          <tbody id="usageEventsBody"></tbody>
        </table>
      </div>
    </div>
  `;

  document.getElementById("usageApplyBtn").addEventListener("click", async () => {
    const f = state.filters.usage;
    f.groupBy = String(document.getElementById("usageGroupBy").value || "provider");
    f.provider = String(document.getElementById("usageProvider").value || "").trim();
    f.model = String(document.getElementById("usageModel").value || "").trim();
    f.capability = String(document.getElementById("usageCapability").value || "").trim();
    f.limit = Number(document.getElementById("usageLimit").value || "200");
    await refreshUsageTabData();
    renderUsage();
  });

  document.getElementById("usageResetBtn").addEventListener("click", async () => {
    state.filters.usage = { groupBy: "provider", provider: "", model: "", capability: "", limit: 200 };
    document.getElementById("usageGroupBy").value = "provider";
    document.getElementById("usageProvider").value = "";
    document.getElementById("usageModel").value = "";
    document.getElementById("usageCapability").value = "";
    document.getElementById("usageLimit").value = "200";
    await refreshUsageTabData();
    renderUsage();
  });

  state.layouts.usage = true;
}

function renderUsage() {
  ensureUsageLayout();
  const agg = state.tabs.usageAggregate || { items: [] };
  const events = state.tabs.usageEvents || { items: [] };

  const rows = agg.items || [];
  const totalReq = rows.reduce((sum, row) => sum + Number(row.requests_total || 0), 0);
  const totalSuccess = rows.reduce((sum, row) => sum + Number(row.success_total || 0), 0);
  const totalErrors = rows.reduce((sum, row) => sum + Number(row.error_total || 0), 0);
  const totalTokens = rows.reduce((sum, row) => sum + Number(row.tokens_total || 0), 0);
  const avgLatency = rows.length ? rows.reduce((sum, row) => sum + Number(row.latency_avg_ms || 0), 0) / rows.length : 0;

  setText("usageKpiReq", fmtInt(totalReq));
  setText("usageKpiRate", totalReq ? fmtPct(totalSuccess / totalReq) : "0%");
  setText("usageKpiLatency", fmtMs(avgLatency));
  setText("usageKpiTokens", fmtInt(totalTokens));

  const topRows = topN(rows, 14);
  const cats = topRows.map((row) => row.group || "-");

  setChartOption("chartUsageVolume", {
    tooltip: { trigger: "axis" },
    legend: { bottom: 0 },
    grid: { left: 58, right: 16, top: 18, bottom: 42 },
    xAxis: { type: "category", data: cats },
    yAxis: { type: "value" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16 }],
    series: [
      {
        name: "requests",
        type: "bar",
        stack: "volume",
        data: topRows.map((row) => Number(row.requests_total || 0)),
        itemStyle: { color: "#ff9f45" },
      },
      {
        name: "errors",
        type: "bar",
        stack: "volume",
        data: topRows.map((row) => Number(row.error_total || 0)),
        itemStyle: { color: "#d9533f" },
      },
    ],
  });

  setChartOption("chartUsageQuality", {
    tooltip: { trigger: "axis" },
    legend: { bottom: 0 },
    grid: { left: 58, right: 58, top: 18, bottom: 42 },
    xAxis: { type: "category", data: cats },
    yAxis: [
      { type: "value", name: "rate %" },
      { type: "value", name: "latency ms" },
    ],
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16 }],
    series: [
      {
        name: "success%",
        type: "line",
        smooth: true,
        data: topRows.map((row) => Number(row.success_rate || 0) * 100),
        itemStyle: { color: "#2f9f5b" },
      },
      {
        name: "latency",
        type: "line",
        yAxisIndex: 1,
        smooth: true,
        data: topRows.map((row) => Number(row.latency_avg_ms || 0)),
        itemStyle: { color: "#ff7a18" },
      },
    ],
  });

  setChartOption("chartUsageTokens", {
    tooltip: { trigger: "axis" },
    grid: { left: 58, right: 16, top: 18, bottom: 28 },
    xAxis: { type: "category", data: cats },
    yAxis: { type: "value" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16 }],
    series: [
      {
        type: "bar",
        data: topRows.map((row) => Number(row.tokens_total || 0)),
        itemStyle: { color: "#f6a34d", borderRadius: [6, 6, 0, 0] },
      },
    ],
  });

  const aggBody = document.getElementById("usageAggBody");
  aggBody.innerHTML = rows
    .map((row) => `
      <tr>
        <td><span class="code">${esc(row.group || "-")}</span></td>
        <td>${fmtInt(row.requests_total)}</td>
        <td>${fmtInt(row.success_total)}</td>
        <td>${fmtInt(row.error_total)}</td>
        <td>${fmtPct(row.success_rate)}</td>
        <td>${fmtInt(row.status_429)}</td>
        <td>${fmtMs(row.latency_avg_ms)}</td>
        <td>${fmtMs(row.latency_p95_ms)}</td>
        <td>${fmtInt(row.tokens_total)}</td>
        <td>${esc(row.last_error || "-")}</td>
      </tr>
    `)
    .join("");
  if (!rows.length) {
    aggBody.innerHTML = `<tr><td colspan="10" class="muted">No aggregate rows</td></tr>`;
  }

  const usageBody = document.getElementById("usageEventsBody");
  usageBody.innerHTML = (events.items || [])
    .map((row) => {
      const ok = Boolean(row.ok);
      return `
        <tr>
          <td>${esc(formatTime(row.ts))}</td>
          <td>${esc(row.provider || "")}</td>
          <td><span class="code">${esc(row.model || "")}</span></td>
          <td>${esc(row.capability || "")}</td>
          <td><span class="code">${esc(row?.key?.id || "")}</span></td>
          <td><span class="badge ${ok ? "ok" : "err"}">${ok ? "ok" : `err ${esc(row.status_code)}`}</span></td>
          <td>${fmtMs(row.latency_ms)}</td>
          <td>${fmtInt(row?.tokens?.total_tokens)}</td>
          <td>${esc(row.error || "-")}</td>
        </tr>
      `;
    })
    .join("");
  if (!(events.items || []).length) {
    usageBody.innerHTML = `<tr><td colspan="9" class="muted">No usage events</td></tr>`;
  }
}

function ensureHttpLayout() {
  if (state.layouts.http) return;
  const root = document.getElementById("page-http");
  root.innerHTML = `
    <div class="card">
      <div class="row">
        <label>Group By
          <select id="httpGroupBy">
            <option value="path">path</option>
            <option value="method">method</option>
            <option value="status_code">status_code</option>
            <option value="status_class">status_class</option>
            <option value="path_method">path_method</option>
          </select>
        </label>
        <button id="httpApplyBtn" class="btn">Apply</button>
      </div>
    </div>

    <div class="grid cols-2" style="margin-top:10px">
      <div class="chart-card"><h3>HTTP Request Volume</h3><div id="chartHttpVolume" class="chart small"></div></div>
      <div class="chart-card"><h3>HTTP Latency (Avg/P95)</h3><div id="chartHttpLatency" class="chart small"></div></div>
    </div>

    <div class="card" style="margin-top:10px">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Group</th><th>Requests</th><th>Success</th><th>Error</th><th>Rate</th><th>2xx</th><th>4xx</th><th>5xx</th><th>Avg</th><th>P95</th>
            </tr>
          </thead>
          <tbody id="httpBody"></tbody>
        </table>
      </div>
    </div>
  `;

  document.getElementById("httpApplyBtn").addEventListener("click", async () => {
    state.filters.http.groupBy = String(document.getElementById("httpGroupBy").value || "path");
    applyWsFilters();
    await refreshHttpTabData();
    renderHttp();
  });

  state.layouts.http = true;
}

function renderHttp() {
  ensureHttpLayout();
  document.getElementById("httpGroupBy").value = state.filters.http.groupBy;
  const stats = state.tabs.httpStats || state.live.httpStats || { items: [] };
  const items = stats.items || [];

  const topRows = topN(items, 15);
  const cats = topRows.map((row) => row.group || "-");

  setChartOption("chartHttpVolume", {
    tooltip: { trigger: "axis" },
    grid: { left: 58, right: 18, top: 18, bottom: 30 },
    xAxis: { type: "category", data: cats },
    yAxis: { type: "value" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16 }],
    series: [
      {
        type: "bar",
        data: topRows.map((row) => Number(row.requests_total || 0)),
        itemStyle: { color: "#ff9340", borderRadius: [6, 6, 0, 0] },
      },
    ],
  });

  setChartOption("chartHttpLatency", {
    tooltip: { trigger: "axis" },
    legend: { bottom: 0 },
    grid: { left: 58, right: 18, top: 18, bottom: 40 },
    xAxis: { type: "category", data: cats },
    yAxis: { type: "value" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16 }],
    series: [
      {
        name: "avg",
        type: "line",
        smooth: true,
        data: topRows.map((row) => Number(row.latency_avg_ms || 0)),
        itemStyle: { color: "#ff7a18" },
      },
      {
        name: "p95",
        type: "line",
        smooth: true,
        data: topRows.map((row) => Number(row.latency_p95_ms || 0)),
        itemStyle: { color: "#ba2f24" },
      },
    ],
  });

  const body = document.getElementById("httpBody");
  body.innerHTML = items
    .map((row) => `
      <tr>
        <td><span class="code">${esc(row.group || "-")}</span></td>
        <td>${fmtInt(row.requests_total)}</td>
        <td>${fmtInt(row.success_total)}</td>
        <td>${fmtInt(row.error_total)}</td>
        <td>${fmtPct(row.success_rate)}</td>
        <td>${fmtInt(row.status_2xx)}</td>
        <td>${fmtInt(row.status_4xx)}</td>
        <td>${fmtInt(row.status_5xx)}</td>
        <td>${fmtMs(row.latency_avg_ms)}</td>
        <td>${fmtMs(row.latency_p95_ms)}</td>
      </tr>
    `)
    .join("");
  if (!items.length) {
    body.innerHTML = `<tr><td colspan="10" class="muted">No HTTP stats</td></tr>`;
  }
}

function ensureRouterLayout() {
  if (state.layouts.router) return;
  const root = document.getElementById("page-router");
  root.innerHTML = `
    <div class="grid cols-3">
      <div class="card"><h3>Candidates Tracked</h3><div id="routerKpiCandidates" class="stat">0</div><div class="sub">provider-model pairs</div></div>
      <div class="card"><h3>Total Calls</h3><div id="routerKpiCalls" class="stat">0</div><div class="sub">all attempts in memory</div></div>
      <div class="card"><h3>Total 429</h3><div id="routerKpi429" class="stat">0</div><div class="sub">rate-limit incidents</div></div>
    </div>

    <div class="grid cols-2" style="margin-top:10px">
      <div class="chart-card"><h3>Average Latency By Candidate</h3><div id="chartRouterLatency" class="chart small"></div></div>
      <div class="chart-card"><h3>Failures / Rate-Limited</h3><div id="chartRouterFailures" class="chart small"></div></div>
    </div>

    <div class="card" style="margin-top:10px">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Provider</th><th>Model</th><th>Total Calls</th><th>Failures</th><th>Rate Limited</th><th>Avg Latency</th>
            </tr>
          </thead>
          <tbody id="routerBody"></tbody>
        </table>
      </div>
    </div>
  `;
  state.layouts.router = true;
}

function renderRouter() {
  ensureRouterLayout();
  const items = state.live.routerScores?.items || [];
  setText("routerKpiCandidates", fmtInt(items.length));
  setText("routerKpiCalls", fmtInt(items.reduce((sum, row) => sum + Number(row.total_calls || 0), 0)));
  setText("routerKpi429", fmtInt(items.reduce((sum, row) => sum + Number(row.rate_limited || 0), 0)));

  const topRows = topN(items.slice().sort((a, b) => Number(b.total_calls || 0) - Number(a.total_calls || 0)), 14);
  const labels = topRows.map((row) => `${row.provider}/${row.model}`);

  setChartOption("chartRouterLatency", {
    tooltip: { trigger: "axis" },
    grid: { left: 58, right: 16, top: 18, bottom: 30 },
    xAxis: { type: "category", data: labels },
    yAxis: { type: "value" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16 }],
    series: [
      {
        type: "bar",
        data: topRows.map((row) => Number(row.avg_latency_ms || 0)),
        itemStyle: { color: "#ff9340", borderRadius: [6, 6, 0, 0] },
      },
    ],
  });

  setChartOption("chartRouterFailures", {
    tooltip: { trigger: "axis" },
    legend: { bottom: 0 },
    grid: { left: 58, right: 16, top: 18, bottom: 42 },
    xAxis: { type: "category", data: labels },
    yAxis: { type: "value" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16 }],
    series: [
      {
        name: "failures",
        type: "bar",
        data: topRows.map((row) => Number(row.failures || 0)),
        itemStyle: { color: "#d9533f" },
      },
      {
        name: "rate_limited",
        type: "bar",
        data: topRows.map((row) => Number(row.rate_limited || 0)),
        itemStyle: { color: "#c9851a" },
      },
    ],
  });

  const body = document.getElementById("routerBody");
  body.innerHTML = items
    .map((row) => `
      <tr>
        <td>${esc(row.provider || "")}</td>
        <td><span class="code">${esc(row.model || "")}</span></td>
        <td>${fmtInt(row.total_calls)}</td>
        <td>${fmtInt(row.failures)}</td>
        <td>${fmtInt(row.rate_limited)}</td>
        <td>${fmtMs(row.avg_latency_ms)}</td>
      </tr>
    `)
    .join("");
  if (!items.length) {
    body.innerHTML = `<tr><td colspan="6" class="muted">No router stats yet</td></tr>`;
  }
}

function ensureKeysLayout() {
  if (state.layouts.keys) return;
  const root = document.getElementById("page-keys");
  root.innerHTML = `
    <div class="card">
      <div class="row">
        <label>Provider
          <input id="keysProvider" type="text" placeholder="groq / gemini / pollinations" />
        </label>
        <button id="keysApplyBtn" class="btn">Apply</button>
      </div>
    </div>

    <div class="grid cols-2" style="margin-top:10px">
      <div class="chart-card"><h3>Top Keys By Requests</h3><div id="chartKeysVolume" class="chart small"></div></div>
      <div class="chart-card"><h3>Key Errors and 429</h3><div id="chartKeysErrors" class="chart small"></div></div>
    </div>

    <div class="card" style="margin-top:10px">
      <h3 style="margin:0 0 8px">Latest Key Limit Headers</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Captured At</th><th>Provider</th><th>Model</th><th>Key ID</th><th>Status</th><th>Headers</th><th>Details</th>
            </tr>
          </thead>
          <tbody id="keysLimitsBody"></tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-top:10px">
      <h3 style="margin:0 0 8px">Aggregated Key Usage</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Key</th><th>Requests</th><th>Success</th><th>Error</th><th>429</th><th>Rate</th><th>Avg Latency</th><th>Last Seen</th><th>Last Error</th>
            </tr>
          </thead>
          <tbody id="keysAggBody"></tbody>
        </table>
      </div>
    </div>
  `;

  document.getElementById("keysApplyBtn").addEventListener("click", async () => {
    state.filters.keys.provider = String(document.getElementById("keysProvider").value || "").trim();
    await refreshKeysTabData();
    renderKeys();
  });

  state.layouts.keys = true;
}

function renderKeys() {
  ensureKeysLayout();
  document.getElementById("keysProvider").value = state.filters.keys.provider;

  const limits = state.tabs.keyLimits?.items || [];
  const keyUsage = state.live.usageByKey?.items || [];

  const topKeys = topN(keyUsage, 14);
  const names = topKeys.map((row) => row.group || "-");
  setChartOption("chartKeysVolume", {
    tooltip: { trigger: "axis" },
    grid: { left: 58, right: 16, top: 18, bottom: 30 },
    xAxis: { type: "category", data: names },
    yAxis: { type: "value" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16 }],
    series: [
      {
        type: "bar",
        data: topKeys.map((row) => Number(row.requests_total || 0)),
        itemStyle: { color: "#ff9f45", borderRadius: [6, 6, 0, 0] },
      },
    ],
  });

  setChartOption("chartKeysErrors", {
    tooltip: { trigger: "axis" },
    legend: { bottom: 0 },
    grid: { left: 58, right: 16, top: 18, bottom: 42 },
    xAxis: { type: "category", data: names },
    yAxis: { type: "value" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 16 }],
    series: [
      {
        name: "errors",
        type: "bar",
        data: topKeys.map((row) => Number(row.error_total || 0)),
        itemStyle: { color: "#cf4f3a" },
      },
      {
        name: "429",
        type: "bar",
        data: topKeys.map((row) => Number(row.status_429 || 0)),
        itemStyle: { color: "#c6801b" },
      },
    ],
  });

  const limitsBody = document.getElementById("keysLimitsBody");
  limitsBody.innerHTML = limits
    .map((row, idx) => {
      const headers = row.headers || {};
      return `
        <tr>
          <td>${esc(formatTime(row.captured_at))}</td>
          <td>${esc(row.provider || "")}</td>
          <td><span class="code">${esc(row.model || "")}</span></td>
          <td><span class="code">${esc(row.key_id || row.key?.id || "")}</span></td>
          <td>${esc(row.status_code || "")}</td>
          <td><span class="code">${esc(compactJson(headers, 220))}</span></td>
          <td><button class="btn btn-ghost" data-limit-index="${idx}">open</button></td>
        </tr>
      `;
    })
    .join("");
  if (!limits.length) {
    limitsBody.innerHTML = `<tr><td colspan="7" class="muted">No key headers captured yet</td></tr>`;
  }
  limitsBody.querySelectorAll("button[data-limit-index]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.getAttribute("data-limit-index") || 0);
      const item = limits[idx];
      if (item) openModal("Key Limit Headers", item);
    });
  });

  const aggBody = document.getElementById("keysAggBody");
  aggBody.innerHTML = keyUsage
    .map((row) => `
      <tr>
        <td><span class="code">${esc(row.group || "")}</span></td>
        <td>${fmtInt(row.requests_total)}</td>
        <td>${fmtInt(row.success_total)}</td>
        <td>${fmtInt(row.error_total)}</td>
        <td>${fmtInt(row.status_429)}</td>
        <td>${fmtPct(row.success_rate)}</td>
        <td>${fmtMs(row.latency_avg_ms)}</td>
        <td>${esc(formatTime(row.last_seen))}</td>
        <td>${esc(row.last_error || "-")}</td>
      </tr>
    `)
    .join("");
  if (!keyUsage.length) {
    aggBody.innerHTML = `<tr><td colspan="9" class="muted">No key aggregate rows</td></tr>`;
  }
}

function renderCurrentPage() {
  if (state.currentPage === "dashboard") renderDashboard();
  if (state.currentPage === "events") renderEvents();
  if (state.currentPage === "usage") renderUsage();
  if (state.currentPage === "http") renderHttp();
  if (state.currentPage === "router") renderRouter();
  if (state.currentPage === "keys") renderKeys();
}

function showApp() {
  els.loginView.classList.add("hidden");
  els.app.classList.remove("hidden");
}

function showLogin() {
  els.app.classList.add("hidden");
  els.loginView.classList.remove("hidden");
}

function disconnectWs() {
  clearReconnect();
  if (state.ws) {
    try {
      state.ws.close();
    } catch {
      // no-op
    }
    state.ws = null;
  }
}

async function enterPanel() {
  await refreshCoreData();
  await refreshUsageTabData();
  await refreshHttpTabData();
  await refreshKeysTabData();

  showApp();
  await setPage(state.currentPage);
  connectWs();
  toast("Admin panel connected");
}

async function attemptLogin() {
  els.loginError.textContent = "";
  state.token = String(els.adminToken.value || "").trim();
  state.headerName = String(els.adminHeader.value || "x-admin-token").trim() || "x-admin-token";

  if (!state.token) {
    els.loginError.textContent = "Admin token is required";
    return;
  }

  try {
    storeAuth();
    await enterPanel();
  } catch (err) {
    showLogin();
    els.loginError.textContent = String(err.message || err);
    setWsStatus("err", "auth failed");
  }
}

async function manualRefresh() {
  try {
    await refreshCoreData();
    await refreshPageData(state.currentPage);
    renderCurrentPage();
    toast("Refreshed");
  } catch (err) {
    toast(String(err.message || err), true);
  }
}

function bindGlobalEvents() {
  els.loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await attemptLogin();
  });

  els.logoutBtn.addEventListener("click", () => {
    disconnectWs();
    clearAuth();
    showLogin();
    setWsStatus("warn", "disconnected");
    toast("Logged out");
  });

  els.refreshBtn.addEventListener("click", async () => {
    await manualRefresh();
  });

  els.pauseWsBtn.addEventListener("click", () => {
    state.wsPaused = !state.wsPaused;
    els.pauseWsBtn.textContent = state.wsPaused ? "Resume Live" : "Pause Live";
    if (!state.wsPaused) {
      renderCurrentPage();
      applyWsFilters();
    }
  });

  els.sinceSelect.addEventListener("change", async () => {
    state.sinceMinutes = Number(els.sinceSelect.value || "60");
    els.sinceLabel.textContent = `${state.sinceMinutes}m`;
    applyWsFilters();
    await manualRefresh();
  });

  els.modalCloseBtn.addEventListener("click", closeModal);
  els.modalOverlay.addEventListener("click", (event) => {
    if (event.target === els.modalOverlay) closeModal();
  });

  window.addEventListener("resize", debounce(resizeCharts, 200));
}

async function bootstrap() {
  buildMenu();
  bindGlobalEvents();
  loadAuth();

  els.sinceSelect.value = String(state.sinceMinutes);
  els.sinceLabel.textContent = `${state.sinceMinutes}m`;
  els.lastTick.textContent = "-";
  setWsStatus("warn", "disconnected");

  if (!state.token) {
    showLogin();
    return;
  }

  try {
    await enterPanel();
  } catch (err) {
    showLogin();
    els.loginError.textContent = String(err.message || err);
  }
}

bootstrap();
