/**
 * BilletVision Operator Dashboard
 * Vanilla JS — no build step, no framework.
 *
 * API used:
 *   WS  /events                        — inspection_result | kpi_update | alert
 *   GET /api/health
 *   GET /api/stats
 *   GET /api/log?n=&status=
 *   GET /api/log/count
 *   GET /api/tolerances
 *   PUT /api/tolerances/{profile}       body: {updates:{}}
 *   PUT /api/tolerances/{profile}/activate
 *   GET /api/review/pending
 *   POST /api/review/{seq}              body: {action, notes}
 *   GET /api/alerts
 *   GET /api/alerts/active
 *   POST /api/alerts/clear
 */

(function () {
  "use strict";

  // Re-attach the MJPEG stream if it drops (pipeline restart / network blip)
  const _liveImg = document.getElementById("live-stream");
  if (_liveImg) {
    _liveImg.onerror = () => setTimeout(() => { _liveImg.src = `/video?t=${Date.now()}`; }, 1000);
  }

  /* ── state ──────────────────────────────────────────────────────────── */
  let _lastRecord = null;         // most-recent inspection result
  let _lastSeenSeq = -1;          // sequence deduplication guard
  let _logRows = [];              // current visible log rows (filtered)
  let _allRows = [];              // full fetched list (pre-filter)
  let _tolerances = {};           // {profile: {key:val}}
  let _activeProfile = null;

  /* ── audio (Web Audio API — no file needed) ─────────────────────────── */
  let _audioCtx = null;
  function _beep(frequency = 880, duration = 0.25, type = "square") {
    try {
      if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc  = _audioCtx.createOscillator();
      const gain = _audioCtx.createGain();
      osc.connect(gain);
      gain.connect(_audioCtx.destination);
      osc.type = type;
      osc.frequency.value = frequency;
      gain.gain.setValueAtTime(0.25, _audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, _audioCtx.currentTime + duration);
      osc.start(_audioCtx.currentTime);
      osc.stop(_audioCtx.currentTime + duration);
    } catch (_) { /* audio blocked — no crash */ }
  }
  function _alarmBeep()  { _beep(440, 0.4, "sawtooth"); }
  function _passBeep()   { _beep(1046, 0.12, "sine"); }

  /* ── WebSocket ──────────────────────────────────────────────────────── */
  const _wsUrl = (() => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/events`;
  })();

  let _ws = null;
  let _wsRetry = 1000;   // ms, doubles on failure up to 16 s

  function _connectWs() {
    _ws = new WebSocket(_wsUrl);

    _ws.onopen = () => {
      _wsRetry = 1000;
      _setWsStatus(true);
    };

    _ws.onmessage = (ev) => {
      try { _handleWsMsg(JSON.parse(ev.data)); }
      catch (_) { /* malformed payload — ignore */ }
    };

    _ws.onclose = () => {
      _setWsStatus(false);
      setTimeout(_connectWs, _wsRetry);
      _wsRetry = Math.min(_wsRetry * 2, 16000);
    };

    _ws.onerror = () => _ws.close();
  }

  function _setWsStatus(connected) {
    const dot   = document.getElementById("ws-dot");
    const label = document.getElementById("ws-label");
    if (!dot) return;
    dot.className   = "ws-dot " + (connected ? "connected" : "disconnected");
    label.textContent = connected ? "Live" : "Reconnecting…";
  }

  function _handleWsMsg(msg) {
    if (msg.type === "kpi_update")        _onKpi(msg);
    else if (msg.type === "inspection_result") _onInspection(msg);
    else if (msg.type === "alert")        _onAlertMsg(msg);
  }

  /* ── KPI update ─────────────────────────────────────────────────────── */
  function _onKpi(msg) {
    _setText("kpi-fps",       _fmt(msg.fps, 1));
    _setText("kpi-latency",   msg.latency_ms != null ? `${msg.latency_ms.toFixed(0)} ms` : "—");
    _setText("kpi-total",     msg.total ?? "—");
    _setText("kpi-pass-rate", msg.pass_rate != null ? `${(msg.pass_rate * 100).toFixed(1)}%` : "—");
    _setText("kpi-ocr-rate",  msg.ocr_rate  != null ? `${(msg.ocr_rate  * 100).toFixed(1)}%` : "—");
  }

  /* ── Inspection result ───────────────────────────────────────────────── */
  function _onInspection(msg) {
    if (msg.billet_seq && msg.billet_seq === _lastSeenSeq) return;
    if (msg.billet_seq) _lastSeenSeq = msg.billet_seq;

    _lastRecord = msg;
    _updateStatusTile(msg);
    _updateMeasDetails(msg);
    _prependLogRow(msg, true);

    if (msg.status === "FAIL") { _alarmBeep(); _showAlertBanner(msg); }
    else if (msg.status === "REWORK") { _alarmBeep(); _showAlertBanner(msg); }
    else if (msg.status === "PASS") { _passBeep(); _hideAlertBanner(); }

    // update review badge count
    if (msg.status === "REVIEW") {
      const badge = document.getElementById("review-badge");
      if (badge) {
        const cur = parseInt(badge.textContent) || 0;
        badge.textContent = cur + 1;
        badge.classList.remove("hidden");
      }
    }
  }

  function _updateStatusTile(msg) {
    const tile  = document.getElementById("status-tile");
    const text  = document.getElementById("status-text");
    const subid = document.getElementById("status-billet-id");
    if (!tile) return;
    tile.className = `status-tile ${(msg.status || "idle").toLowerCase()}`;
    text.textContent = msg.status || "?";
    subid.textContent = msg.billet_id || "—";
  }

  function _updateMeasDetails(msg) {
    const fmt = (v, d=1, u="mm") => v != null ? `${(+v).toFixed(d)} ${u}` : "—";
    _setText("m-length", fmt(msg.length_mm));
    _setText("m-width",  fmt(msg.width_mm));
    _setText("m-height", fmt(msg.height_mm));
    _setText("m-diam",   fmt(msg.diameter_mm));
    _setText("m-oval",   msg.ovality != null ? `${(+msg.ovality).toFixed(2)}%` : "—");
    _setText("m-diag",   fmt(msg.diag_diff_mm));
    _setText("m-ocr",    msg.ocr_confidence != null ? `${(msg.ocr_confidence * 100).toFixed(0)}%` : "—");
    _setText("m-ms",     msg.processing_ms  != null ? `${(+msg.processing_ms).toFixed(0)} ms` : "—");

    const reasons = msg.fail_reasons || [];
    const box  = document.getElementById("fail-reasons-box");
    const list = document.getElementById("fail-reasons-list");
    if (!box) return;
    if (reasons.length > 0) {
      list.innerHTML = reasons.map(r => `<li>${_esc(r)}</li>`).join("");
      box.classList.remove("hidden");
    } else {
      box.classList.add("hidden");
    }
  }

  /* ── Alert banner ───────────────────────────────────────────────────── */
  function _showAlertBanner(msg) {
    const banner = document.getElementById("alert-banner");
    if (!banner) return;
    const title  = document.getElementById("alert-title");
    const detail = document.getElementById("alert-detail");
    banner.className = `alert-banner ${(msg.status || "fail").toLowerCase()}`;
    title.textContent = `${msg.status} — Billet ${msg.billet_id || "?"}`;
    const reasons = msg.fail_reasons || msg.reasons || [];
    detail.textContent = reasons.slice(0, 2).join(" | ");
    banner.classList.remove("hidden");

    // push to alert history list
    _addAlertHistoryItem(msg);
  }

  function _hideAlertBanner() {
    const banner = document.getElementById("alert-banner");
    if (banner) banner.classList.add("hidden");
  }

  function _addAlertHistoryItem(msg) {
    const list = document.getElementById("alert-history-list");
    if (!list) return;
    // Remove "no alerts" placeholder
    const empty = list.querySelector(".alert-empty");
    if (empty) empty.remove();

    const reasons = (msg.fail_reasons || msg.reasons || []).slice(0, 2).join("; ");
    const time = msg.timestamp ? msg.timestamp.slice(11, 19) : new Date().toLocaleTimeString();
    const li = document.createElement("li");
    li.className = `alert-item ${(msg.status || "").toLowerCase()}`;
    li.innerHTML = `
      <span class="alert-item-time">${_esc(time)}</span>
      <span class="alert-item-id">${_esc(msg.billet_id || "?")}</span>
      <span class="alert-item-body">${_esc(reasons) || msg.status}</span>`;
    list.insertBefore(li, list.firstChild);
    // Keep history list tidy
    while (list.children.length > 20) list.removeChild(list.lastChild);
  }

  function _onAlertMsg(msg) {
    _showAlertBanner(msg);
    _alarmBeep();
  }

  /* ── Log table ──────────────────────────────────────────────────────── */
  const LOG_MAX = 100;

  function _prependLogRow(msg, isNew = false) {
    if (isNew) msg._isNew = true;
    _allRows.unshift(msg);
    if (_allRows.length > LOG_MAX) _allRows.pop();
    _renderLogTable();
    if (isNew) {
      setTimeout(() => { msg._isNew = false; }, 2000);
    }
  }

  function _renderLogTable() {
    const idFilter  = (document.getElementById("log-filter-id")?.value || "").trim().toUpperCase();
    const stFilter  = (document.getElementById("log-filter-status")?.value || "");
    _logRows = _allRows.filter(r => {
      if (stFilter && r.status !== stFilter) return false;
      if (idFilter && !(r.billet_id || "").toUpperCase().includes(idFilter)) return false;
      return true;
    });

    const tbody = document.getElementById("log-tbody");
    if (!tbody) return;

    if (_logRows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" class="table-empty">No records match the current filter.</td></tr>`;
      _setText("log-count-label", "0 records");
      return;
    }

    tbody.innerHTML = _logRows.map(r => `
      <tr class="${r._isNew ? 'new-row' : ''}">
        <td class="mono" style="white-space:nowrap">${_esc(_fmtTime(r.timestamp))}</td>
        <td class="mono">${r.billet_seq ?? "—"}</td>
        <td class="mono"><strong>${_esc(r.billet_id || "UNKNOWN")}</strong></td>
        <td class="mono">${_fmtDims(r)}</td>
        <td class="mono">${_fmtOval(r)}</td>
        <td class="mono">${_fmtDiag(r)}</td>
        <td class="mono">${r.ocr_confidence != null ? (r.ocr_confidence*100).toFixed(0)+"%" : "—"}</td>
        <td><span class="badge-status badge-${(r.status||"").toLowerCase()}">${_esc(r.status||"—")}</span></td>
        <td style="max-width:200px;white-space:normal;font-size:0.78rem">${_esc((r.fail_reasons||[]).join("; ")||"—")}</td>
        <td><button class="btn btn-sm" data-drill="${_esc(JSON.stringify(r))}">Detail</button></td>
      </tr>`).join("");

    // Attach drill-down handlers
    tbody.querySelectorAll("[data-drill]").forEach(btn => {
      btn.addEventListener("click", () => {
        try { _openModal(JSON.parse(btn.dataset.drill)); } catch (_) {}
      });
    });

    _setText("log-count-label", `${_logRows.length} record${_logRows.length !== 1 ? "s" : ""}`);
  }

  function _fmtTime(ts) {
    if (!ts) return "—";
    return ts.replace("T", " ").slice(0, 19);
  }
  function _fmtDims(r) {
    const v = (x) => x != null ? (+x).toFixed(1) : "—";
    return `${v(r.length_mm)} × ${v(r.width_mm)} × ${v(r.height_mm)}`;
  }
  function _fmtOval(r) {
    const d = r.diameter_mm != null ? `Ø${(+r.diameter_mm).toFixed(1)}` : "";
    const o = r.ovality != null ? ` ov=${(+r.ovality).toFixed(2)}%` : "";
    return d + o || "—";
  }
  function _fmtDiag(r) {
    return r.diag_diff_mm != null ? `${(+r.diag_diff_mm).toFixed(2)} mm` : "—";
  }

  async function _fetchLog() {
    const stFilter = (document.getElementById("log-filter-status")?.value || "");
    let url = `/api/log?n=${LOG_MAX}`;
    if (stFilter) url += `&status=${stFilter}`;
    try {
      const res = await fetch(url);
      if (!res.ok) return;
      const rows = await res.json();
      _allRows = rows;
      _renderLogTable();
      if (!_lastRecord && rows.length > 0) {
        _lastRecord = rows[0];
        _updateStatusTile(rows[0]);
        _updateMeasDetails(rows[0]);
      }
    } catch (_) {}
  }

  /* ── Review queue ───────────────────────────────────────────────────── */
  async function _fetchReviewQueue() {
    try {
      const res = await fetch("/api/review/pending?n=50");
      if (!res.ok) return;
      const items = await res.json();
      _renderReviewCards(items);

      const badge = document.getElementById("review-badge");
      if (badge) {
        badge.textContent = items.length;
        items.length > 0 ? badge.classList.remove("hidden") : badge.classList.add("hidden");
      }
    } catch (_) {}
  }

  function _renderReviewCards(items) {
    const container = document.getElementById("review-cards");
    if (!container) return;
    if (items.length === 0) {
      container.innerHTML = `<p class="table-empty">No pending reviews.</p>`;
      return;
    }
    container.innerHTML = items.map(r => _reviewCardHtml(r)).join("");
    container.querySelectorAll(".review-approve-btn").forEach(btn => {
      btn.addEventListener("click", () => _resolveReview(btn.dataset.seq, "approve", ""));
    });
    container.querySelectorAll(".review-reject-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const inp = container.querySelector(`input[data-seq="${btn.dataset.seq}"]`);
        _resolveReview(btn.dataset.seq, "reject", inp ? inp.value : "");
      });
    });
  }

  function _reviewCardHtml(r) {
    const dims = _fmtDims(r);
    const imgHtml = r.image_path
      ? `<img src="/static/snapshots/${_esc(r.image_path.split(/[\\/]/).pop())}" alt="crop">`
      : `<span>No snapshot</span>`;

    return `<div class="review-card">
      <div class="review-card-header">
        <span class="review-card-id">${_esc(r.billet_id || "UNKNOWN")}</span>
        <span class="review-card-seq">#${r.billet_seq}</span>
      </div>
      <div class="review-card-crop">${imgHtml}</div>
      <div class="review-card-meas">${_esc(dims)} · OCR ${r.ocr_confidence != null ? (r.ocr_confidence*100).toFixed(0)+"%" : "—"}</div>
      <div class="review-card-actions">
        <input class="review-id-input" type="text" placeholder="Override ID (optional)" data-seq="${r.billet_seq}" value="${_esc(r.billet_id||"")}">
        <button class="btn btn-pass btn-sm review-approve-btn" data-seq="${r.billet_seq}">Approve</button>
        <button class="btn btn-fail btn-sm review-reject-btn"  data-seq="${r.billet_seq}">Reject</button>
      </div>
    </div>`;
  }

  async function _resolveReview(seq, action, notes) {
    try {
      const res = await fetch(`/api/review/${seq}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, notes }),
      });
      if (res.ok) {
        await _fetchReviewQueue();
      }
    } catch (_) {}
  }

  /* ── Tolerances ─────────────────────────────────────────────────────── */
  async function _fetchTolerances() {
    try {
      const res = await fetch("/api/tolerances");
      if (!res.ok) return;
      _tolerances = await res.json();
      _populateProfileSelect();
    } catch (_) {}
  }

  function _populateProfileSelect() {
    const sel = document.getElementById("profile-select");
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = Object.keys(_tolerances).map(p =>
      `<option value="${_esc(p)}">${_esc(p)}</option>`).join("");
    if (prev && _tolerances[prev]) sel.value = prev;
    _loadProfileFields(sel.value);
  }

  function _loadProfileFields(profile) {
    const tol = _tolerances[profile];
    if (!tol) return;
    _setInputVal("tol-width-nom",   tol.width_nominal_mm);
    _setInputVal("tol-width-tol",   tol.width_tol_mm);
    _setInputVal("tol-height-nom",  tol.height_nominal_mm);
    _setInputVal("tol-height-tol",  tol.height_tol_mm);
    _setInputVal("tol-length-nom",  tol.length_nominal_mm);
    _setInputVal("tol-length-tol",  tol.length_tol_mm);
    _setInputVal("tol-diam-nom",    tol.diameter_nominal_mm);
    _setInputVal("tol-diam-tol",    tol.diameter_tol_mm);
    _setInputVal("tol-ovality",     tol.max_ovality_pct);
    _setInputVal("tol-diag-diff",   tol.max_diag_diff_mm);
    _setInputVal("tol-camber",      tol.max_camber_mm);
    _setInputVal("tol-cs-var",      tol.max_cross_section_var_mm);
    _setInputVal("tol-surface",     tol.max_surface_anomaly_score);
  }

  async function _saveTolerances() {
    const profile = document.getElementById("profile-select")?.value;
    if (!profile) return;
    const updates = {};
    const add = (id, key) => {
      const el = document.getElementById(id);
      if (el && el.value !== "") updates[key] = parseFloat(el.value);
    };
    add("tol-width-nom",  "width_nominal_mm");
    add("tol-width-tol",  "width_tol_mm");
    add("tol-height-nom", "height_nominal_mm");
    add("tol-height-tol", "height_tol_mm");
    add("tol-length-nom", "length_nominal_mm");
    add("tol-length-tol", "length_tol_mm");
    add("tol-diam-nom",   "diameter_nominal_mm");
    add("tol-diam-tol",   "diameter_tol_mm");
    add("tol-ovality",    "max_ovality_pct");
    add("tol-diag-diff",  "max_diag_diff_mm");
    add("tol-camber",     "max_camber_mm");
    add("tol-cs-var",     "max_cross_section_var_mm");
    add("tol-surface",    "max_surface_anomaly_score");

    _showTolStatus("Saving…", "");
    try {
      const res = await fetch(`/api/tolerances/${profile}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates }),
      });
      if (res.ok) {
        const updated = await res.json();
        _tolerances[profile] = updated;
        _showTolStatus("Saved successfully.", "ok");
      } else {
        const err = await res.json().catch(() => ({}));
        _showTolStatus(`Error: ${err.detail || res.status}`, "err");
      }
    } catch (e) {
      _showTolStatus(`Network error: ${e.message}`, "err");
    }
  }

  async function _activateProfile() {
    const profile = document.getElementById("profile-select")?.value;
    if (!profile) return;
    try {
      const res = await fetch(`/api/tolerances/${profile}/activate`, { method: "PUT" });
      if (res.ok) {
        _activeProfile = profile;
        const badge = document.getElementById("active-profile-badge");
        const name  = document.getElementById("active-profile-name");
        if (badge) { badge.classList.remove("hidden"); }
        if (name)  { name.textContent = profile; }
        _showTolStatus(`Profile "${profile}" activated.`, "ok");
      }
    } catch (_) {}
  }

  function _showTolStatus(msg, cls) {
    const el = document.getElementById("tol-save-status");
    if (!el) return;
    el.textContent = msg;
    el.className = `tol-status ${cls}`;
    el.classList.remove("hidden");
    setTimeout(() => el.classList.add("hidden"), 4000);
  }

  /* ── Alert REST ──────────────────────────────────────────────────────── */
  async function _fetchAlerts() {
    try {
      const res = await fetch("/api/alerts?n=20");
      if (!res.ok) return;
      const items = await res.json();
      const list  = document.getElementById("alert-history-list");
      if (!list || items.length === 0) return;
      list.innerHTML = items.map(a => {
        const reasons = (a.reasons || []).slice(0, 2).join("; ");
        return `<li class="alert-item ${(a.status||"").toLowerCase()}">
          <span class="alert-item-time">${_esc(_fmtTime(a.timestamp))}</span>
          <span class="alert-item-id">${_esc(a.billet_id||"?")}</span>
          <span class="alert-item-body">${_esc(reasons) || a.status}</span>
        </li>`;
      }).join("");
    } catch (_) {}
  }

  async function _fetchActiveAlert() {
    try {
      const res = await fetch("/api/alerts/active");
      if (!res.ok) return;
      const a = await res.json();
      if (a) _showAlertBanner(a);
      else   _hideAlertBanner();
    } catch (_) {}
  }

  async function _clearActiveAlert() {
    try {
      await fetch("/api/alerts/clear", { method: "POST" });
      _hideAlertBanner();
    } catch (_) {}
  }

  /* ── Drill-down modal ─────────────────────────────────────────────────── */
  function _openModal(record) {
    const backdrop = document.getElementById("modal-backdrop");
    const title    = document.getElementById("modal-title");
    const body     = document.getElementById("modal-body");
    if (!backdrop) return;

    title.textContent = `Billet ${record.billet_id || "UNKNOWN"} — ${record.status || "—"}`;

    const m = (label, val) => val != null
      ? `<div class="modal-meas-item"><div class="label">${_esc(label)}</div><div class="value">${_esc(String(val))}</div></div>`
      : "";

    const reasons = (record.fail_reasons || []);
    const reasonsHtml = reasons.length
      ? `<ul class="modal-reasons">${reasons.map(r => `<li>${_esc(r)}</li>`).join("")}</ul>`
      : `<p style="color:var(--text-muted);font-size:.85rem">All parameters within tolerance.</p>`;

    const imgHtml = record.image_path
      ? `<img src="/static/snapshots/${_esc(record.image_path.split(/[\\/]/).pop())}" alt="snapshot">`
      : `<p class="no-img">No snapshot saved.</p>`;

    body.innerHTML = `
      <div class="modal-section">
        <h3>Identity</h3>
        <div class="modal-meas-grid">
          ${m("Billet ID", record.billet_id)}
          ${m("Seq #",     record.billet_seq)}
          ${m("Batch",     record.batch_id)}
          ${m("Timestamp", _fmtTime(record.timestamp))}
          ${m("Status",    record.status)}
          ${m("Proc. time", record.processing_ms != null ? record.processing_ms.toFixed(0)+" ms" : null)}
        </div>
      </div>

      <div class="modal-section">
        <h3>Dimensions</h3>
        <div class="modal-meas-grid">
          ${m("Length (mm)",   record.length_mm  != null ? (+record.length_mm).toFixed(2)  : null)}
          ${m("Width (mm)",    record.width_mm   != null ? (+record.width_mm).toFixed(2)   : null)}
          ${m("Height (mm)",   record.height_mm  != null ? (+record.height_mm).toFixed(2)  : null)}
          ${m("Diameter (mm)", record.diameter_mm!= null ? (+record.diameter_mm).toFixed(2): null)}
          ${m("Ovality (%)",   record.ovality    != null ? (+record.ovality).toFixed(3)    : null)}
          ${m("Diag Diff (mm)",record.diag_diff_mm!=null ? (+record.diag_diff_mm).toFixed(3): null)}
        </div>
      </div>

      <div class="modal-section">
        <h3>OCR / ID</h3>
        <div class="modal-meas-grid">
          ${m("OCR Confidence", record.ocr_confidence != null ? (record.ocr_confidence*100).toFixed(1)+"%" : null)}
          ${m("Defects",        record.defects || "none")}
        </div>
      </div>

      <div class="modal-section">
        <h3>Decision Reasoning</h3>
        ${reasonsHtml}
      </div>

      <div class="modal-section modal-snapshot">
        <h3>Snapshot</h3>
        ${imgHtml}
      </div>`;

    backdrop.classList.remove("hidden");
  }

  function _closeModal() {
    const backdrop = document.getElementById("modal-backdrop");
    if (backdrop) backdrop.classList.add("hidden");
  }

  /* ── Tab switching ───────────────────────────────────────────────────── */
  function _initTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        const panel = document.getElementById(`tab-${btn.dataset.tab}`);
        if (panel) panel.classList.remove("hidden");

        // Lazy-load tab data on first switch
        if (btn.dataset.tab === "log")        _fetchLog();
        if (btn.dataset.tab === "review")     _fetchReviewQueue();
        if (btn.dataset.tab === "tolerances") _fetchTolerances();
      });
    });
  }

  /* ── Periodic refresh ────────────────────────────────────────────────── */
  function _startPolling() {
    // Pull stats every 2 s even without WS (fallback)
    setInterval(async () => {
      try {
        const r = await fetch("/api/stats");
        if (r.ok) _onKpi(await r.json());
      } catch (_) {}
    }, 2000);

    // Refresh log every 10 s if on log tab
    setInterval(() => {
      const logPanel = document.getElementById("tab-log");
      if (logPanel && !logPanel.classList.contains("hidden")) _fetchLog();
    }, 10000);

    // Refresh review queue every 15 s if on review tab
    setInterval(() => {
      const rvPanel = document.getElementById("tab-review");
      if (rvPanel && !rvPanel.classList.contains("hidden")) _fetchReviewQueue();
    }, 15000);
  }

  /* ── Utility ─────────────────────────────────────────────────────────── */
  function _setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }
  function _setInputVal(id, val) {
    const el = document.getElementById(id);
    if (el && val != null) el.value = val;
  }
  function _fmt(val, decimals = 1) {
    return val != null ? (+val).toFixed(decimals) : "—";
  }
  function _esc(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /* ── Init ────────────────────────────────────────────────────────────── */
  function init() {
    _initTabs();

    // Tolerances tab buttons
    document.getElementById("save-tol-btn")?.addEventListener("click", _saveTolerances);
    document.getElementById("activate-profile-btn")?.addEventListener("click", _activateProfile);
    document.getElementById("profile-select")?.addEventListener("change", e => {
      _loadProfileFields(e.target.value);
    });

    // Alert banner dismiss
    document.getElementById("alert-dismiss")?.addEventListener("click", _clearActiveAlert);
    document.getElementById("clear-alerts-btn")?.addEventListener("click", _clearActiveAlert);

    // Log filters
    document.getElementById("log-filter-id")?.addEventListener("input", _renderLogTable);
    document.getElementById("log-filter-status")?.addEventListener("change", () => {
      _allRows = []; // clear so fresh fetch respects status filter
      _fetchLog();
    });
    document.getElementById("log-refresh-btn")?.addEventListener("click", _fetchLog);
    document.getElementById("review-refresh-btn")?.addEventListener("click", _fetchReviewQueue);

    // Drill-down modal on "View Full" button
    document.getElementById("drill-last-btn")?.addEventListener("click", () => {
      if (_lastRecord) _openModal(_lastRecord);
    });

    // Modal close
    document.getElementById("modal-close")?.addEventListener("click", _closeModal);
    document.getElementById("modal-backdrop")?.addEventListener("click", e => {
      if (e.target === e.currentTarget) _closeModal();
    });
    document.addEventListener("keydown", e => {
      if (e.key === "Escape") _closeModal();
    });

    // Start
    _connectWs();
    _startPolling();
    _fetchAlerts();
    _fetchActiveAlert();
    _fetchTolerances();   // preload even on Live tab so save works

    // Initial log fetch silently (populates if already on log tab)
    _fetchLog();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
