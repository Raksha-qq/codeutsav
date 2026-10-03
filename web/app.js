// BilletVision Operator Dashboard Client Script
document.addEventListener("DOMContentLoaded", () => {
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProtocol}//${window.location.host}/events`;
  
  let ws = null;
  let lastSeenSeq = -1;
  const audioCtx = window.AudioContext ? new (window.AudioContext || window.webkitAudioContext)() : null;

  function playAlertBeep(frequency = 880, duration = 0.25) {
    if (!audioCtx) return;
    try {
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);
      gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("Audio alert suppressed:", e);
    }
  }

  // --- Live Stream Recovery ---
  const liveImg = document.getElementById("live-stream");
  if (liveImg) {
    liveImg.onerror = () => {
      setTimeout(() => {
        liveImg.src = `/video?t=${Date.now()}`;
      }, 1000);
    };
  }

  // --- WebSocket Connection ---
  function connectWs() {
    ws = new WebSocket(wsUrl);
    const indicator = document.getElementById("stream-status");

    ws.onopen = () => {
      if (indicator) indicator.textContent = "Live WS Connected";
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWsEvent(data);
      } catch (err) {
        console.error("Error parsing WS event", err);
      }
    };

    ws.onclose = () => {
      if (indicator) indicator.textContent = "Disconnected (polling fallback active)";
      setTimeout(connectWs, 2500);
    };
  }

  function handleWsEvent(msg) {
    if (msg.type === "kpi_update") {
      updateKpis(msg);
    } else if (msg.type === "inspection_result") {
      if (msg.billet_seq && msg.billet_seq === lastSeenSeq) return;
      lastSeenSeq = msg.billet_seq || lastSeenSeq;
      updateStatusTile(msg.status, msg.billet_id, msg.fail_reasons || msg.reasons);
      addLogRow(msg, true);
      if (msg.status === "FAIL") {
        playAlertBeep(750, 0.4);
      } else if (msg.status === "REVIEW") {
        playAlertBeep(440, 0.2);
      }
    }
  }

  function updateKpis(msg) {
    if (msg.fps !== undefined) document.getElementById("kpi-fps").textContent = msg.fps.toFixed(1);
    if (msg.latency !== undefined) document.getElementById("kpi-latency").textContent = `${msg.latency.toFixed(0)} ms`;
    if (msg.total !== undefined) document.getElementById("kpi-total").textContent = msg.total;
    if (msg.pass_rate !== undefined) document.getElementById("kpi-pass-rate").textContent = `${msg.pass_rate.toFixed(1)}%`;
    if (msg.ocr_rate !== undefined) document.getElementById("kpi-ocr-rate").textContent = `${msg.ocr_rate.toFixed(1)}%`;
  }

  function updateStatusTile(status, billetId, reasons) {
    const tile = document.getElementById("status-tile");
    const text = document.getElementById("status-text");
    const sub = document.getElementById("status-sub");
    const reasonsList = document.getElementById("reasons-list");

    const st = (status || "WAITING").toUpperCase();
    tile.className = `status-tile status-${st.toLowerCase()}`;
    text.textContent = st;
    sub.textContent = `Billet ID: ${billetId || "UNKNOWN"}`;

    reasonsList.innerHTML = "";
    if (reasons && reasons.length > 0) {
      reasons.forEach(r => {
        const li = document.createElement("li");
        li.textContent = r;
        reasonsList.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "All parameters within specified tolerance.";
      reasonsList.appendChild(li);
    }
  }

  function addLogRow(data, isNew = false) {
    const tbody = document.getElementById("logs-tbody");
    if (!tbody) return;

    // Check if row seq already exists
    const existing = Array.from(tbody.querySelectorAll("tr")).find(
      r => r.dataset.seq == data.billet_seq
    );
    if (existing) return;

    const tr = document.createElement("tr");
    tr.dataset.seq = data.billet_seq || "";
    if (isNew) tr.classList.add("new-row");

    const reasons = data.fail_reasons || data.reasons || [];
    const reasonsStr = Array.isArray(reasons) ? reasons.join("; ") : reasons;
    const st = (data.status || "PASS").toUpperCase();

    tr.innerHTML = `
      <td>${new Date().toLocaleTimeString()}</td>
      <td>${data.billet_seq || "-"}</td>
      <td><strong>${data.billet_id || "UNKNOWN"}</strong></td>
      <td>${data.length_mm || "-"} × ${data.width_mm || "-"} × ${data.height_mm || "-"}</td>
      <td>${data.ovality !== undefined ? data.ovality : "-"} / ${data.camber_mm !== undefined ? data.camber_mm : "-"}</td>
      <td>${data.ocr_confidence ? (data.ocr_confidence * 100).toFixed(0) + "%" : "-"}</td>
      <td><span class="badge ${st.toLowerCase()}">${st}</span></td>
      <td>${reasonsStr || "None"}</td>
      <td><button class="btn" style="padding:4px 8px;font-size:12px;">View</button></td>
    `;

    tbody.insertBefore(tr, tbody.firstChild);

    while (tbody.children.length > 100) {
      tbody.removeChild(tbody.lastChild);
    }
  }

  // --- Initial Logs & Polling Fallback ---
  async function fetchLogs() {
    try {
      const [logsRes, kpiRes] = await Promise.all([
        fetch("/api/logs"),
        fetch("/api/kpi")
      ]);
      const logs = await logsRes.json();
      const kpis = await kpiRes.json();

      if (kpis) updateKpis(kpis);

      if (logs && Array.isArray(logs) && logs.length > 0) {
        logs.forEach(rec => addLogRow(rec, false));
        const latest = logs[0];
        if (latest) {
          updateStatusTile(latest.status, latest.billet_id, latest.fail_reasons);
        }
      }
    } catch (e) {
      console.warn("Could not fetch logs:", e);
    }
  }

  // --- Profiles & Tolerances Integration ---
  async function loadTolerances() {
    try {
      const [profilesRes, tolsRes] = await Promise.all([
        fetch("/api/profiles"),
        fetch("/api/tolerances")
      ]);
      const profiles = await profilesRes.json();
      const tolData = await tolsRes.json();

      const select = document.getElementById("profile-select");
      if (select && profiles.length > 0) {
        select.innerHTML = "";
        profiles.forEach(p => {
          const opt = document.createElement("option");
          opt.value = p.id;
          opt.textContent = `${p.name} (${p.shape})`;
          if (p.id === tolData.active_profile) opt.selected = true;
          select.appendChild(opt);
        });
      }

      const tols = tolData.tolerances || {};
      if (document.getElementById("tol-width-nom")) {
        document.getElementById("tol-width-nom").value = tols.width_nominal_mm || 130.0;
        document.getElementById("tol-width-tol").value = tols.width_tol_mm || 1.0;
        document.getElementById("tol-height-nom").value = tols.height_nominal_mm || 130.0;
        document.getElementById("tol-height-tol").value = tols.height_tol_mm || 1.0;
      }
    } catch (err) {
      console.warn("Failed to load initial tolerances:", err);
    }
  }

  const saveBtn = document.getElementById("save-tolerances-btn");
  if (saveBtn) {
    saveBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      const profileId = document.getElementById("profile-select").value;
      const widthNom = parseFloat(document.getElementById("tol-width-nom").value);
      const widthTol = parseFloat(document.getElementById("tol-width-tol").value);
      const heightNom = parseFloat(document.getElementById("tol-height-nom").value);
      const heightTol = parseFloat(document.getElementById("tol-height-tol").value);

      try {
        const resp = await fetch("/api/tolerances", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            profile_id: profileId,
            tolerances: {
              width_nominal_mm: widthNom,
              width_tol_mm: widthTol,
              height_nominal_mm: heightNom,
              height_tol_mm: heightTol,
            }
          })
        });
        if (resp.ok) {
          saveBtn.textContent = "Saved ✓";
          setTimeout(() => { saveBtn.textContent = "Save Live"; }, 1500);
        }
      } catch (err) {
        alert("Failed to save tolerances: " + err);
      }
    });
  }

  // --- Export Buttons ---
  const csvBtn = document.getElementById("export-csv-btn");
  if (csvBtn) {
    csvBtn.addEventListener("click", () => {
      window.open("/api/export/csv", "_blank");
    });
  }

  const xlsxBtn = document.getElementById("export-xlsx-btn");
  if (xlsxBtn) {
    xlsxBtn.addEventListener("click", () => {
      window.open("/api/export/xlsx", "_blank");
    });
  }

  // Start initialization
  loadTolerances();
  fetchLogs();
  connectWs();

  // Polling fallback keeps dashboard completely up to date
  setInterval(fetchLogs, 3500);
});
