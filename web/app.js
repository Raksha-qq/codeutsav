// BilletVision Operator Dashboard Client Script
document.addEventListener("DOMContentLoaded", () => {
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProtocol}//${window.location.host}/events`;
  
  let ws = null;

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
      if (indicator) indicator.textContent = "Disconnected (reconnecting...)";
      setTimeout(connectWs, 2000);
    };
  }

  function handleWsEvent(msg) {
    if (msg.type === "kpi_update") {
      if (msg.fps) document.getElementById("kpi-fps").textContent = msg.fps.toFixed(1);
      if (msg.latency) document.getElementById("kpi-latency").textContent = `${msg.latency.toFixed(0)} ms`;
      if (msg.total) document.getElementById("kpi-total").textContent = msg.total;
      if (msg.pass_rate) document.getElementById("kpi-pass-rate").textContent = `${msg.pass_rate.toFixed(1)}%`;
      if (msg.ocr_rate) document.getElementById("kpi-ocr-rate").textContent = `${msg.ocr_rate.toFixed(1)}%`;
    } else if (msg.type === "inspection_result") {
      updateStatusTile(msg.status, msg.billet_id, msg.reasons);
      addLogRow(msg);
    }
  }

  function updateStatusTile(status, billetId, reasons) {
    const tile = document.getElementById("status-tile");
    const text = document.getElementById("status-text");
    const sub = document.getElementById("status-sub");
    const reasonsList = document.getElementById("reasons-list");

    tile.className = `status-tile status-${status.toLowerCase()}`;
    text.textContent = status;
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

  function addLogRow(data) {
    const tbody = document.getElementById("logs-tbody");
    if (!tbody) return;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${data.timestamp || new Date().toLocaleTimeString()}</td>
      <td>${data.billet_seq || "-"}</td>
      <td><strong>${data.billet_id || "UNKNOWN"}</strong></td>
      <td>${data.length_mm || "-"} × ${data.width_mm || "-"} × ${data.height_mm || "-"}</td>
      <td>${data.ovality || "-"} / ${data.camber_mm || "-"}</td>
      <td>${data.ocr_confidence ? (data.ocr_confidence * 100).toFixed(0) + "%" : "-"}</td>
      <td><span class="badge ${data.status ? data.status.toLowerCase() : ''}">${data.status || "-"}</span></td>
      <td>${(data.fail_reasons || []).join(", ") || "None"}</td>
      <td><button class="btn" onclick="alert('Drill-down: ' + JSON.stringify(${JSON.stringify(data)}))">View</button></td>
    `;
    tbody.insertBefore(tr, tbody.firstChild);
  }

  connectWs();
});
