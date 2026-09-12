/* Unified homepage: map + live CPCB + forecast + health in one SPA. */
(() => {
  const CDN_OK = typeof maplibregl !== "undefined";

  const BANDS = {
    pm25_raw: [
      { name: "Good", max: 30, color: [0, 228, 0] },
      { name: "Satisfactory", max: 60, color: [255, 255, 0] },
      { name: "Moderate", max: 90, color: [255, 126, 0] },
      { name: "Poor", max: 120, color: [255, 0, 0] },
      { name: "Very poor", max: 250, color: [143, 63, 151] },
      { name: "Severe", max: Infinity, color: [126, 0, 35] },
    ],
    pm10_raw: [
      { name: "Good", max: 50, color: [0, 228, 0] },
      { name: "Satisfactory", max: 100, color: [255, 255, 0] },
      { name: "Moderate", max: 250, color: [255, 126, 0] },
      { name: "Poor", max: 350, color: [255, 0, 0] },
      { name: "Very poor", max: 430, color: [143, 63, 151] },
      { name: "Severe", max: Infinity, color: [126, 0, 35] },
    ],
    o3_raw: [
      { name: "Good", max: 50, color: [0, 228, 0] },
      { name: "Satisfactory", max: 100, color: [255, 255, 0] },
      { name: "Moderate", max: 168, color: [255, 126, 0] },
      { name: "Poor", max: 208, color: [255, 0, 0] },
      { name: "Very poor", max: 748, color: [143, 63, 151] },
      { name: "Severe", max: Infinity, color: [126, 0, 35] },
    ],
  };

  const app = {
    state: null, map: null,
    hourIdx: 0, model: "map", field: "pm25_raw", met: "none",
    blink: false, blinker: null,
  };

  const bandsFor = () => BANDS[app.field] || BANDS.pm25_raw;
  const bandOf = (v) => {
    const bands = bandsFor();
    for (let i = 0; i < bands.length; i++) {
      if (v <= bands[i].max) return { idx: i, band: bands[i] };
    }
    return { idx: bands.length - 1, band: bands[bands.length - 1] };
  };
  const bandToCss = (b, a) => `rgb(${b.color[0]},${b.color[1]},${b.color[2]}${a ? "," + a + ")" : ")"}`;

  const nodes = () => ({ xs: app.state.coupled.lon_nodes, ys: app.state.coupled.lat_nodes });
  const gridDims = () => ({ ny: app.state.coupled.ny, nx: app.state.coupled.nx });
  const corners = () => {
    const { xs, ys } = nodes();
    return [[xs[0], ys[0]], [xs[xs.length - 1], ys[0]],
            [xs[xs.length - 1], ys[ys.length - 1]], [xs[0], ys[ys.length - 1]]];
  };
  const current = () => app.state.coupled.hours[app.hourIdx];
  const range = () => {
    let lo = Infinity, hi = -Infinity;
    for (const h of app.state.coupled.hours) {
      for (const v of h[app.field]) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
    if (lo === hi) hi = lo + 1;
    return { lo, hi };
  };

  const pmCanvas = () => {
    const h = current(); const { ny, nx } = gridDims();
    const cv = document.createElement("canvas"); cv.width = nx; cv.height = ny;
    const ctx = cv.getContext("2d"); const img = ctx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = h[app.field][j * nx + i]; const b = bandOf(v).band;
      const idx = (j * nx + i) * 4;
      img.data[idx] = b.color[0]; img.data[idx+1] = b.color[1];
      img.data[idx+2] = b.color[2]; img.data[idx+3] = 215;
    }
    ctx.putImageData(img, 0, 0); return cv;
  };

  const extrusionFeatures = () => {
    const h = current(); const { ny, nx } = gridDims(); const { xs, ys } = nodes(); const feats = [];
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const v = h[app.field][j * nx + i]; if (!(v > 0)) continue;
      const b = bandOf(v); const height = 140 + b.idx * 230;
      feats.push({ type: "Feature", properties: { color: bandToCss(b.band), height },
        geometry: { type: "Polygon", coordinates: [[[xs[i],ys[j]],[xs[i+1],ys[j]],[xs[i+1],ys[j+1]],[xs[i],ys[j+1]]]] } });
    }
    return { type: "FeatureCollection", features: feats };
  };

  const metFeatures = () => {
    const last = app.state.coupled.met_last; const { ny, nx } = gridDims(); const { xs, ys } = nodes(); const feats = [];
    if (app.met === "wind") {
      for (let j = 0; j < ny; j += 2) for (let i = 0; i < nx; i += 2) {
        const dir = ((last.wd_deg[j * nx + i] || 0) * Math.PI) / 180;
        feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString",
          coordinates: [[xs[i],ys[j]], [xs[i]+Math.sin(dir)*0.05, ys[j]+Math.cos(dir)*0.05]] } });
      }
      return { feats, col: null };
    }
    if (app.met === "pblh" || app.met === "t2m") {
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++)
        feats.push({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [xs[i],ys[j]] } });
      return { feats, col: app.met === "pblh" ? "pblh_m" : "t2m_c" };
    }
    return { feats: [], col: null };
  };

  const stationTowers = () => {
    const st = app.state.stations || []; const maxV = Math.max(1, ...st.map(s => s.pm25_raw || 0));
    return { type: "FeatureCollection", features: st.map(s => {
      const v = s.pm25_raw || 0; const b = bandOf(v);
      return { type: "Feature", properties: { station_id: s.station_id||"", pm: v, color: bandToCss(b.band), height: Math.max(80, ((v||0)/maxV)*1800) },
        geometry: { type: "Point", coordinates: [s.lon, s.lat] } };
    })};
  };

  const spotValue = (s) => {
    const { xs, ys } = nodes(); const { nx } = gridDims();
    let j = 0, i = 0, db = Infinity, bb = Infinity;
    for (let k = 0; k < ys.length; k++) { const d = Math.abs(ys[k] - s.lat); if (d < db) { db = d; j = k; } }
    for (let k = 0; k < xs.length; k++) { const d = Math.abs(xs[k] - s.lon); if (d < bb) { bb = d; i = k; } }
    return current()[app.field][j * nx + i];
  };

  const spotsData = () => ({ type: "FeatureCollection", features: (app.state.stations||[]).map(s => {
    const v = spotValue(s); const b = bandOf(v);
    return { type: "Feature", properties: { pm: v, color: bandToCss(b.band) }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } };
  })});

  function startBlink() {
    if (app.blinker) return;
    app.blinker = setInterval(() => { if (!app.map) return; app.blink = !app.blink;
      app.map.setPaintProperty("spots", "circle-radius", app.blink ? 11 : 6);
      app.map.setPaintProperty("spots", "circle-opacity", app.blink ? 1.0 : 0.35);
    }, 650);
  }

  const emptyFC = () => ({ type: "FeatureCollection", features: [] });
  const vis = (map, id, show) => map.setLayoutProperty(id, "visibility", show ? "visible" : "none");

  function buildLayers(map) {
    map.addSource("pm25", { type: "canvas", canvas: document.createElement("canvas"), coordinates: [[0,0],[1,0],[1,1],[0,1]] });
    map.addLayer({ id: "pm25-raster", type: "raster", source: "pm25", paint: { "raster-opacity": 0.78, "raster-fade-duration": 0 } });
    map.addSource("pm25-ext", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "pm25-3d", type: "fill-extrusion", source: "pm25-ext",
      paint: { "fill-extrusion-color": ["get","color"], "fill-extrusion-height": ["get","height"], "fill-extrusion-opacity": 0.7 } });
    map.addSource("fires", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "fires", type: "circle", source: "fires",
      paint: { "circle-radius": ["interpolate",["linear"],["get","frp_mw"],0,6,2000,26], "circle-color": "#ff3b30", "circle-opacity": 0.85, "circle-stroke-color": "#7a0f0a", "circle-stroke-width": 1 } });
    map.addSource("spots", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "spots", type: "circle", source: "spots",
      paint: { "circle-radius": 7, "circle-color": ["get","color"], "circle-opacity": 0.95, "circle-stroke-color": "#fff", "circle-stroke-width": 1.5 } });
    map.addSource("stations", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "stations", type: "circle", source: "stations", paint: { "circle-radius": 5.5, "circle-color": "#fff", "circle-stroke-color": "#0b2e13", "circle-stroke-width": 1.5 } });
    map.addLayer({ id: "station-label", type: "symbol", source: "stations",
      layout: { "text-field": ["get","station_id"], "text-size": 10, "text-offset": [0,1.1], "text-anchor": "top" },
      paint: { "text-color": "#111", "text-halo-color": "#fff", "text-halo-width": 1 } });
    map.addSource("met", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "met-lines", type: "line", source: "met", layout: { "line-cap": "round" }, paint: { "line-color": "#111", "line-width": 1.4, "line-opacity": 0.85 } });
    map.addLayer({ id: "met-pts", type: "circle", source: "met", paint: { "circle-radius": 3, "circle-color": "#111", "circle-opacity": 0.8 }, layout: { visibility: "none" } });
    map.addSource("towers", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "towers", type: "fill-extrusion", source: "towers", paint: { "fill-extrusion-color": ["get","color"], "fill-extrusion-height": ["get","height"], "fill-extrusion-opacity": 0.55 }, layout: { visibility: "none" } });
  }

  function renderAll() {
    const map = app.map; const src = map.getSource("pm25");
    src.updateImage(pmCanvas()); src.setCoordinates(corners());
    map.getSource("pm25-ext").setData(extrusionFeatures());
    map.getSource("fires").setData({ type: "FeatureCollection", features: (app.state.fires||[]).map(f => ({ type: "Feature", properties: { frp_mw: f.frp_mw||0 }, geometry: { type: "Point", coordinates: [f.lon, f.lat] } })) });
    map.getSource("stations").setData({ type: "FeatureCollection", features: (app.state.stations||[]).map(s => ({ type: "Feature", properties: { station_id: s.station_id, pm: s.pm25_raw||0 }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } })) });
    map.getSource("spots").setData(spotsData()); map.getSource("towers").setData(stationTowers());
    vis(map, "pm25-raster", app.model === "map"); vis(map, "pm25-3d", app.model === "ext"); vis(map, "towers", app.model === "tower");
    const { feats, col } = metFeatures(); map.getSource("met").setData({ type: "FeatureCollection", features: feats });
    vis(map, "met-lines", app.met !== "none" && col === null); vis(map, "met-pts", app.met === "pblh" || app.met === "t2m");
    drawLegend(); updateStats();
  }

  function severityStyle(v) { const b = bandOf(v); return b.band.name + " " + v.toFixed(1) + " µg·m⁻³"; }

  function drawLegend() {
    const host = document.getElementById("legend-bands"); if (!host) return;
    const name = { pm25_raw: "PM2.5", pm10_raw: "PM10", o3_raw: "O3" }[app.field] || app.field;
    document.getElementById("legend-title").textContent = name + " · AQI severity";
    let prev = 0; host.innerHTML = bandsFor().map(b => {
      const t = prev + "–" + (b.max === Infinity ? "+" : b.max);
      const html = `<div class="band-row"><span class="chip" style="background:${bandToCss(b)}"></span><b>${b.name}</b><span>${t}</span></div>`;
      prev = b.max === Infinity ? b.max : b.max + 1; return html;
    }).join("");
    const { lo, hi } = range(); document.getElementById("lmin").textContent = lo.toFixed(0); document.getElementById("lmax").textContent = hi.toFixed(0);
    document.getElementById("tlabel").textContent = app.state.coupled.times[app.hourIdx] || "";
    document.getElementById("bandline").textContent = "worst: " + bandOf(Math.max(...current()[app.field])).band.name;
  }

  function updateStats() {
    const st = app.state; const h = st.coupled.hours[app.hourIdx]; const vals = h[app.field];
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    document.getElementById("stats").innerHTML =
      `<div class="card-header">Domain Summary</div>` +
      `<div>run <b>${st.coupled.run_id}</b></div>` +
      `<div>hours <b>${st.coupled.times.length}</b> · cells <b>${vals.length}</b></div>` +
      `<div>field mean <b>${mean.toFixed(1)}</b> µg·m⁻³</div>` +
      `<div>fires <b>${(st.fires||[]).length}</b> · stations <b>${(st.stations||[]).length}</b></div>`;
    const fc = st.forecasts;
    document.getElementById("fc").textContent = fc && fc.run_id
      ? (fc.rmse_improvement_pct != null ? fc.run_id + " (Δ" + (fc.rmse_improvement_pct > 0 ? "+" : "") + fc.rmse_improvement_pct.toFixed(0) + "%)" : "available")
      : "n/a";
  }

  /* ── live panels (fetch from API) ──────────────────────────────────── */

  async function loadCPCB() {
    const el = document.getElementById("cpcb-table").querySelector("tbody");
    const status = document.getElementById("cpcb-status");
    try {
      const r = await fetch("/observations/latest"); const d = await r.json();
      const rows = d.rows || [];
      status.textContent = `${rows.length} readings · ${d.partitions?.[0]?.partition || "—"}`;
      const stMap = {};
      rows.forEach(o => { if (!stMap[o.station_id]) stMap[o.station_id] = {}; stMap[o.station_id][o.metric] = o; });
      el.innerHTML = Object.entries(stMap).map(([sid, m]) => {
        const pm25 = m["PM2.5"] ? m["PM2.5"].value.toFixed(1) : "—";
        const pm10 = m["PM10"]   ? m["PM10"].value.toFixed(1)   : "—";
        const o3   = m["O3"]     ? m["O3"].value.toFixed(1)     : "—";
        const no2  = m["NO2"]    ? m["NO2"].value.toFixed(1)    : "—";
        const co   = m["CO"]     ? m["CO"].value.toFixed(2)     : "—";
        return `<tr><td>${sid}</td><td>${pm25}</td><td>${pm10}</td><td>${o3}</td><td>${no2}</td><td>${co}</td></tr>`;
      }).join("");
    } catch (e) { status.textContent = "unavailable: " + e.message; }
  }

  async function loadForecast() {
    const el = document.getElementById("fc-summary");
    try {
      const r = await fetch("/forecasts/latest"); const d = await r.json();
      el.innerHTML =
        `<div>run <b>${d.run_id}</b></div>` +
        `<div>stations <b>${d.stations ?? "—"}</b></div>` +
        (d.rmse_improvement_pct != null ? `<div>RMSE improvement <b>${d.rmse_improvement_pct > 0 ? "+" : ""}${d.rmse_improvement_pct.toFixed(1)}%</b></div>` : "") +
        (d.interval_coverage_90 != null ? `<div>90% coverage <b>${(d.interval_coverage_90*100).toFixed(1)}%</b></div>` : "");
    } catch (e) { el.textContent = "n/a"; }
  }

  async function loadHealth() {
    const el = document.getElementById("health-grid");
    try {
      const r = await fetch("/health"); const d = await r.json();
      const sources = d.sources || {};
      el.innerHTML =
        `<div class="health-row"><span class="dot green"></span>API <b>ok</b></div>` +
        Object.entries(sources).map(([k, v]) =>
          `<div class="health-row"><span class="dot ${v.completed_partitions > 0 ? 'green' : 'yellow'}"></span>${k} <b>${v.completed_partitions} part.</b> ${v.latest || ""}</div>`
        ).join("") +
        (d.forecasts ? `<div class="health-row"><span class="dot green"></span>forecast <b>${d.forecasts.run_id}</b></div>` : "");
    } catch (e) { el.textContent = "unavailable"; }
  }

  /* ── bootstrap ─────────────────────────────────────────────────────── */

  async function main() {
    if (!CDN_OK) document.getElementById("offline").classList.remove("hidden");
    let st;
    try { st = await (await fetch("/dashboard/state")).json(); }
    catch (e) { document.getElementById("runinfo").textContent = "state unavailable: " + e.message; return; }
    if (!st.coupled || !st.coupled.hours.length) {
      document.getElementById("runinfo").textContent = "no coupled run — run module-2 first"; return;
    }
    if (!CDN_OK) { document.getElementById("runinfo").textContent = "state loaded — CDN unavailable"; return; }
    app.state = st;

    const { xs, ys } = nodes();
    app.map = new maplibregl.Map({
      container: "map",
      style: {
        version: 8,
        glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
        sources: { osm: { type: "raster", tiles: ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png","https://b.tile.openstreetmap.org/{z}/{x}/{y}.png","https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, attribution: "© OpenStreetMap contributors" } },
        layers: [{ id: "osm", type: "raster", source: "osm" }],
      },
      center: [xs[Math.floor(xs.length/2)], ys[Math.floor(ys.length/2)]],
      zoom: 9, pitch: 50, bearing: 15,
    });
    app.map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }));

    document.getElementById("runinfo").innerHTML =
      `<div>init <b>${st.coupled.run_id}</b></div><div>valid hours <b>${st.coupled.times.length}</b></div>`;

    app.map.on("load", () => {
      buildLayers(app.map);
      document.getElementById("slider").max = st.coupled.hours.length - 1;
      renderAll(); startBlink();
    });

    document.getElementById("slider").addEventListener("input", e => { app.hourIdx = Number(e.target.value); renderAll(); });
    document.getElementById("field").addEventListener("change", e => { app.field = e.target.value; renderAll(); });
    document.getElementById("model").addEventListener("change", e => { app.model = e.target.value; renderAll(); });
    document.getElementById("met").addEventListener("change", e => { app.met = e.target.value; renderAll(); });

    app.map.on("click", "spots", e => {
      const p = e.features[0].properties;
      new maplibregl.Popup({ offset: 20 }).setLngLat(e.lngLat)
        .setHTML(`<b>${p.station_id}</b><br/>${severityStyle(p.pm)}<div style="color:#888">·</div>hour ${app.state.coupled.times[app.hourIdx]}`)
        .addTo(app.map);
    });
    app.map.on("click", "stations", e => {
      const p = e.features[0].properties;
      new maplibregl.Popup({ offset: 20 }).setLngLat(e.lngLat)
        .setHTML(`<b>${p.station_id}</b><br/>${severityStyle(p.pm)}<div style="color:#888">·</div>hour ${app.state.coupled.times[app.hourIdx]}`)
        .addTo(app.map);
    });

    // load live panels in parallel
    loadCPCB(); loadForecast(); loadHealth();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main); else main();
})();
