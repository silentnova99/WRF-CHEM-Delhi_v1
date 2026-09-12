/* Module-5 3D WebGIS dashboard (MapLibre GL).
 * Consumes /dashboard/state → coupled PM2.5 hour grids (canvas overlay),
 * 3D extruded cells, met overlay lines, fire points, station towers/popups
 * and the Module-3 forecast summary.
 */
(() => {
  const PATH_STATE = "/dashboard/state";
  const CDN_OK = typeof maplibregl !== "undefined";

  const VIRIDIS = [
    [68, 1, 84], [71, 26, 111], [65, 50, 118], [55, 73, 120],
    [46, 91, 122], [38, 113, 121], [32, 117, 119], [28, 139, 108],
    [35, 154, 88], [63, 169, 72], [109, 181, 54], [157, 190, 44],
    [197, 194, 37], [224, 190, 30], [253, 231, 37], [250, 244, 63],
  ];

  const app = {
    state: null, map: null,
    hourIdx: 0, model: "map", field: "pm25_raw", met: "none",
  };

  const colorT = (t) => {
    t = Math.min(1, Math.max(0, t));
    const x = t * (VIRIDIS.length - 1);
    const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
    const f = x - i;
    const a = VIRIDIS[i], b = VIRIDIS[i + 1];
    return [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f),
    ];
  };

  const rgb = (c) => "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";

  const nodes = () => ({
    xs: app.state.coupled.lon_nodes,
    ys: app.state.coupled.lat_nodes,
  });

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
      for (const v of h[app.field]) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (lo === hi) hi = lo + 1;
    return { lo, hi };
  };

  const pmCanvas = () => {
    const h = current();
    const { ny, nx } = gridDims();
    const { lo, hi } = range();
    const cv = document.createElement("canvas");
    cv.width = nx; cv.height = ny;
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = h[app.field][j * nx + i];
        const c = colorT((v - lo) / (hi - lo));
        const idx = (j * nx + i) * 4;
        img.data[idx] = c[0]; img.data[idx + 1] = c[1];
        img.data[idx + 2] = c[2]; img.data[idx + 3] = 215;
      }
    }
    ctx.putImageData(img, 0, 0);
    return cv;
  };

  const extrusionFeatures = () => {
    const h = current();
    const { ny, nx } = gridDims();
    const { xs, ys } = nodes();
    const { hi } = range();
    const feats = [];
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const v = h[app.field][j * nx + i];
        if (!(v > 0)) continue;
        const height = Math.max(60, (v / hi) * 1600);
        feats.push({
          type: "Feature",
          properties: { color: rgb(colorT(v / hi)), height },
          geometry: { type: "Polygon", coordinates: [[
            [xs[i], ys[j]], [xs[i + 1], ys[j]],
            [xs[i + 1], ys[j + 1]], [xs[i], ys[j + 1]]]] },
        });
      }
    }
    return feats;
  };

  const metFeatures = () => {
    const last = app.state.coupled.met_last;
    const { ny, nx } = gridDims();
    const { xs, ys } = nodes();
    const feats = [];
    if (app.met === "wind") {
      for (let j = 0; j < ny; j += 2) {
        for (let i = 0; i < nx; i += 2) {
          const spd = last.ws10_ms[j * nx + i];
          const dir = ((last.wd_deg[j * nx + i] || 0) * Math.PI) / 180;
          const dx = Math.sin(dir) * 0.05, dy = Math.cos(dir) * 0.05;
          feats.push({
            type: "Feature",
            properties: { speed: spd },
            geometry: { type: "LineString", coordinates: [
              [xs[i], ys[j]], [xs[i] + dx, ys[j] + dy]] },
          });
        }
      }
    } else if (app.met === "pblh" || app.met === "t2m") {
      const col = app.met === "pblh" ? "pblh_m" : "t2m_c";
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          feats.push({
            type: "Feature", properties: {},
            geometry: { type: "Point", coordinates: [xs[i], ys[j]] },
          });
        }
      }
      return { feats, col };
    }
    return { feats, col: null };
  };

  const stationTowers = () => {
    const feats = (app.state.stations || [])
      .filter((s) => s.pm25_raw > 0)
      .map((s) => {
        const hmax = Math.max(1, ...(app.state.stations || []).map((x) => x.pm25_raw || 0));
        return {
          type: "Feature",
          properties: {
            station_id: s.station_id || "",
            pm: s.pm25_raw || 0,
            height: Math.max(80, ((s.pm25_raw || 0) / hmax) * 1800),
          },
          geometry: { type: "Point", coordinates: [s.lon, s.lat] },
        };
      })
      .map((f) => ({ ...f, properties: { ...f.properties, color: "#1e88e5" } }));
    return { type: "FeatureCollection", features: feats };
  };

  /* ------------------------------------------------------- layers ------ */
  function buildLayers(map) {
    map.addSource("pm25", { type: "canvas", canvas: document.createElement("canvas"), coordinates: [[0, 0], [1, 0], [1, 1], [0, 1]] });
    map.addLayer({ id: "pm25-raster", type: "raster", source: "pm25",
      paint: { "raster-opacity": 0.78, "raster-fade-duration": 0 } });

    map.addSource("pm25-ext", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "pm25-3d", type: "fill-extrusion", source: "pm25-ext",
      paint: { "fill-extrusion-color": ["get", "color"], "fill-extrusion-height": ["get", "height"], "fill-extrusion-opacity": 0.7 } });

    map.addSource("fires", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "fires", type: "circle", source: "fires",
      paint: { "circle-radius": ["interpolate", ["linear"], ["get", "frp_mw"], 0, 6, 2000, 26],
        "circle-color": "#ff3b30", "circle-opacity": 0.85,
        "circle-stroke-color": "#7a0f0a", "circle-stroke-width": 1 } });

    map.addSource("stations", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "stations", type: "circle", source: "stations",
      paint: { "circle-radius": 5.5, "circle-color": "#ffffff",
        "circle-stroke-color": "#0b2e13", "circle-stroke-width": 1.5 } });
    map.addLayer({ id: "station-label", type: "symbol", source: "stations",
      layout: { "text-field": ["get", "station_id"], "text-size": 10,
        "text-offset": [0, 1.1], "text-anchor": "top" },
      paint: { "text-color": "#111", "text-halo-color": "#fff", "text-halo-width": 1 } });

    map.addSource("met", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "met-lines", type: "line", source: "met",
      layout: { "line-cap": "round" },
      paint: { "line-color": "#111", "line-width": 1.4, "line-opacity": 0.85 } });
    map.addLayer({ id: "met-pts", type: "circle", source: "met",
      paint: { "circle-radius": 3, "circle-color": "#111", "circle-opacity": 0.8 },
      layout: { visibility: app.met === "t2m" || app.met === "pblh" ? "visible" : "none" } });

    map.addSource("towers", { type: "geojson", data: emptyFC() });
    map.addLayer({ id: "towers", type: "fill-extrusion", source: "towers",
      paint: { "fill-extrusion-color": ["get", "color"], "fill-extrusion-height": ["get", "height"], "fill-extrusion-opacity": 0.5 },
      layout: { visibility: "none" } });
  }

  const emptyFC = () => ({ type: "FeatureCollection", features: [] });

  /* ------------------------------------------------------- rendering --- */
  function renderAll() {
    const map = app.map;
    const src = map.getSource("pm25");
    src.updateImage(pmCanvas());
    src.setCoordinates(corners());

    map.getSource("pm25-ext").setData({ type: "FeatureCollection", features: extrusionFeatures() });

    map.getSource("fires").setData({
      type: "FeatureCollection",
      features: (app.state.fires || []).map((f) => ({
        type: "Feature",
        properties: { frp_mw: f.frp_mw || 0 },
        geometry: { type: "Point", coordinates: [f.lon, f.lat] },
      })),
    });

    const stationFC = {
      type: "FeatureCollection",
      features: (app.state.stations || []).map((s) => ({
        type: "Feature",
        properties: { station_id: s.station_id, pm: s.pm25_raw },
        geometry: { type: "Point", coordinates: [s.lon, s.lat] },
      })),
    };
    map.getSource("stations").setData(stationFC);
    map.getSource("towers").setData(stationTowers());

    vis(map, "pm25-raster", app.model === "map");
    vis(map, "pm25-3d", app.model === "ext");
    vis(map, "towers", app.model === "tower");

    const { feats, col } = metFeatures();
    map.getSource("met").setData({ type: "FeatureCollection", features: feats });
    const showMet = app.met !== "none";
    vis(map, "met-lines", showMet && col === null);
    vis(map, "met-pts", app.met === "pblh" || app.met === "t2m");

    drawLegend();
    updateStats();
  }

  const vis = (map, id, show) =>
    map.setLayoutProperty(id, "visibility", show ? "visible" : "none");

  function drawLegend() {
    const cv = document.getElementById("legend");
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const g = ctx.createLinearGradient(0, 0, cv.width, 0);
    for (let i = 0; i <= 10; i++) g.addColorStop(i / 10, rgb(colorT(i / 10)));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, cv.width, cv.height);
    const { lo, hi } = range();
    document.getElementById("lmin").textContent = lo.toFixed(0);
    document.getElementById("lmax").textContent = hi.toFixed(0);
    document.getElementById("tlabel").textContent =
      app.state.coupled.times[app.hourIdx] || "";
  }

  function updateStats() {
    const st = app.state;
    const h = st.coupled.hours[app.hourIdx];
    const vals = h[app.field];
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    document.getElementById("stats").innerHTML =
      `<div>run <b>${st.coupled.run_id}</b></div>` +
      `<div>hours <b>${st.coupled.times.length}</b> · cells <b>${vals.length}</b></div>` +
      `<div>field mean <b>${mean.toFixed(1)}</b> µg·m⁻³</div>` +
      `<div>fires <b>${(st.fires || []).length}</b> · stations <b>${(st.stations || []).length}</b></div>`;
    const fc = st.forecasts;
    document.getElementById("fc").textContent = fc && fc.run_id
      ? (fc.rmse_improvement_pct != null
        ? fc.run_id + " (Δ" + (fc.rmse_improvement_pct > 0 ? "+" : "") + fc.rmse_improvement_pct.toFixed(0) + "%)"
        : "available")
      : "n/a";
  }

  /* -------------------------------------------------------- bootstrap --- */
  async function main() {
    if (!CDN_OK) document.getElementById("offline").classList.remove("hidden");
    let st;
    try {
      st = await (await fetch(PATH_STATE)).json();
    } catch (e) {
      document.getElementById("runinfo").textContent = "state unavailable: " + e.message;
      return;
    }
    if (!st.coupled || !st.coupled.hours.length) {
      document.getElementById("runinfo").textContent =
        "no coupled run artifact yet — run aqf_delhi.wrf.coupling.coupled_run() first";
      return;
    }
    if (!CDN_OK) {
      document.getElementById("runinfo").textContent =
        "state loaded — MapLibre CDN unavailable, install maplibre-gl locally to view";
      return;
    }
    app.state = st;

    const { xs, ys } = nodes();
    app.map = new maplibregl.Map({
      container: "map",
      style: {
        version: 8,
        glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256, attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [{ id: "osm", type: "raster", source: "osm" }],
      },
      center: [xs[Math.floor(xs.length / 2)], ys[Math.floor(ys.length / 2)]],
      zoom: 9, pitch: 50, bearing: 15,
    });
    app.map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }));

    document.getElementById("runinfo").innerHTML =
      `<div>init <b>${st.coupled.run_id}</b></div>` +
      `<div>forecast valid hours <b>${st.coupled.times.length}</b></div>`;

    app.map.on("load", () => {
      buildLayers(app.map);
      document.getElementById("slider").max = st.coupled.hours.length - 1;
      renderAll();
    });

    document.getElementById("slider").addEventListener("input", (e) => {
      app.hourIdx = Number(e.target.value);
      renderAll();
    });
    document.getElementById("field").addEventListener("change", (e) => {
      app.field = e.target.value;
      renderAll();
    });
    document.getElementById("model").addEventListener("change", (e) => {
      app.model = e.target.value;
      renderAll();
    });
    document.getElementById("met").addEventListener("change", (e) => {
      app.met = e.target.value;
      renderAll();
    });

    app.map.on("click", "stations", (e) => {
      const p = e.features[0].properties;
      new maplibregl.Popup({ offset: 20 })
        .setLngLat(e.lngLat)
        .setHTML(`<b>${p.station_id}</b><br/>PM2.5 raw: <b>${p.pm.toFixed(1)}</b> µg·m⁻³`)
        .addTo(app.map);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();