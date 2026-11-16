"""One self-contained HTML page for a View (whole scan plus a linked close-up per
region), from the same Scenes as the widgets. It embeds the NiiVue version
ipyniivue bundles, fetched from npm once and cached, unless niivue_js= is given.
"""

from __future__ import annotations

import base64
import gzip
import html
import json
import pathlib
import re
import urllib.request

import ipyniivue
import numpy as np
import pooch

from .view import PALETTE, TILES, Scene, View, _rgb, gap_params


def write_html(
    view: View,
    path: str | pathlib.Path,
    title: str | None = None,
    regions: int | None = None,
    scan_step: int | None = None,
    opacity: float = 0.5,
    crosshair_gap: float = 1.0,
    min_mm: float = 60.0,
    niivue_js: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Writes the page and returns its path. regions= caps how many regions are
    embedded (the first ones); each adds its cropped volume to the file size.
    scan_step= downsamples the whole-scan view (None = the View's
    scan_max_voxels rule, see load()); close-ups are always full resolution.
    niivue_js= overrides the NiiVue bundle: a file (embedded) or a URL (loaded
    by the page, so it then needs the network)."""
    path = pathlib.Path(path).with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    title = title or str(view.meta.get("id") or view.meta.get("source") or "roivue")
    n = len(view.regions) if regions is None else min(regions, len(view.regions))

    scenes = [_scene_json(view.scan_scene(scan_step))]
    scenes += [_scene_json(view.region_scene(k, min_mm)) for k in range(n)]
    data = {
        "title": title,
        "meta": view.meta,
        "summary": {
            "shape": list(view.image.shape),
            "spacing_mm": [round(float(s), 3) for s in np.linalg.norm(view.affine[:3, :3], axis=0)],
            "masks": list(view.masks),
            "regions": len(view.regions),
        },
        "rows": view.regions[:n],
        "scenes": {s["id"]: s for s in scenes},
        "tiles": [list(t) for t in TILES],
        "css": PALETTE,
        "radiological": view.radiological,
        "opacity": opacity,
        "gap": crosshair_gap,
    }
    page = (
        _TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("<script>__NIIVUE__</script>", _niivue_tag(niivue_js))
        .replace("__DATA__", json.dumps(data, default=_plain).replace("</", "<\\/"))
    )
    path.write_text(page, encoding="utf-8")
    return path


def niivue_version() -> str:
    """The NiiVue version inside the installed ipyniivue, so exports match Jupyter."""
    widget = pathlib.Path(ipyniivue.__file__).parent / "static" / "widget.js"
    found = re.search(r'name:"@niivue/niivue",version:"([^"]+)"', widget.read_text("utf-8"))
    if not found:
        raise RuntimeError(f"no NiiVue version in {widget}; pass niivue_js= to write_html")
    return found[1]


def niivue_bundle(version: str | None = None) -> pathlib.Path:
    """niivue.umd.js for `version` (default: ipyniivue's), from npm, checked, cached."""
    version = version or niivue_version()
    cache = pooch.os_cache("roivue") / f"niivue-{version}"
    js = cache / f"niivue-{version}.tgz.untar" / "package" / "dist" / "niivue.umd.js"
    if not js.exists():
        with urllib.request.urlopen(f"https://registry.npmjs.org/@niivue/niivue/{version}") as r:
            dist = json.load(r)["dist"]
        sha512 = base64.b64decode(dist["integrity"].removeprefix("sha512-")).hex()
        pooch.retrieve(
            dist["tarball"],
            f"sha512:{sha512}",
            fname=f"niivue-{version}.tgz",
            path=cache,
            processor=pooch.Untar(["package/dist/niivue.umd.js"]),
            progressbar=False,
        )
    return js


def _niivue_tag(niivue_js: str | pathlib.Path | None) -> str:
    """The <script> element that provides NiiVue."""
    if isinstance(niivue_js, str) and niivue_js.startswith(("http://", "https://")):
        return f'<script src="{html.escape(niivue_js)}"></script>'
    js = pathlib.Path(niivue_js or niivue_bundle()).read_text(encoding="utf-8")
    return "<script>" + js.replace("</script", "<\\/script") + "</script>"


def _scene_json(s: Scene) -> dict:
    radius, reach = gap_params(s.affine, s.shape)
    return {
        "id": "scan" if s.kind == "scan" else f"r{s.index}",
        "title": s.title,
        "image": _b64(s.image),
        "layers": [
            {
                "name": L.name,
                "cmap": L.cmap,
                "on": L.on,
                "rgb": _rgb(L.cmap),
                "overlay": _b64(L.overlay),
                "mesh": base64.b64encode(L.mesh).decode("ascii") if L.mesh else None,
            }
            for L in s.layers
        ],
        "info": s.info,
        "center_mm": None
        if s.center is None
        else [float(x) for x in (s.affine @ [*s.center, 1])[:3]],
        "gap_radius": radius,
        "gap_reach": reach,
    }


def _b64(nifti_gz: bytes) -> str:
    """Scenes hold fast level-1 gzip; recompress harder for a file on disk."""
    return base64.b64encode(gzip.compress(gzip.decompress(nifti_gz), compresslevel=6)).decode(
        "ascii"
    )


def _plain(o):
    """json.dumps default: numpy scalars/arrays as numbers/lists, else str."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
html,body{margin:0;height:100%;background:#000;color:#ddd;font:13px sans-serif}
#main{display:flex;flex-direction:column;height:100%}
#scanwrap{flex:1;min-height:0;display:none}
#scanwrap canvas{width:100%;height:100%;display:block}
#gridwrap{flex:1;min-height:0;display:none;grid-template-columns:repeat(4,1fr);grid-template-rows:20px 1fr 1fr;gap:3px}
#gridwrap .head{text-align:center;color:#aaa;line-height:20px}
#gridwrap .tile{position:relative;min-height:0}
#gridwrap canvas{position:absolute;inset:0;width:100%;height:100%}
#menu{flex:none;display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:6px 8px;border-top:1px solid #444}
#menu label{white-space:nowrap}
#infopanel{flex:none;display:none;max-height:30vh;overflow:auto;padding:6px 8px;border-top:1px solid #333;font-size:12px}
#infopanel td:first-child{color:#888;padding-right:12px;vertical-align:top;white-space:nowrap}
#nav{flex:none;display:flex;gap:16px;align-items:stretch;padding:6px 8px;border-top:1px solid #333;max-height:30vh;overflow:auto}
#info{flex:none;min-width:220px;color:#aaa}
#info b{color:#ddd} #status{color:#999;margin-left:8px}
#list{display:flex;gap:6px;overflow-x:auto;align-items:stretch}
.row{flex:none;padding:6px 8px;border:1px solid #333;border-radius:4px;cursor:pointer;min-width:150px}
.row:hover{background:#222} .row.active{background:#2b3f5c;border-color:#2b3f5c}
.row small{color:#999;display:block;white-space:pre-wrap}
button{background:#333;color:#ddd;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:11px}
dl{margin:4px 0 0} dt{float:left;width:100px;clear:left} dd{margin:0 0 2px 104px;word-break:break-word}
input[type=range]{width:140px;vertical-align:middle}
</style></head><body>
<div id="main">
 <div id="scanwrap"><canvas id="scan"></canvas></div>
 <div id="gridwrap"></div>
 <div id="menu"></div>
 <div id="infopanel"></div>
 <div id="nav">
  <div id="info"><b>__TITLE__</b> · <span id="title"></span><span id="status"></span><dl id="case"></dl></div>
  <div id="list"></div>
 </div>
</div>
<script>__NIIVUE__</script>
<script>const DATA = __DATA__;</script>
<script>
const fmt = v => Array.isArray(v) ? v.map(x => Array.isArray(x) ? `(${fmt(x)})` : fmt(x)).join(", ") || "-"
  : v && typeof v === "object" ? Object.entries(v).map(([k, x]) => `${k}: ${fmt(x)}`).join("; ")
  : typeof v === "number" && !Number.isInteger(v) ? String(+v.toPrecision(4)) : String(v);
const $ = id => document.getElementById(id);
const { SLICE_TYPE, MULTIPLANAR_TYPE, SHOW_RENDER } = niivue;
const KINDS = { axial: SLICE_TYPE.AXIAL, coronal: SLICE_TYPE.CORONAL, sagittal: SLICE_TYPE.SAGITTAL, render: SLICE_TYPE.RENDER };
const bufCache = {};
let scanNV = null, tiles = [], current = null, busy = false, opacity = DATA.opacity, gap = DATA.gap;

async function buffer(key, b64) {
  if (!bufCache[key])
    bufCache[key] = await (await fetch("data:application/octet-stream;base64," + b64)).arrayBuffer();
  return bufCache[key].slice(0);                       // copy: niivue may detach it
}

async function makeNV(canvas, kind) {                 // kind: "row" | tile kind | "surfaces"
  const nv = new niivue.Niivue({
    backColor: [0, 0, 0, 1], isColorbar: false, isRadiologicalConvention: DATA.radiological,
    isOrientCube: kind === "row" || kind === "render" || kind === "surfaces",
    multiplanarLayout: MULTIPLANAR_TYPE.ROW, multiplanarShowRender: SHOW_RENDER.ALWAYS,
    multiplanarEqualSize: true, crosshairGap: 0,
  });
  await nv.attachToCanvas(canvas);
  nv.setSliceType(kind === "row" ? SLICE_TYPE.MULTIPLANAR : kind === "surfaces" ? SLICE_TYPE.RENDER : KINDS[kind]);
  return nv;
}

async function fill(nv, scene, withMasks, kind) {     // load a scene's volumes (+ surfaces) into nv
  while (nv.meshes.length) nv.removeMesh(nv.meshes[0]);
  while (nv.volumes.length) nv.removeVolume(nv.volumes[0]);
  await nv.addVolumeFromUrl({ url: scene.id + ".nii.gz", name: scene.id + ".nii.gz", colormap: "gray",
                              opacity: kind === "surfaces" ? 0 : 1, buffer: await buffer(scene.id, scene.image) });
  if (withMasks)
    for (const [k, L] of scene.layers.entries())
      await nv.addVolumeFromUrl({ url: `${scene.id}_m${k}.nii.gz`, name: `${scene.id}_m${k}.nii.gz`,
                                  colormap: L.cmap, opacity: L.on ? opacity : 0,
                                  buffer: await buffer(`${scene.id}_m${k}`, L.overlay) });
  if (kind === "surfaces")
    for (const [k, L] of scene.layers.entries()) {
      await nv.loadFromArrayBuffer(await buffer(`${scene.id}_s${k}`, L.mesh), `${scene.id}_s${k}.obj`);
      const id = nv.meshes[nv.meshes.length - 1].id;
      nv.setMeshProperty(id, "rgba255", [...L.rgb, 255]);
      nv.setMeshProperty(id, "opacity", opacity);
      nv.setMeshProperty(id, "visible", L.on);
    }
  setGap(nv, scene);
}

function setGap(nv, scene) {
  nv.opts.crosshairGap = gap * scene.gap_reach / scene.gap_radius;
  if (nv.crosshairs3D) nv.crosshairs3D.mm[0] = NaN;    // force NiiVue to rebuild the crosshair
  nv.drawScene();
}

function center(nvs, mm) {
  for (const nv of nvs) {
    nv.scene.crosshairPos = nv.mm2frac(mm);
    nv.scene.pan2Dxyzmm = [0, 0, 0, 1];
    nv.drawScene();
  }
}

function applyLayers(on, nvs, surfaces) {            // on: per-layer booleans
  for (const nv of nvs)
    on.forEach((v, k) => nv.setOpacity(k + 1, v ? opacity : 0));
  if (surfaces)
    surfaces.meshes.forEach((m, k) => { surfaces.setMeshProperty(m.id, "visible", on[k]);
                                        surfaces.setMeshProperty(m.id, "opacity", opacity); });
}

function menu(scene, nvs, surfaces, centerMm) {
  const el = $("menu"); el.innerHTML = "";
  const on = scene.layers.map(L => L.on);
  const boxes = [];
  const add = (node) => { el.appendChild(node); return node; };
  add(Object.assign(document.createElement("b"), { textContent: scene.title }));
  scene.layers.forEach((L, k) => {
    const lab = document.createElement("label");
    const cb = Object.assign(document.createElement("input"), { type: "checkbox", checked: L.on });
    cb.onchange = () => { on[k] = cb.checked; applyLayers(on, nvs, surfaces); };
    boxes.push(cb);
    lab.append(cb, ` `, Object.assign(document.createElement("span"),
               { textContent: "■ ", style: `color:${DATA.css[L.cmap]}` }), L.name);
    add(lab);
  });
  const setAll = v => { boxes.forEach((cb, k) => { cb.checked = v; on[k] = v; }); applyLayers(on, nvs, surfaces); };
  add(Object.assign(document.createElement("button"), { textContent: "all", onclick: () => setAll(true) }));
  add(Object.assign(document.createElement("button"), { textContent: "none", onclick: () => setAll(false) }));
  const slider = (label, value, min, max, step, cb) => {
    const lab = document.createElement("label");
    const r = Object.assign(document.createElement("input"), { type: "range", min, max, step, value });
    const out = Object.assign(document.createElement("span"), { textContent: Number(value).toFixed(2) });
    r.oninput = () => { out.textContent = Number(r.value).toFixed(2); cb(Number(r.value)); };
    lab.append(label + " ", r, " ", out);
    add(lab);
  };
  slider("opacity", opacity, 0.05, 1, 0.05, v => { opacity = v; applyLayers(on, nvs, surfaces); });
  if (centerMm)
    add(Object.assign(document.createElement("button"), { textContent: "⌖ center", onclick: () => center(nvs, centerMm) }));
  slider("crosshair gap", gap, 0, 1, 0.02, v => { gap = v; nvs.forEach(nv => setGap(nv, scene)); });
  add(Object.assign(document.createElement("span"), { textContent: "through ↔ edge", style: "opacity:.7" }));
  const panel = $("infopanel");
  panel.innerHTML = "<table>" + Object.entries(scene.info).map(([k, v]) => {
    const td = s => Object.assign(document.createElement("td"), { textContent: s }).outerHTML;
    return "<tr>" + td(k) + td(fmt(v)) + "</tr>";
  }).join("") + "</table>";
  add(Object.assign(document.createElement("button"), {
    textContent: "ⓘ info", title: "rendering metadata: grids, downsampling, window, sizes",
    onclick: () => { panel.style.display = panel.style.display === "block" ? "none" : "block"; },
  }));
}

async function showScan(mm = null) {
  const scene = DATA.scenes.scan;
  $("scanwrap").style.display = "block"; $("gridwrap").style.display = "none";
  if (!scanNV) { scanNV = await makeNV($("scan"), "row"); await fill(scanNV, scene, true, "row"); }
  scanNV.resizeListener();
  if (mm) center([scanNV], mm);
  menu(scene, [scanNV], null, null);
  $("title").textContent = scene.title + (mm ? " · at region" : "");
}

async function showRegion(id) {
  const scene = DATA.scenes[id];
  $("scanwrap").style.display = "none"; $("gridwrap").style.display = "grid";
  if (!tiles.length) {                                 // build the 8 tiles once, link them in the browser
    const g = $("gridwrap");
    for (const h of ["raw", "with masks", "with masks", "raw · bottom: surfaces only"])
      g.appendChild(Object.assign(document.createElement("div"), { className: "head", textContent: h }));
    for (const [left, right] of DATA.tiles) {
      const outer = right === "render" ? "surfaces" : right;
      for (const [kind, masks] of [[left, false], [left, true], [right, true], [outer, false]]) {
        const cell = Object.assign(document.createElement("div"), { className: "tile" });
        const canvas = document.createElement("canvas");
        cell.appendChild(canvas); g.appendChild(cell);
        tiles.push({ nv: await makeNV(canvas, kind), kind, masks });
      }
    }
    for (const t of tiles) t.nv.broadcastTo(tiles.filter(o => o !== t).map(o => o.nv), { "2d": true, "3d": true });
  }
  if (current !== id)
    for (const t of tiles) await fill(t.nv, scene, t.masks, t.kind);
  const nvs = tiles.map(t => t.nv);
  nvs.forEach(nv => nv.resizeListener());
  center(nvs, scene.center_mm);
  menu(scene, tiles.filter(t => t.masks).map(t => t.nv), tiles.find(t => t.kind === "surfaces").nv, scene.center_mm);
  $("title").textContent = scene.title;
}

async function show(id, mm = null) {
  if (busy) return; busy = true;
  $("status").textContent = "loading…";
  try {
    if (id === "scan") await showScan(mm); else await showRegion(id);
    current = id;
    document.querySelectorAll(".row").forEach(r => r.classList.toggle("active", r.dataset.id === id));
    history.replaceState(null, "", "#" + id);
    $("status").textContent = "";
  } catch (e) { $("status").textContent = "error: " + e; console.error(e); }
  busy = false;
}

function navigation() {
  const info = { ...DATA.summary, ...DATA.meta };
  $("case").innerHTML = Object.entries(info).map(([k, v]) =>
    `<dt>${k}</dt><dd>${fmt(v)}</dd>`).join("");
  const list = $("list");
  const add = (id, inner) => {
    const d = document.createElement("div");
    d.className = "row"; d.dataset.id = id; d.innerHTML = inner;
    d.onclick = () => show(id);
    list.appendChild(d); return d;
  };
  add("scan", `<b>Whole scan</b><small>${DATA.summary.masks.length} masks, ${DATA.summary.regions} regions</small>`);
  for (const r of DATA.rows) {
    const d = add("r" + r.index, `<b>Region #${r.index}</b> <button>in scan</button>` +
      `<small>${r.voxels} vox · ${r.volume_ml.toFixed(2)} ml\nin: ${r.masks.join(", ")}\nat ${r.centroid_mm.map(x => x.toFixed(1)).join(", ")} mm</small>`);
    d.querySelector("button").onclick = e => { e.stopPropagation(); show("scan", r.centroid_mm); };
  }
}

(async () => {
  navigation();
  const start = location.hash.slice(1);
  await show(DATA.scenes[start] ? start : "scan");
  window.addEventListener("resize", () => { if (scanNV) scanNV.resizeListener(); tiles.forEach(t => t.nv.resizeListener()); });
})();
</script></body></html>
"""
