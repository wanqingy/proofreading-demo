// EM proofreading — Phase B: annotation queue with Spelunker links.
//
// Lightweight page — no neuroglancer. Fetches the WAL annotation list (which already
// contains backend-generated Spelunker URLs via nglui), sorts it by error type
// (merge → split → extend → question), and renders each annotation as a "spelunker ↗"
// link that opens Spelunker at that annotation's xyz.

const params = new URLSearchParams(location.search);
const API = (params.get("api") || "http://localhost:8000").replace(/\/$/, "");
const ROOT_ID = params.get("root") || "864691135413357554";
const DATASTACK = params.get("datastack") || "minnie65_public";

const TAG_COLORS: Record<string, string> = {
  "merge error": "#ff3333",
  "split error": "#33aaff",
  extend: "#33ff66",
  question: "#ffcc00",
};
const TAG_PRIORITY: Record<string, number> = {
  "merge error": 0,
  "split error": 1,
  extend: 2,
  question: 3,
};

interface Ann {
  uuid: string;
  tag: string;
  xyz: number[]; // nm
  supervoxel: string | null;
  done: boolean;
  spelunker_url?: string;
}

const $ = (id: string) => document.getElementById(id)!;
const setStatus = (msg: string, cls = "") => {
  $("status").textContent = msg;
  $("status").className = cls;
};

// --- per-annotation done/todo status (persisted in the WAL via backend) ---
let doneSet: Set<string> = new Set();

async function toggleDone(uuid: string) {
  // optimistic update
  if (doneSet.has(uuid)) doneSet.delete(uuid); else doneSet.add(uuid);
  const cell = document.getElementById(`st-${uuid}`);
  if (cell) cell.innerHTML = statusBadge(uuid);
  updateAnnCount();
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/annotations/${uuid}/status`, { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    // reconcile with server truth (in case optimistic guess was wrong)
    if (resp.done) doneSet.add(uuid); else doneSet.delete(uuid);
    if (cell) cell.innerHTML = statusBadge(uuid);
    updateAnnCount();
  } catch (e) {
    // revert on failure
    if (doneSet.has(uuid)) doneSet.delete(uuid); else doneSet.add(uuid);
    if (cell) cell.innerHTML = statusBadge(uuid);
    updateAnnCount();
    console.warn("[review] toggleDone failed", e);
  }
}

function statusBadge(uuid: string): string {
  const done = doneSet.has(uuid);
  return done
    ? `<button data-uuid="${uuid}" style="background:#1a3a22;color:#8aff9c;border:1px solid #3a784a;border-radius:4px;padding:2px 8px;cursor:pointer;font:inherit;font-size:11px">✓ done</button>`
    : `<button data-uuid="${uuid}" style="background:#1c2740;color:#6a7a9a;border:1px solid #2c3a5a;border-radius:4px;padding:2px 8px;cursor:pointer;font:inherit;font-size:11px">todo</button>`;
}

let _annCount = 0;
function updateAnnCount() {
  const done = doneSet.size;
  $("anncount").textContent =
    `${_annCount} annotation${_annCount === 1 ? "" : "s"} · ${done} done · ${_annCount - done} todo · root …${ROOT_ID.slice(-6)}`;
}

function renderAnnotations(anns: Ann[]) {
  const sorted = [...anns].sort(
    (a, b) => (TAG_PRIORITY[a.tag] ?? 9) - (TAG_PRIORITY[b.tag] ?? 9),
  );
  _annCount = sorted.length;
  const tbody = $("anntbody");
  if (!sorted.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="color:#9fb0d0;text-align:center;padding:14px">No annotations yet — run Phase A first to add error marks</td></tr>`;
    updateAnnCount();
    return;
  }
  tbody.innerHTML = sorted
    .map((ann, i) => {
      const color = TAG_COLORS[ann.tag] ?? "#d7e3ff";
      const xMm = (ann.xyz[0] / 1000).toFixed(0);
      const yMm = (ann.xyz[1] / 1000).toFixed(0);
      const zMm = (ann.xyz[2] / 1000).toFixed(0);
      const sv = ann.supervoxel
        ? `<span style="font-size:10px;color:#9fb0d0">${ann.supervoxel.slice(-8)}</span>`
        : `<span style="color:#2c3a5a">—</span>`;
      const editCell = ann.spelunker_url
        ? `<a class="splink" href="${ann.spelunker_url}" target="_blank" rel="noopener">spelunker ↗</a>`
        : `<span style="color:#4a5a80">—</span>`;
      return `<tr>
        <td style="text-align:center;color:#4a5a80">${i + 1}</td>
        <td><span class="tag" style="color:${color}">${ann.tag}</span></td>
        <td style="color:#9fb0d0">[${xMm}, ${yMm}, ${zMm}]</td>
        <td>${sv}</td>
        <td>${editCell}</td>
        <td id="st-${ann.uuid}">${statusBadge(ann.uuid)}</td>
      </tr>`;
    })
    .join("");
  updateAnnCount();
  // event delegation: one listener on tbody handles all toggle clicks
  tbody.addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLElement>("[data-uuid]");
    if (btn?.dataset.uuid) toggleDone(btn.dataset.uuid);
  });
}

async function resolveSupervoxels() {
  setStatus("resolving supervoxels…");
  ($("resolve") as HTMLButtonElement).disabled = true;
  try {
    const r = await fetch(`${API}/api/cells/${ROOT_ID}/resolve`, { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const resp = await r.json();
    const newRoot: string = resp.new_root_id;
    const count: number = resp.resolved_count;
    setStatus(`resolved ${count} annotation${count === 1 ? "" : "s"} · new root: ${newRoot}`, "ok");
    const btn = $("newroot") as HTMLButtonElement;
    btn.textContent = `Load new root (…${newRoot.slice(-6)})`;
    btn.style.display = "";
    btn.onclick = () => {
      const p = new URLSearchParams(location.search);
      p.set("root", newRoot);
      location.search = p.toString();
    };
    // refresh the table with updated spelunker URLs (new root_id in links)
    if (resp.annotations) renderAnnotations(resp.annotations);
  } catch (e) {
    setStatus(`✗ resolve failed: ${e}`, "warn");
    ($("resolve") as HTMLButtonElement).disabled = false;
  }
}

async function main() {
  const cellInput = $("cellid") as HTMLInputElement;
  cellInput.value = ROOT_ID;
  const loadCell = () => {
    const id = cellInput.value.trim();
    if (!id || id === ROOT_ID) return;
    const p = new URLSearchParams(location.search);
    p.set("root", id);
    location.search = p.toString();
  };
  ($("loadcell") as HTMLButtonElement).onclick = loadCell;
  cellInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadCell();
  });
  ($("resolve") as HTMLButtonElement).onclick = resolveSupervoxels;

  setStatus(`opening cell ${ROOT_ID}…`);
  try {
    const hr = await fetch(`${API}/api/cells`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ root_id: ROOT_ID, datastack: DATASTACK }),
    });
    if (!hr.ok) throw new Error(`HTTP ${hr.status}`);
  } catch (e) {
    setStatus(
      `✗ backend not reachable — start it with: uv run --extra em --extra serve python -m proofreading.em.serve  (${e})`,
      "warn",
    );
    return;
  }

  let anns: Ann[] = [];
  try {
    const ar = await fetch(`${API}/api/cells/${ROOT_ID}/annotations`);
    if (!ar.ok) throw new Error(`HTTP ${ar.status}`);
    anns = (await ar.json()).annotations || [];
  } catch (e) {
    setStatus(`✗ annotations fetch failed: ${e}`, "warn");
    return;
  }

  // seed done state from the WAL (persisted on backend, survives reload/browser changes)
  doneSet = new Set(anns.filter((a: Ann) => a.done).map((a: Ann) => a.uuid));
  renderAnnotations(anns);
  setStatus(
    anns.length
      ? `${anns.length} annotations loaded — click spelunker ↗ to open each in Spelunker`
      : "no annotations yet — run Phase A first",
    anns.length ? "ok" : "warn",
  );
}

main();
