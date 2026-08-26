// Breadcrumb recorder for crashes that kill the renderer process.
//
// A blank page (HUD and all -- the HUD is static HTML) means the DOM is gone, i.e. Chrome killed
// the renderer, usually on memory. Nothing in the page can report that: `window.onerror` never
// fires, the console is wiped, and DevTools detaches. So instead of trying to catch the crash we
// leave a trail: sample a few numbers every second into a ring buffer in localStorage, which lives
// in the browser process and therefore SURVIVES the renderer dying. After the reload we read the
// last samples and see what was climbing.
//
// Cost is deliberately tiny -- one small JSON write per flush, no allocation-heavy stats calls --
// because a recorder that itself pressures memory would corrupt the measurement it exists to make.

const KEY = "flykernel.crashwatch.v1";
const MAX_SAMPLES = 90; // ~90s of history at the default 1s interval
const MAX_NOTES = 30;

export interface CrashSample {
  t: number; // ms since the recorder started
  [field: string]: number | string | null;
}

export interface CrashRecord {
  startedAt: number; // wall clock (ms) -- only for showing "how long ago"
  cleanExit: boolean;
  url: string;
  samples: CrashSample[];
  notes: { t: number; what: string }[];
}

export interface PreviousSession {
  crashed: boolean; // had samples but never recorded a clean exit
  record: CrashRecord;
  ageMs: number;
}

export interface CrashWatch {
  note(what: string): void;
  markCleanExit(): void;
  stop(): void;
  /** The tail of the previous session, if it ended without a clean exit. */
  previous(): PreviousSession | null;
  /** Human-readable summary of the previous session, or null if it exited cleanly. */
  report(): string | null;
}

function readPrevious(): PreviousSession | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const record = JSON.parse(raw) as CrashRecord;
    if (!record?.samples?.length) return null;
    return {
      crashed: !record.cleanExit,
      record,
      ageMs: Date.now() - record.startedAt,
    };
  } catch {
    return null; // malformed / storage disabled -- never let diagnostics break the page
  }
}

function heapFields(): Record<string, number> {
  // Chrome-only and coarse-grained without --enable-precise-memory-info, but the TREND is what
  // matters here, and the trend is exactly what a coarse counter still shows.
  const m = (performance as any).memory;
  if (!m) return {};
  return {
    heapMB: Math.round(m.usedJSHeapSize / 1e6),
    heapLimitMB: Math.round(m.jsHeapSizeLimit / 1e6),
  };
}

export function startCrashWatch(opts: {
  getStats: () => Record<string, number | string | null>;
  intervalMs?: number;
}): CrashWatch {
  const previous = readPrevious();

  const record: CrashRecord = {
    startedAt: Date.now(),
    cleanExit: false,
    url: location.href,
    samples: [],
    notes: [],
  };
  const t0 = performance.now();
  let dirty = false;

  const flush = () => {
    if (!dirty) return;
    try {
      localStorage.setItem(KEY, JSON.stringify(record));
      dirty = false;
    } catch {
      /* quota / disabled -- diagnostics are best-effort */
    }
  };

  const tick = () => {
    let stats: Record<string, number | string | null> = {};
    try {
      stats = opts.getStats();
    } catch {
      stats = { statsError: 1 };
    }
    record.samples.push({ t: Math.round(performance.now() - t0), ...heapFields(), ...stats });
    if (record.samples.length > MAX_SAMPLES) record.samples.shift();
    dirty = true;
    flush(); // flush every sample: the whole point is to survive an unannounced kill
  };

  const timer = window.setInterval(tick, opts.intervalMs ?? 1000);
  tick(); // one immediately, so a crash within the first second still leaves something

  // A normal reload/navigation must not look like a crash.
  const onExit = () => {
    record.cleanExit = true;
    dirty = true;
    flush();
  };
  window.addEventListener("pagehide", onExit);
  window.addEventListener("beforeunload", onExit);

  return {
    note(what: string) {
      record.notes.push({ t: Math.round(performance.now() - t0), what });
      if (record.notes.length > MAX_NOTES) record.notes.shift();
      dirty = true;
      flush();
    },
    markCleanExit: onExit,
    stop() {
      window.clearInterval(timer);
      window.removeEventListener("pagehide", onExit);
      window.removeEventListener("beforeunload", onExit);
    },
    previous: () => previous,
    report() {
      if (!previous?.crashed) return null;
      const { record: r, ageMs } = previous;
      const tail = r.samples.slice(-12);
      const keys = Object.keys(tail[tail.length - 1] ?? {}).filter((k) => k !== "t");
      const lines = [
        `previous session ended WITHOUT a clean exit ${Math.round(ageMs / 1000)}s ago ` +
          `(${r.samples.length} samples, ${r.url})`,
        `  last notes: ${r.notes.slice(-5).map((n) => `${(n.t / 1000).toFixed(0)}s ${n.what}`).join(" | ") || "(none)"}`,
        `  ${["t(s)", ...keys].join("  ")}`,
      ];
      for (const s of tail) {
        lines.push(`  ${[(s.t / 1000).toFixed(0), ...keys.map((k) => String(s[k] ?? ""))].join("  ")}`);
      }
      const first = r.samples[0];
      const last = r.samples[r.samples.length - 1];
      if (typeof first.heapMB === "number" && typeof last.heapMB === "number") {
        lines.push(
          `  heap ${first.heapMB}MB -> ${last.heapMB}MB of ${last.heapLimitMB}MB limit` +
            (Number(last.heapMB) > 0.85 * Number(last.heapLimitMB)
              ? "  <- at the JS heap ceiling: renderer OOM"
              : "  (heap was not near its ceiling -- look at GPU/chunk memory instead)"),
        );
      }
      return lines.join("\n");
    },
  };
}
