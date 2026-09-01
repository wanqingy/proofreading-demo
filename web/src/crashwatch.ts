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
  /** Longest observed main-thread freeze (ms beyond the sampling interval), and when it happened.
   * A freeze is NOT a crash -- the tab exits cleanly afterwards -- but it is just as unusable, so
   * it is reported separately rather than being invisible. */
  maxStallMs: number;
  maxStallAt: number;
  /** Longest gap that was NOT the page's fault -- see `unrunnable` below. Kept because it explains
   * a hole in the samples, but never reported as a freeze. */
  maxSuspendMs?: number;
}

/** A freeze worth telling the user about. Below this it's ordinary GC / tab throttling noise. */
const STALL_REPORT_MS = 3000;

/** Beyond this, a gap is the machine sleeping rather than the page blocking. Chrome throttles a
 * hidden tab's timers to one per MINUTE after 5 minutes hidden, and a closed lid stops them
 * outright; neither is a fault, and calling them one trains you to ignore the message. */
const MAX_CREDIBLE_STALL_MS = 120_000;

export interface PreviousSession {
  crashed: boolean; // had samples but never recorded a clean exit
  froze: boolean; // exited cleanly, but the main thread was blocked long enough to be unusable
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
    record.maxStallMs ??= 0; // records written before stall tracking existed
    record.maxStallAt ??= 0;
    record.maxSuspendMs ??= 0;
    return {
      crashed: !record.cleanExit,
      froze: record.maxStallMs >= STALL_REPORT_MS,
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
    maxStallMs: 0,
    maxStallAt: 0,
    maxSuspendMs: 0,
  };
  const t0 = performance.now();
  let dirty = false;

  // A late heartbeat only means the MAIN THREAD was blocked if the tab was runnable throughout.
  // Hidden tabs are throttled (to 1/minute after 5 min hidden), Chrome freezes background tabs
  // outright, and a sleeping machine stops them entirely -- measured: a tab frozen 6.5s and then
  // closed normally used to report "previous session FROZE for 6.5s". So watch for the tab being
  // unrunnable at any point between two ticks and don't blame the page for that gap.
  let sawUnrunnable = false;
  const markUnrunnable = () => {
    sawUnrunnable = true;
  };
  const onVisibility = () => {
    if (document.visibilityState !== "visible") sawUnrunnable = true;
  };
  document.addEventListener("visibilitychange", onVisibility);
  window.addEventListener("freeze", markUnrunnable); // Chrome tab-freezing lifecycle
  window.addEventListener("resume", markUnrunnable);

  const flush = () => {
    if (!dirty) return;
    try {
      localStorage.setItem(KEY, JSON.stringify(record));
      dirty = false;
    } catch {
      /* quota / disabled -- diagnostics are best-effort */
    }
  };

  const interval = opts.intervalMs ?? 1000;
  let lastTickAt = performance.now();

  const tick = () => {
    let stats: Record<string, number | string | null> = {};
    try {
      stats = opts.getStats();
    } catch {
      stats = { statsError: 1 };
    }
    // How late this tick was tells us the main thread was BLOCKED, which is the failure mode a
    // death-only recorder cannot see. A frozen tab keeps its process (so `pagehide` still runs on
    // navigation and the session is recorded as a clean exit) while being just as unusable as a
    // crash -- observed for real here: the page stopped answering for 8s+ while zoomed far out and
    // flying, then navigated away normally and reported nothing.
    const now = performance.now();
    const gapMs = Math.max(0, Math.round(now - lastTickAt - interval));
    lastTickAt = now;
    // Blame the page only for a gap it could have caused: tab visible and runnable the whole way
    // through, and short enough to be JS rather than a suspended machine.
    const ourFault =
      !sawUnrunnable && document.visibilityState === "visible" && gapMs < MAX_CREDIBLE_STALL_MS;
    sawUnrunnable = false;
    const stallMs = ourFault ? gapMs : 0;
    if (stallMs > record.maxStallMs) {
      record.maxStallMs = stallMs;
      record.maxStallAt = Math.round(now - t0);
    }
    if (!ourFault && gapMs > record.maxSuspendMs!) record.maxSuspendMs = gapMs;
    record.samples.push({
      t: Math.round(now - t0),
      ...(gapMs > interval ? (ourFault ? { stallMs } : { suspendMs: gapMs }) : {}),
      ...heapFields(),
      ...stats,
    });
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
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("freeze", markUnrunnable);
      window.removeEventListener("resume", markUnrunnable);
    },
    previous: () => previous,
    report() {
      // Two distinct bad endings, both worth reporting and each needing a different fix:
      //   crashed -- the renderer was killed; no clean exit was ever recorded
      //   froze   -- the renderer survived (so it exited cleanly) but the main thread was blocked
      //              long enough to be unusable. Invisible to a death-only check.
      if (!previous || (!previous.crashed && !previous.froze)) return null;
      const { record: r, ageMs } = previous;
      const tail = r.samples.slice(-12);
      const keys = Object.keys(tail[tail.length - 1] ?? {}).filter((k) => k !== "t");
      // Two causes produce "no clean exit" and they are NOT distinguishable from inside the page:
      // the renderer was killed, OR the main thread was still blocked when the tab/window closed,
      // so `pagehide` never got to run (measured: close a wedged page and it looks like a kill).
      // Say both, then let the numbers below discriminate -- heap at the ceiling means a kill,
      // stalls in the tail mean it was already wedged.
      const headline = previous.crashed
        ? `previous session ended WITHOUT a clean exit -- either the renderer was killed, or the ` +
          `page was still blocked when you closed it (a blocked page cannot run its exit handler)`
        : `previous session FROZE: main thread blocked for ${(r.maxStallMs / 1000).toFixed(1)}s ` +
          `at t+${(r.maxStallAt / 1000).toFixed(0)}s (it exited cleanly afterwards)`;
      const lines = [
        `${headline} ${Math.round(ageMs / 1000)}s ago ` +
          `(${r.samples.length} samples, ${r.url})`,
        `  last notes: ${r.notes.slice(-5).map((n) => `${(n.t / 1000).toFixed(0)}s ${n.what}`).join(" | ") || "(none)"}`,
        `  ${["t(s)", ...keys].join("  ")}`,
      ];
      for (const s of tail) {
        lines.push(`  ${[(s.t / 1000).toFixed(0), ...keys.map((k) => String(s[k] ?? ""))].join("  ")}`);
      }
      if (r.maxSuspendMs) {
        // Not a fault, but it explains a hole in the sample timeline -- say so, so the hole isn't
        // read as evidence of one.
        lines.push(
          `  (also paused ${(r.maxSuspendMs / 1000).toFixed(0)}s while hidden/asleep -- not a freeze)`,
        );
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
