import { defineConfig } from "vite";

// Phase 0 spike build config.
//
// neuroglancer ships compiled ESM under lib/ and constructs its web workers with
// `new Worker(new URL("./chunk_worker.bundle.js", import.meta.url), {type:"module"})`.
// Vite's dep optimizer (esbuild) would rewrite those URLs and break the workers, so we
// EXCLUDE neuroglancer from optimization and let Vite serve its modules natively.
export default defineConfig({
  optimizeDeps: {
    // exclude neuroglancer itself so esbuild doesn't rewrite its `import.meta.url` workers...
    exclude: ["neuroglancer"],
    // ...but because it's excluded, Vite's scanner never crawls into it to pre-bundle its
    // CommonJS dependencies, so they'd be served raw with no default/named export interop
    // (e.g. `import CodeMirror from "codemirror"` -> "does not provide an export named
    // default"). Pre-bundle those CJS deps explicitly so the interop shim is generated.
    include: [
      // core-js polyfills pulled in by neuroglancer's util/polyfills.js (TC39 `using`);
      // pure CommonJS, so let esbuild resolve their `require` graph.
      "core-js/actual/symbol/dispose.js",
      "core-js/actual/symbol/async-dispose.js",
      "codemirror",
      // codemirror v5 modes/addons are UMD; served raw as ESM they'd hit the
      // `mod(CodeMirror)` global-fallback branch and throw. Optimize them so esbuild
      // takes the CommonJS `require("../../lib/codemirror")` branch (shared instance).
      "codemirror/mode/javascript/javascript.js",
      "codemirror/addon/fold/foldcode.js",
      "codemirror/addon/fold/foldgutter.js",
      "codemirror/addon/fold/brace-fold.js",
      "codemirror/addon/lint/lint.js",
      "crc-32",
      "crc-32/crc32c.js",
      "nifti-reader-js",
      "msgpackr",
      "numcodecs",
    ],
  },
  worker: { format: "es" },
  server: { port: 5173, strictPort: false },
  // neuroglancer references a handful of optional build-time globals as bare identifiers
  // (all behind `typeof ... !== "undefined"` guards). Define them as `undefined` so the
  // guards take the default branch instead of throwing ReferenceError.
  define: {
    NEUROGLANCER_BUILD_INFO: "undefined",
    NEUROGLANCER_DEFAULT_STATE_FRAGMENT: "undefined",
    NEUROGLANCER_OVERRIDE_DEFAULT_VIEWER_OPTIONS: "undefined",
    NEUROGLANCER_SHOW_OBJECT_SELECTION_TOOLTIP: "undefined",
    NEUROGLANCER_SHOW_LAYER_BAR_EXTRA_BUTTONS: "undefined",
    NEUROGLANCER_CREDIT_LINK: "undefined",
    NEUROGLANCER_GOOGLE_TAG_MANAGER: "undefined",
    NEUROGLANCER_BRAINMAPS_SERVERS: "undefined",
    NEUROGLANCER_BRAINMAPS_CLIENT_ID: "undefined",
  },
});
