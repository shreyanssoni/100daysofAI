// Generates index.html and compositions/*.html from the scene modules.
// Run: node src/build.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { CSS, HELPERS } from "./shared.mjs";
import { AUDIO } from "./audio-cues.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"];
const scenes = [];
for (const n of names) {
  const f = path.join(here, "scenes", n + ".mjs");
  if (fs.existsSync(f)) scenes.push((await import(f)).default);
}

// overlays: compositions that sit above the scenes, placed relative to a scene start
const overlays = [];
for (const f of fs.existsSync(path.join(here, "overlays")) ? fs.readdirSync(path.join(here, "overlays")).filter((x) => x.endsWith(".mjs")) : []) overlays.push((await import(path.join(here, "overlays", f))).default);

fs.mkdirSync(path.join(root, "compositions"), { recursive: true });
for (const s of [...scenes, ...overlays]) {
  const html = `<template>
<style>
${CSS}
${s.css ?? ""}
</style>
<div id="root" data-composition-id="${s.id}" data-width="1920" data-height="1080" data-start="0" data-duration="${s.dur}">
${s.html}
</div>
<script>
(function () {
${HELPERS}
const tl = gsap.timeline({ paused: true });
${s.js}
window.__timelines = window.__timelines || {};
window.__timelines["${s.id}"] = tl;
})();
</script>
</template>
`;
  fs.writeFileSync(path.join(root, "compositions", s.id + ".html"), html);
}

let t = 0;
const hosts = scenes
  .map((s, i) => {
    const h = `      <div id="${s.id}" data-composition-id="${s.id}" data-composition-src="compositions/${s.id}.html" data-start="${t.toFixed(2)}" data-duration="${s.dur}" data-track-index="1" data-track-kind="graphics" data-width="1920" data-height="1080" style="z-index:${i + 1}"></div>`;
    t += s.dur;
    return h;
  })
  .join("\n");
const total = +t.toFixed(2);

const starts = {}; { let acc = 0; for (const sc of scenes) { starts[sc.id] = +acc.toFixed(3); acc += sc.dur; } }
fs.writeFileSync(path.join(here, "timing.json"), JSON.stringify({ starts, total }, null, 2));
const overlayHosts = overlays.map((o) => `      <div id="${o.id}" data-composition-id="${o.id}" data-composition-src="compositions/${o.id}.html" data-start="${(starts[o.at.scene] + o.at.t).toFixed(2)}" data-duration="${o.dur}" data-track-index="9" data-track-kind="graphics" data-width="1920" data-height="1080" style="z-index:100"></div>`).join("\n");
const audio = AUDIO.map((a) => ({ ...a, t: (a.scene ? starts[a.scene] : 0) + a.t })).filter((a) => a.t < total)
  .map((a, i) => `      <audio id="${a.id ?? "sfx" + i}" class="clip" src="assets/${a.file}" data-start="${a.t.toFixed(3)}" ${a.dur ? `data-duration="${a.dur}"` : ""} data-track-index="${a.track ?? 3}" data-volume="${a.vol ?? 0.35}"${a.fadeIn ? ` data-fade-in="${a.fadeIn}"` : ""}${a.fadeOut ? ` data-fade-out="${a.fadeOut}"` : ""}></audio>`)
  .join("\n");

const index = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1920, height=1080" />
    <title>A Morning With Maya — argentm.ai</title>
    <script src="assets/vendor/gsap.min.js"></script>
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      html, body { margin: 0; width: 1920px; height: 1080px; overflow: hidden; background: #FBF8F2; }
      #root { position: relative; width: 1920px; height: 1080px; background: #FBF8F2; }
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="${total}" data-width="1920" data-height="1080">
${hosts}
${overlayHosts}
${audio}
    </div>
    <script>
      window.__timelines = window.__timelines || {};
      window.__timelines["main"] = gsap.timeline({ paused: true });
    </script>
  </body>
</html>
`;
fs.writeFileSync(path.join(root, "index.html"), index);
console.log("built", scenes.map((s) => s.id).join(","), "total", total + "s");
