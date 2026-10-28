// Voice moment: Maya asks, the AI answers. Caption pill + waveform driven by the real voice envelope.
// Spans the end of s4 (Maya's close-up and the bank match) into the start of the s5 montage.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { C } from "../shared.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const ENV = JSON.parse(fs.readFileSync(path.join(here, "vo-env.json"), "utf8"));
const p = "vo";
// local timings (seconds from overlay start)
const M0 = 0.05, A0 = 1.9; // Maya's line, AI's line
const PHR = [[0, 1.46], [1.78, 3.58], [3.9, 5.02], [5.34, 6.76]].map(([a, b]) => [A0 + a, A0 + b]);
const bars = (cls, n) => Array.from({ length: n }, (_, i) => `<i class="${cls}" style="display:block;width:4px;height:30px;border-radius:2px;transform-origin:50% 50%"></i>`).join("");

export default {
  id: p,
  at: { scene: "s4", t: 3.9 },
  dur: 8.95,
  css: `
#root{background:transparent}
.${p}-pill{position:absolute;left:960px;top:888px;transform:translateX(-50%);display:flex;align-items:center;gap:16px;padding:12px 24px 12px 14px;background:rgba(255,255,255,.96);border:1px solid #EAE3D7;border-radius:999px;box-shadow:0 2px 6px rgba(70,50,30,.08),0 18px 40px rgba(70,50,30,.14);white-space:nowrap;opacity:0}
.${p}-who{display:flex;align-items:center;gap:10px;padding-right:16px;border-right:1px solid #EFE8DD}
.${p}-txt{font-size:25px;letter-spacing:-.01em;color:${C.ink}}
.${p}-seg{color:#BDB5A9}
.${p}-seg b{color:inherit}
.${p}-wave{display:flex;align-items:center;gap:4px;height:34px}
`,
  html: `
<div id="${p}-m" class="${p}-pill">
  <div class="${p}-who"><span style="width:36px;height:36px;border-radius:50%;background:${C.bronzeL};color:${C.bronzeD};font-weight:700;font-size:13px;display:inline-flex;align-items:center;justify-content:center">MC</span><span style="font-size:16px;font-weight:700;color:${C.ink2}">Maya</span><span class="${p}-wave">${bars(p + "-mb", 5)}</span></div>
  <span class="${p}-txt">“Show me how this matched.”</span>
</div>
<div id="${p}-a" class="${p}-pill">
  <div class="${p}-who"><span class="chip chip-bronze" style="font-size:15px;padding:5px 12px">AI</span><span class="${p}-wave">${bars(p + "-ab", 9)}</span></div>
  <span class="${p}-txt"><span class="${p}-seg" id="${p}-s0"><b>64</b> lines</span><span class="${p}-seg" id="${p}-s1"> across 3 Helix accounts ·</span><span class="${p}-seg" id="${p}-s2"> <b>59</b> matched ·</span><span class="${p}-seg" id="${p}-s3"> <b>2</b> need you</span></span>
</div>
`,
  js: `
const EM = ${JSON.stringify(ENV.maya)}, EA = ${JSON.stringify(ENV.ai)};
const envAt = (E, t) => { const i = Math.floor(t * 30); return i >= 0 && i < E.length ? E[i] : 0; };
const mb = [...document.querySelectorAll('.${p}-mb')], ab = [...document.querySelectorAll('.${p}-ab')];
const shapeM = [0.55, 0.85, 1, 0.8, 0.5], shapeA = [0.35, 0.6, 0.85, 1, 0.9, 1, 0.8, 0.6, 0.35];
drive(tl, ${8.95}, (t) => {
  mb.forEach((b, i) => { const v = envAt(EM, t - ${M0} - i * 0.025); b.style.transform = 'scaleY(' + (0.16 + 0.84 * v * shapeM[i]).toFixed(3) + ')'; b.style.background = '${C.ink2}'; });
  ab.forEach((b, i) => { const v = envAt(EA, t - ${A0} - Math.abs(i - 4) * 0.02); b.style.transform = 'scaleY(' + (0.12 + 0.88 * v * shapeA[i]).toFixed(3) + ')'; b.style.background = '${C.bronze}'; });
});
tl.fromTo('#${p}-m', { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: 0.35, ease: 'expo.out' }, 0.0);
tl.to('#${p}-m', { opacity: 0, y: -8, duration: 0.2, ease: 'power2.in' }, 1.72);
tl.fromTo('#${p}-a', { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: 0.35, ease: 'expo.out' }, 1.8);
${PHR.map(([a], i) => `tl.fromTo('#${p}-s${i}', { color: '#BDB5A9' }, { color: '${i === 3 ? C.bronzeD : C.ink}', duration: 0.25 }, ${a.toFixed(2)});`).join("\n")}
tl.to('#${p}-a', { opacity: 0, y: -8, duration: 0.25, ease: 'power2.in' }, 8.72);
`,
};
