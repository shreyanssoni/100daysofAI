// 0:00–0:07 — The morning. Maya walks in, sets down her coffee, sits, wakes the monitor.
import { foreground, office, colleague, mayaSide, mayaBack, chair, appShell, dashContent, SCREEN, words, SIDE_POSE, BACK_POSE } from "../shared.mjs";

const p = "s1";
export default {
  id: p,
  dur: 7,
  css: `
#${p}-newrow{display:none}
#${p}-screen{left:${SCREEN.x}px;top:${SCREEN.y}px;width:${SCREEN.w}px;height:${SCREEN.h}px;overflow:hidden;background:#FBF9F5}
#${p}-screen .app{transform:scale(${SCREEN.w / 1280})}
#${p}-off{position:absolute;inset:0;background:#2E2C2A}
#${p}-t1{left:120px;top:96px}
#${p}-t2{left:120px;top:96px}
`,
  html: `
<div id="${p}-world" class="world">
  ${office(p, { colleague: colleague(p), cupOnDesk: false, clock: [8, 58] })}
  <svg class="full" viewBox="0 0 1920 1080" style="position:absolute;left:0;top:0">
    ${mayaBack(p)}
    ${chair(p)}
    ${mayaSide(p)}
  </svg>
  <div id="${p}-screen" class="abs">${appShell("Dashboard", dashContent(p))}<div id="${p}-off"></div></div>
  ${foreground(p)}
</div>
<div id="${p}-t1" class="t-hero">${words("Every morning,", p + "-w1")}<br>${words("payments arrive.", p + "-w1")}</div>
<div id="${p}-t2" class="t-hero">${words("But the", p + "-w2")} <span class="w ${p}-w2 serif">answers</span> ${words("don't.", p + "-w2")}</div>
`,
  js: `
${SIDE_POSE(p)}
${BACK_POSE(p)}
const world = document.getElementById('${p}-world');
const V = 383, T_DEC = 2.7, X0 = -190;
const walkX = (t) => t < T_DEC ? X0 + V*t : X0 + V*T_DEC + V*0.6/2 * (1 - Math.pow(1 - clamp((t - T_DEC)/0.6, 0, 1), 2));
drive(tl, 7, (t) => {
  // ---- Maya walking (side view) ----
  const x = walkX(t);
  const phase = (x - X0) / 230 * Math.PI * 2;
  const amp = 1 - sm((t - 2.95) / 0.35);
  const turn = sm((t - 3.3) / 0.16);
  poseSide({ x, phase, amp, armF: -14, foreF: -80, cup: 1, op: 1 - turn, head: 0 });
  // ---- Maya back view: set coffee down, pull chair, sit, reach keyboard ----
  const sit = eio((t - 4.05) / 0.6);
  const bx = lerp(960, 1095, eio((t - 3.98) / 0.62));
  const reach = key(t, [[3.45, 0], [3.8, 1], [3.95, 1], [4.2, 0]]);
  const type = key(t, [[4.65, 0], [5.05, 1]]);
  const settle = Math.sin(clamp((t - 4.6) / 0.35, 0, 1) * Math.PI) * 3;
  const bob = (t < 3.45 ? Math.sin(clamp((t - 3.3) / 0.15, 0, 1) * Math.PI) * 3 : 0) + settle;
  poseBack({
    x: bx, op: turn, sit, bob,
    armR: lerp(lerp(-4, -8, reach), -12, type), foreR: lerp(lerp(0, -62, reach), -18, type),
    armL: lerp(4, 10, type), foreL: lerp(0, 14, type),
    cup: t < 3.86 ? 1 : 0, head: lerp(0, 4, type) + key(t, [[5.4, 0], [5.8, -2], [6.4, 0]]),
  });
  setO('${p}-deskcup', t >= 3.86 ? 1 : 0);
  // chair: pulled out, then takes her weight
  const cx = 1095 + 18 * key(t, [[3.7, 0], [3.95, 1], [4.4, 0]]);
  setT('${p}-chair', 'translate(' + cx.toFixed(2) + ' 798)');
  setT('${p}-chair-seat', 'translate(0 ' + (4 * Math.sin(clamp((t - 4.5) / 0.45, 0, 1) * Math.PI)).toFixed(2) + ')');
  // ---- colleague looks up and waves as she passes ----
  const look = key(t, [[1.2, 0], [1.5, 1], [2.6, 1], [3.0, 0]]);
  const wave = key(t, [[1.45, 0], [1.75, 1], [2.35, 1], [2.7, 0]]);
  rot('${p}-col-head', -10 * look, 4, -226);
  rot('${p}-col-arm', -105 * wave + 7 * Math.sin(t * 13) * wave, 0, -204);
  rot('${p}-col-fore', (1 - wave) * 3 * Math.sin(t * 22), 0, -146);
  // ---- camera ----
  const s = key(t, [[0, 1.06], [1.2, 1.1], [3.1, 1.3], [4.7, 1.36], [6.5, 3.2], [7, 3.25]]);
  let ccx = key(t, [[0, 760], [1.2, 780], [3.1, 1060], [4.7, 1140], [6.5, 1232], [7, 1234]]);
  let ccy = key(t, [[0, 560], [3.1, 585], [4.7, 590], [6.5, 538], [7, 538]]);
  ccx = clamp(ccx, 960 / s, 1920 - 960 / s); ccy = clamp(ccy, 540 / s, 1080 - 540 / s);
  world.style.transform = 'translate(' + (960 - ccx * s).toFixed(2) + 'px,' + (540 - ccy * s).toFixed(2) + 'px) scale(' + s.toFixed(4) + ')';
});
// monitor wakes
tl.fromTo('#${p}-off', { opacity: 1 }, { opacity: 0, duration: 0.45, ease: 'power2.out' }, 4.95);
// titles
revealWords(tl, '.${p}-w1', 0.55, 0.09);
hideWords(tl, '.${p}-w1', 3.05);
revealWords(tl, '.${p}-w2', 5.15, 0.1);
hideWords(tl, '.${p}-w2', 6.72, 0.03);
`,
};
