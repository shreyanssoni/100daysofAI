// 0:51–0:55.6 — Apply and post. Records travel to the ERP archive; the audit trail unrolls and folds.
import { C, appShell, worldLayer, screenCSS, CAM, BACK_POSE, words, cursorSVG, brandMark } from "../shared.mjs";

const p = "s7";
const REC = [
  ["Applied to 59 invoices", "$48,275.40", C.ink],
  ["Chargebacks · $0 invoice", "−$515.75", C.redD],
  ["Chargeback recovery", "+$212.75", C.bronzeD],
  ["Unapplied credit · held", "$640.00", C.ink],
];
const posted = appShell("Posting & Audit", `
  <div style="display:flex;align-items:center;gap:14px"><div class="a-title">Posted to ERP</div><span class="a-pill" style="background:#EEF2EA;color:#4F6B45;font-size:15px">✓ Complete</span></div>
  <div class="a-sub" style="font-size:20px;margin:6px 0 26px">All payment value accounted for</div>
  <div class="a-card" style="padding:8px 24px">
    ${REC.map((r) => `<div style="display:flex;justify-content:space-between;font-size:20px;padding:13px 0;border-bottom:1px solid #F0EAE0"><span>${r[0]}</span><b class="num" style="color:${r[2]}">${r[1]}</b></div>`).join("")}
    <div style="display:flex;justify-content:space-between;font-size:22px;padding:14px 0;font-weight:700"><span>Bank deposit</span><span class="num">$48,612.40</span></div>
  </div>`);
// world-space records that fly from the monitor to the shelf
const DOCS = [[1508, 318], [1544, 318], [1580, 468], [1616, 468], [1652, 618], [1688, 618]];
const docs = DOCS.map((d, i) => `<div id="${p}-d${i}" class="abs" style="left:0;top:0;width:24px;height:30px;border-radius:3px;background:${C.bronzeL};border:1.5px solid ${C.bronze};opacity:0"><div style="margin:6px 4px 0;height:2.5px;background:${C.bronze};opacity:.6"></div><div style="margin:3px 4px 0;height:2.5px;background:${C.bronze};opacity:.4"></div></div>`).join("");
const amber = `<div id="${p}-amber" class="abs" style="left:1410px;top:468px;width:58px;height:34px;border-radius:5px;background:${C.amberL};border:1.5px solid ${C.amber};font:700 8.5px Inter;color:${C.amberD};padding:4px 5px;line-height:1.25;opacity:0">INV-55203<br>$118.40 open</div>`;

export default {
  id: p,
  dur: 4.6,
  css: `
${screenCSS(p)}
#${p}-ui{transform-origin:0 0;background:${C.ivory}}
#${p}-zoom{transform-origin:0 0}
#${p}-world{opacity:0}
#${p}-audit{left:260px;top:620px;width:1400px;padding:26px 34px;opacity:0;transform-origin:50% 100%}
#${p}-rec{left:960px;top:930px;transform:translateX(-50%);opacity:0}
#${p}-t{left:110px;top:80px}
`,
  html: `
${worldLayer(p, posted, { extra: docs + amber })}
<div id="${p}-ui" class="full">
  <div id="${p}-zoom" class="full">
    <div class="card" style="left:360px;top:220px;width:1200px;height:640px;padding:44px 52px">
      <div style="display:flex;align-items:center;gap:16px">${brandMark(34)}<span style="font-size:22px;color:${C.ink2}">Remittance HX-778120 · Helix Health Distribution</span></div>
      <div style="font-size:52px;font-weight:700;letter-spacing:-.03em;margin:18px 0 22px">Ready to post</div>
      ${REC.map((r) => `<div style="display:flex;justify-content:space-between;width:620px;font-size:26px;padding:12px 0;border-bottom:1px solid #F0EAE0"><span style="color:${C.ink2}">${r[0]}</span><b class="num" style="color:${r[2]}">${r[1]}</b></div>`).join("")}
      <div style="display:flex;justify-content:space-between;width:620px;font-size:28px;padding:14px 0;font-weight:700"><span>= Bank deposit</span><span class="num">$48,612.40</span></div>
      <div id="${p}-apply" class="btn btn-bronze" style="position:absolute;right:52px;bottom:52px;width:330px;height:92px;font-size:32px;gap:14px"><span id="${p}-albl">Apply &amp; post</span>
        <svg id="${p}-spin" width="34" height="34" viewBox="0 0 34 34" style="position:absolute;opacity:0"><circle cx="17" cy="17" r="13" stroke="#fff" stroke-opacity=".35" stroke-width="4" fill="none"/><path d="M17 4 A13 13 0 0 1 30 17" stroke="#fff" stroke-width="4" fill="none" stroke-linecap="round"/></svg>
        <svg id="${p}-done" width="38" height="38" viewBox="0 0 22 22" style="position:absolute;opacity:0"><path d="M4 11.5 L9 16 L18 6" stroke="#fff" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
    </div>
  </div>
</div>
<div id="${p}-audit" class="card">
  <div style="display:flex;font-size:18px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:${C.ink2};padding-bottom:12px;border-bottom:1px solid ${C.line}"><span style="flex:1">Who</span><span style="flex:2.6">What</span><span style="flex:.8">When</span><span style="flex:1.6">Why</span></div>
  ${[["argentm.ai", "Applied 59 invoices across HX-1001–1003", "9:12 AM", "Invoice no. + amount"], ["argentm.ai", "Held $640.00 as unapplied credit", "9:12 AM", "Duplicate of paid INV-54877"], ["Maya Chen", "Approved $0 invoice for CB-40112, CB-40113", "9:14 AM", "Chargeback handling"], ["Maya Chen", "Kept INV-55203 open · $118.40", "9:14 AM", "Short payment"]].map((r) => `<div style="display:flex;font-size:22px;padding:12px 0;border-bottom:1px solid #F0EAE0"><b style="flex:1">${r[0]}</b><span style="flex:2.6">${r[1]}</span><span style="flex:.8;color:${C.ink2}">${r[2]}</span><span style="flex:1.6;color:${C.ink2}">${r[3]}</span></div>`).join("")}
</div>
<div id="${p}-rec" class="abs"><span class="chip chip-white" style="font-size:22px;padding:10px 20px">${brandMark(22)} Audit record AR-20931 · 6 entries · Who · What · When · Why</span></div>
<div id="${p}-t" class="t-hero">${words("Applied.", p + "-w")} ${words("Posted.", p + "-w")} <span class="w ${p}-w serif">Traceable.</span></div>
${cursorSVG(p + "-cur")}
`,
  js: `
${BACK_POSE(p)}
${CAM(p)}
drive(tl, 4.6, (t) => {
  seated(t, { head: 4 + key(t, [[1.4, 0], [1.8, 6], [2.6, 6], [3.2, 2]]), headX: 0 });
  cam(key(t, [[1.2, 1.55], [4.6, 1.5]]), 1380, 520);
});
// close on the button
tl.fromTo('#${p}-zoom', { scale: 1.55, x: -1540 * 0.55 + 0, y: -820 * 0.55 }, { scale: 1.3, x: -1540 * 0.3, y: -820 * 0.3, duration: 1.0, ease: 'power2.out' }, 0);
tl.fromTo('#${p}-cur', { x: 1700, y: 1150, opacity: 1 }, { x: 1400, y: 760, duration: 0.65, ease: 'power3.out' }, 0.1);
tl.to('#${p}-cur', { scale: 0.86, transformOrigin: '10px 6px', duration: 0.07, yoyo: true, repeat: 1 }, 0.8);
tl.to('#${p}-apply', { scale: 0.95, duration: 0.08, yoyo: true, repeat: 1, ease: 'power2.out' }, 0.8);
tl.to('#${p}-albl', { opacity: 0, duration: 0.1 }, 0.9);
tl.fromTo('#${p}-spin', { opacity: 0, rotation: 0 }, { opacity: 1, rotation: 300, transformOrigin: '50% 50%', duration: 0.3, ease: 'none' }, 0.9);
tl.to('#${p}-spin', { opacity: 0, duration: 0.08 }, 1.2);
tl.fromTo('#${p}-done', { opacity: 0, scale: 0.6 }, { opacity: 1, scale: 1, duration: 0.25, ease: 'back.out(2)' }, 1.2);
tl.to('#${p}-cur', { opacity: 0, duration: 0.1 }, 1.25);
// pull out of the monitor
tl.fromTo('#${p}-world', { opacity: 0 }, { opacity: 1, duration: 0.3 }, 1.35);
tl.to('#${p}-ui', { scale: 344 / 1920, x: 650, y: 434.6, duration: 0.55, ease: 'power3.inOut' }, 1.3);
tl.to('#${p}-ui', { opacity: 0, duration: 0.2 }, 1.8);
// records travel to the ERP archive
${DOCS.map((d, i) => `tl.fromTo('#${p}-d${i}', { x: 1300, y: 500, opacity: 0, scale: 0.6 }, { keyframes: [{ x: ${(1300 + d[0]) / 2}, y: ${Math.min(d[1], 500) - 90}, opacity: 1, scale: 1, duration: 0.3, ease: 'power2.out' }, { x: ${d[0] + 3}, y: ${d[1] + 40}, duration: 0.3, ease: 'power2.in' }] }, ${(1.8 + i * 0.09).toFixed(2)});
tl.to('#${p}-d${i}', { backgroundColor: '#FFFFFF', borderColor: '#D9CFC0', duration: 0.3 }, ${(2.45 + i * 0.09).toFixed(2)});
tl.fromTo('#${p}-bind-${Math.floor(i / 2)}-${i}', { attr: { stroke: '#CDBFA9' } }, { attr: { stroke: '${C.bronze}' }, duration: 0.12, yoyo: true, repeat: 1 }, ${(2.38 + i * 0.09).toFixed(2)});`).join("\n")}
tl.fromTo('#${p}-amber', { opacity: 0, x: -40 }, { opacity: 1, x: 0, duration: 0.4, ease: 'expo.out' }, 2.05);
// audit trail: unroll, then fold into one record
tl.fromTo('#${p}-audit', { opacity: 0, clipPath: 'inset(0% 0% 100% 0%)' }, { opacity: 1, clipPath: 'inset(0% 0% 0% 0%)', duration: 0.5, ease: 'power3.out' }, 2.55);
tl.to('#${p}-audit', { scaleY: 0.12, opacity: 0, y: 140, duration: 0.4, ease: 'power3.in' }, 3.65);
tl.fromTo('#${p}-rec', { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: 0.35, ease: 'expo.out' }, 3.95);
revealWords(tl, '.${p}-w', 2.75, 0.22);
`,
};
