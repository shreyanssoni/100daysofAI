// 0:44–0:51 — The human moment. Two decisions, then Maya leans back.
import { C, appShell, worldLayer, screenCSS, CAM, BACK_POSE, mayaFace, words, cursorSVG } from "../shared.mjs";

const p = "s6";
const btn = (id, txt, cls, x, y, w) => `<div id="${id}" class="btn ${cls}" style="position:absolute;left:${x}px;top:${y}px;width:${w}px;height:44px;font-size:16px">${txt}</div>`;
const review = `
<div style="display:flex;align-items:center;gap:12px"><div class="a-title">Needs review</div><span id="${p}-badge" style="min-width:34px;height:34px;border-radius:17px;background:${C.amber};color:#fff;font-weight:700;font-size:18px;display:inline-flex;align-items:center;justify-content:center;padding:0 10px">2</span><span id="${p}-clear" class="a-pill" style="background:#EEF2EA;color:#4F6B45;font-size:14px;opacity:0">All decisions made</span></div>
<div class="a-sub">Remittance HX-778120 · Helix Health Distribution · 62 of 64 lines resolved automatically</div>
<div id="${p}-k1" class="a-card" style="position:absolute;left:30px;top:110px;width:640px;height:200px;padding:20px 24px;transform-origin:0 0">
  <span class="a-pill" style="background:${C.redL};color:${C.redD}">Chargebacks</span>
  <div style="font-size:24px;font-weight:700;margin-top:10px">Create $0 invoice for chargebacks</div>
  <div style="font-size:15px;color:${C.ink2};margin-top:6px">Offsets CB-40112 + CB-40113 · −$515.75 · Rule: chargeback handling</div>
  ${btn(p + "-approve", "Approve", "btn-bronze", 24, 132, 130)}${btn("", "Edit", "btn-ghost", 166, 132, 100)}
</div>
<div id="${p}-k2" class="a-card" style="position:absolute;left:30px;top:330px;width:640px;height:200px;padding:20px 24px">
  <span class="a-pill" style="background:${C.amberL};color:${C.amberD}">Short payment</span>
  <div style="font-size:24px;font-weight:700;margin-top:10px">INV-55203 — $118.40 remaining</div>
  <div style="font-size:15px;color:${C.ink2};margin-top:6px">Keep invoice open? Paid $881.60 of $1,000.00 · HX-1003</div>
  <div id="${p}-k2b">${btn(p + "-keep", "Keep open", "btn-bronze", 24, 132, 130)}${btn("", "Write off", "btn-ghost", 166, 132, 110)}</div>
  <span id="${p}-kept" class="a-pill" style="position:absolute;left:24px;top:142px;background:${C.amberL};color:${C.amberD};font-size:15px;opacity:0;border:1.5px solid ${C.amber}">Kept open · $118.40 tracked on INV-55203</span>
</div>
<div class="a-card" style="position:absolute;left:700px;top:110px;width:330px;height:420px;padding:18px 20px">
  <div style="font-size:15px;font-weight:700;margin-bottom:12px">ERP posting queue</div>
  ${[["59 invoice applications", "$48,275.40"], ["Discounts taken", "$412.60"], ["Chargeback recovery", "$212.75"], ["Unapplied credit", "$640.00"]].map((r) => `<div style="display:flex;justify-content:space-between;font-size:13.5px;padding:9px 0;border-top:1px solid #F0EAE0"><span>${r[0]}</span><b class="num">${r[1]}</b></div>`).join("")}
  <div id="${p}-q0" style="display:flex;justify-content:space-between;font-size:13.5px;padding:9px 0;border-top:1px solid #F0EAE0;opacity:0"><span>$0 invoice · chargebacks</span><b class="num">−$515.75</b></div>
</div>`;

export default {
  id: p,
  dur: 7,
  css: `
${screenCSS(p)}
#${p}-face{opacity:0}
#${p}-t1{left:110px;top:900px;font-size:64px}
#${p}-t2{left:110px;top:96px}
`,
  html: `
${worldLayer(p, appShell("Exceptions", review))}
<div id="${p}-t1" class="t-hero">${words("AI handles the complexity.", p + "-w1")}</div>
<div id="${p}-face" class="full">${mayaFace(p)}
  <div id="${p}-t2" class="t-hero">${words("You stay in", p + "-w2")} <span class="w ${p}-w2 serif">control.</span></div>
</div>
${cursorSVG(p + "-cur")}
`,
  js: `
${BACK_POSE(p)}
${CAM(p)}
drive(tl, 7, (t) => {
  seated(t, { head: 4 + 1.2 * Math.sin(t * 0.9) });
  const s = key(t, [[0, 3.25], [0.7, 3.25], [1.5, 6.0]]);
  cam(s, key(t, [[0, 1234], [0.7, 1234], [1.5, 1291]]), key(t, [[0, 538], [0.7, 538], [1.5, 535]]));
});
// decision 1: approve the $0 invoice
tl.fromTo('#${p}-cur', { x: 1100, y: 1000, opacity: 0 }, { x: 604, y: 302, opacity: 1, duration: 0.6, ease: 'power3.out' }, 1.65);
tl.to('#${p}-cur', { scale: 0.86, transformOrigin: '10px 6px', duration: 0.07, yoyo: true, repeat: 1 }, 2.3);
tl.fromTo('#${p}-approve', { backgroundColor: '${C.bronze}' }, { backgroundColor: '${C.bronzeD}', duration: 0.08, yoyo: true, repeat: 1 }, 2.3);
tl.to('#${p}-k1', { scale: 0.32, x: 690, y: 330, opacity: 0, duration: 0.6, ease: 'power3.inOut' }, 2.5);
tl.fromTo('#${p}-q0', { opacity: 0, x: -12 }, { opacity: 1, x: 0, duration: 0.35, ease: 'expo.out' }, 2.95);
tl.to('#${p}-k2', { y: -220, duration: 0.5, ease: 'power3.inOut' }, 2.9);
tl.fromTo('#${p}-badge', { scale: 1 }, { scale: 1.2, duration: 0.12, yoyo: true, repeat: 1, onStart: () => { $('#${p}-badge').textContent = '1'; }, onReverseComplete: () => { $('#${p}-badge').textContent = '2'; } }, 2.7);
// decision 2: keep INV-55203 open
tl.to('#${p}-cur', { x: 604, y: 311, duration: 0.5, ease: 'power3.inOut' }, 3.2);
tl.to('#${p}-cur', { scale: 0.86, transformOrigin: '10px 6px', duration: 0.07, yoyo: true, repeat: 1 }, 3.75);
tl.fromTo('#${p}-keep', { backgroundColor: '${C.bronze}' }, { backgroundColor: '${C.bronzeD}', duration: 0.08, yoyo: true, repeat: 1 }, 3.75);
tl.to('#${p}-k2b', { opacity: 0, duration: 0.15 }, 3.9);
tl.fromTo('#${p}-kept', { opacity: 0, x: -8 }, { opacity: 1, x: 0, duration: 0.35, ease: 'expo.out' }, 3.95);
tl.fromTo('#${p}-badge', { opacity: 1 }, { opacity: 1, duration: 0.01, onStart: () => { const b = $('#${p}-badge'); b.textContent = '0'; b.style.background = '${C.bronze}'; }, onReverseComplete: () => { const b = $('#${p}-badge'); b.textContent = '1'; b.style.background = '${C.amber}'; } }, 4.0);
tl.fromTo('#${p}-clear', { opacity: 0 }, { opacity: 1, duration: 0.3 }, 4.1);
tl.to('#${p}-cur', { opacity: 0, duration: 0.2 }, 4.3);
revealWords(tl, '.${p}-w1', 1.6, 0.09);
hideWords(tl, '.${p}-w1', 4.4, 0.03);
// Maya leans back
tl.set('#${p}-face', { opacity: 1 }, 4.6);
tl.fromTo('#${p}-world', { opacity: 1 }, { opacity: 0, duration: 0.05 }, 4.62);
tl.fromTo('#${p}-fig', { rotation: 0, x: 0 }, { rotation: -3.5, x: -30, svgOrigin: '780 1080', duration: 0.9, ease: 'power2.inOut' }, 4.7);
tl.fromTo('#${p}-mouth', { attr: { d: 'M872 538 C884 534 898 533 910 536 C900 546 884 548 872 538 Z' } }, { attr: { d: 'M869 533 C884 537 898 535 913 527 C903 547 882 551 869 533 Z' }, duration: 0.4, ease: 'power2.out' }, 5.0);
tl.fromTo('#${p}-mouthL', { attr: { d: 'M872 538 C886 541 898 540 910 536' } }, { attr: { d: 'M869 533 C886 541 900 538 913 527' }, duration: 0.4, ease: 'power2.out' }, 5.0);
tl.fromTo('#${p}-eyeN', { scaleY: 1 }, { scaleY: 0.1, transformOrigin: '50% 50%', duration: 0.07, yoyo: true, repeat: 1 }, 5.75);
tl.fromTo('#${p}-eyeF', { scaleY: 1 }, { scaleY: 0.1, transformOrigin: '50% 50%', duration: 0.07, yoyo: true, repeat: 1 }, 5.75);
revealWords(tl, '.${p}-w2', 5.0, 0.1);
`,
};
