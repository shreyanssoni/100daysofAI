// 0:23–0:30 — Connecting the accounts, Maya's recognition, bank match.
import { C, appShell, worldLayer, screenCSS, CAM, BACK_POSE, mayaFace, words } from "../shared.mjs";

const p = "s4";
const ACCTS = [
  ["HX-1001", "Helix Health — East", "24 invoices"],
  ["HX-1002", "Helix Health — Central", "21 invoices"],
  ["HX-1003", "Helix Pharmacy Services", "14 invoices"],
];
const CX0 = [460, 960, 1460], CX1 = [590, 960, 1330];
const card = (a, i) => `
<div id="${p}-c${i}" class="card" style="left:${CX0[i] - 170}px;top:520px;width:340px;height:200px;padding:26px 28px">
  <div style="font-size:40px;font-weight:700;letter-spacing:-.02em">${a[0]}</div>
  <div style="font-size:21px;color:${C.ink2};margin-top:6px">${a[1]}</div>
  <div style="margin-top:18px"><span class="chip chip-grey" style="font-size:17px">${a[2]} on this remittance</span></div>
</div>`;
const screenApp = appShell("Customers", `
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px"><div class="a-title">Helix Health Distribution</div><span class="a-pill" style="background:${C.bronzeL};color:${C.bronzeD};font-size:14px">Pays for 3 related accounts</span></div>
  <div class="a-sub" style="margin-bottom:22px">Remittance HX-778120 · $48,612.40 · 64 lines</div>
  <svg width="1040" height="80" style="display:block"><path d="M520 0 V30 M180 30 H860 M180 30 V80 M520 30 V80 M860 30 V80" stroke="${C.bronze}" stroke-width="2.5" fill="none"/></svg>
  <div style="display:flex;gap:22px">${ACCTS.map((a) => `<div class="a-card" style="flex:1;padding:18px 20px"><div style="font-size:24px;font-weight:700">${a[0]}</div><div style="font-size:14px;color:${C.ink2};margin:4px 0 12px">${a[1]}</div><span class="a-pill" style="background:${C.grey};color:${C.ink2}">${a[2]}</span></div>`).join("")}</div>
  <div class="a-card" style="margin-top:22px;padding:16px 20px;display:flex;justify-content:space-between;font-size:15px"><span>Bank deposit <b class="num">$48,612.40</b></span><span>Remittance total <b class="num">$48,612.40</b></span><span class="a-pill" style="background:${C.blueL};color:${C.blueD}">Exact match</span></div>
`);

export default {
  id: p,
  dur: 7,
  css: `
${screenCSS(p)}
#${p}-map{transform-origin:0 0;background:${C.ivory}}
#${p}-world{opacity:0}
#${p}-face{opacity:0}
#${p}-match{opacity:0;background:${C.ivory}}
#${p}-t1{left:110px;top:90px}
#${p}-t2{left:110px;top:180px}
.${p}-chip{position:absolute;top:575px;transform:translate(-50%,-50%)}
`,
  html: `
${worldLayer(p, screenApp)}
<div id="${p}-map" class="full">
  <svg class="full" viewBox="0 0 1920 1080" style="position:absolute;left:0;top:0">
    <path id="${p}-ln1" d="M${CX0[0] + 170} 620 L${CX0[1] - 170} 620" stroke="${C.bronze}" stroke-width="3" fill="none" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/>
    <path id="${p}-ln2" d="M${CX0[1] + 170} 620 L${CX0[2] - 170} 620" stroke="${C.bronze}" stroke-width="3" fill="none" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/>
    <path id="${p}-tree" d="M960 430 V475 M${CX1[0]} 475 H${CX1[2]} M${CX1[0]} 475 V520 M960 475 V520 M${CX1[2]} 475 V520" stroke="${C.bronze}" stroke-width="3" fill="none" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/>
  </svg>
  <div id="${p}-why" class="abs" style="left:960px;top:566px;transform:translateX(-50%);opacity:0"><span class="chip chip-blue" style="font-size:18px">Same parent · same remit-to address</span></div>
  ${ACCTS.map(card).join("")}
  <div id="${p}-parent" class="card" style="left:560px;top:270px;width:800px;height:160px;padding:30px 36px;opacity:0;display:flex;align-items:center;justify-content:space-between;border-color:${C.bronze}">
    <div><div class="label" style="color:${C.bronzeD}">Parent customer</div><div style="font-size:44px;font-weight:700;letter-spacing:-.025em;margin-top:6px">Helix Health Distribution</div></div>
  </div>
  <div id="${p}-pays" class="abs" style="left:1210px;top:322px;opacity:0"><span class="chip chip-bronze" style="font-size:20px">Pays for 3 related accounts</span></div>
  ${ACCTS.map((a, i) => `<span id="${p}-ch${i}" class="chip chip-white ${p}-chip" style="left:${CX0[i]}px;font-size:40px;padding:14px 30px">${a[0]}</span>`).join("")}
  <div id="${p}-t1" class="t-hero">${words("Different accounts.", p + "-w1")}</div>
  <div id="${p}-t2" class="t-hero">${words("One", p + "-w2")} <span class="w ${p}-w2 serif">clear</span> ${words("picture.", p + "-w2")}</div>
</div>
<div id="${p}-face" class="full">${mayaFace(p)}</div>
<div id="${p}-match" class="full">
  <div id="${p}-mbank" class="card" style="left:300px;top:400px;width:600px;padding:32px 36px">
    <span class="chip chip-grey" style="font-size:18px">Bank deposit · First Harbor ••4417</span>
    <div class="num" style="font-size:72px;font-weight:700;letter-spacing:-.03em;margin-top:14px">$48,612.40</div>
  </div>
  <div id="${p}-eq" class="abs" style="left:930px;top:470px;font-size:80px;font-weight:700;color:${C.bronze}">=</div>
  <div id="${p}-mrem" class="card" style="left:1020px;top:400px;width:600px;padding:32px 36px">
    <span class="chip chip-blue" style="font-size:18px">Remittance HX-778120 · total</span>
    <div class="num" style="font-size:72px;font-weight:700;letter-spacing:-.03em;margin-top:14px">$48,612.40</div>
  </div>
  <div id="${p}-ok" class="abs" style="left:960px;top:680px;transform:translateX(-50%);display:flex;gap:14px;align-items:center;white-space:nowrap">
    <span class="chip chip-bronze" style="font-size:24px;padding:10px 20px"><svg width="22" height="22" viewBox="0 0 22 22"><path d="M4 11.5 L9 16 L18 6" stroke="${C.bronzeD}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>Exact match</span>
    <span style="font-size:22px;color:${C.ink2}">Bank ref HELIX HLTH DIST PMT ↔ HX-778120</span>
  </div>
</div>
`,
  js: `
${BACK_POSE(p)}
${CAM(p)}
drive(tl, 7, (t) => {
  seated(t, { head: 4 + key(t, [[3.9, 0], [4.15, 5], [4.4, 1], [4.6, 3]]) });
  cam(3.25, 1234, 538);
});
// chips become account cards
[0, 1, 2].forEach((i) => {
  tl.fromTo('#${p}-c' + i, { opacity: 0, scale: 0.5 }, { opacity: 1, scale: 1, duration: 0.6, ease: 'expo.out' }, 0.15 + i * 0.06);
  tl.fromTo('#${p}-ch' + i, { opacity: 1, y: 0 }, { opacity: 0, y: 20, duration: 0.3, ease: 'power2.in' }, 0.15 + i * 0.06);
});
// the AI sees the relationship
tl.to('#${p}-ln1', { strokeDashoffset: 0, duration: 0.45, ease: 'power2.inOut' }, 0.95);
tl.to('#${p}-ln2', { strokeDashoffset: 0, duration: 0.45, ease: 'power2.inOut' }, 1.1);
tl.fromTo('#${p}-why', { opacity: 0, y: 8 }, { opacity: 1, y: 0, duration: 0.4, ease: 'expo.out' }, 1.3);
// they group under one parent
tl.to('#${p}-ln1, #${p}-ln2, #${p}-why', { opacity: 0, duration: 0.25 }, 1.95);
[0, 1, 2].forEach((i) => tl.to('#${p}-c' + i, { x: ${CX1[0] - CX0[0]} * (i === 0) + ${CX1[2] - CX0[2]} * (i === 2), duration: 0.6, ease: 'power3.inOut' }, 2.0));
tl.fromTo('#${p}-parent', { opacity: 0, scaleX: 0.3, scaleY: 0.6 }, { opacity: 1, scaleX: 1, scaleY: 1, duration: 0.6, ease: 'expo.out' }, 2.25);
tl.to('#${p}-tree', { strokeDashoffset: 0, duration: 0.5, ease: 'power2.out' }, 2.45);
tl.fromTo('#${p}-pays', { opacity: 0, x: -10 }, { opacity: 1, x: 0, duration: 0.4, ease: 'expo.out' }, 2.65);
revealWords(tl, '.${p}-w1', 0.5);
revealWords(tl, '.${p}-w2', 2.35, 0.1);
hideWords(tl, '.${p}-w1, .${p}-w2', 3.15, 0.02);
// pull back: the picture is on Maya's monitor
tl.fromTo('#${p}-world', { opacity: 0 }, { opacity: 1, duration: 0.3 }, 3.35);
tl.fromTo('#${p}-map', { scale: 1, x: 0, y: 0 }, { scale: 721.5 / 1920, x: 784.5, y: 260.5, duration: 0.6, ease: 'power3.inOut' }, 3.3);
tl.to('#${p}-map', { opacity: 0, duration: 0.25 }, 3.85);
// recognition, close
tl.set('#${p}-face', { opacity: 1 }, 4.55);
tl.set('#${p}-face', { opacity: 0 }, 5.55);
tl.fromTo('#${p}-browN', { y: 0 }, { y: -7, duration: 0.25, ease: 'power2.out' }, 4.8);
tl.fromTo('#${p}-browF', { y: 0 }, { y: -5, duration: 0.25, ease: 'power2.out' }, 4.82);
tl.fromTo('#${p}-mouth', { attr: { d: 'M852 513 Q866 517 879 512' } }, { attr: { d: 'M850 511 Q866 523 881 509' }, duration: 0.3, ease: 'power2.out' }, 4.95);
tl.fromTo('#${p}-headg', { rotation: 0 }, { rotation: 2.5, svgOrigin: '780 720', duration: 0.22, ease: 'power2.out', yoyo: true, repeat: 1 }, 5.05);
tl.fromTo('#${p}-irisN', { x: 0 }, { x: 3, duration: 0.3 }, 4.6);
// bank match
tl.set('#${p}-match', { opacity: 1 }, 5.55);
tl.to('#${p}-world', { opacity: 0, duration: 0.05 }, 4.56);

tl.fromTo('#${p}-mbank', { x: -260, opacity: 0 }, { x: 0, opacity: 1, duration: 0.6, ease: 'expo.out' }, 5.55);
tl.fromTo('#${p}-mrem', { x: 260, opacity: 0 }, { x: 0, opacity: 1, duration: 0.6, ease: 'expo.out' }, 5.6);
tl.fromTo('#${p}-eq', { opacity: 0, scale: 0.6 }, { opacity: 1, scale: 1, duration: 0.35, ease: 'expo.out' }, 6.05);
tl.fromTo('#${p}-ok', { opacity: 0, y: 12 }, { opacity: 1, y: 0, duration: 0.4, ease: 'expo.out' }, 6.25);
`,
};
