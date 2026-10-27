// 0:07–0:15 — The complex payment. Notification, remittance, push into the screen, three fragmented sources.
import { C, appShell, dashContent, worldLayer, screenCSS, CAM, BACK_POSE, remitPDF, words, cursorSVG } from "../shared.mjs";

const p = "s2";
const toast = `
<div id="${p}-toast" class="a-card" style="position:absolute;left:680px;top:58px;width:372px;padding:14px 18px;display:flex;gap:14px;align-items:center;box-shadow:0 2px 4px rgba(70,50,30,.08),0 18px 40px rgba(70,50,30,.16)">
  <div style="width:44px;height:44px;border-radius:12px;background:${C.bronzeL};display:flex;align-items:center;justify-content:center;font-weight:700;color:${C.bronzeD};font-size:22px">$</div>
  <div><div style="font-size:13px;color:${C.ink2}">Bank deposit received · ACH</div><div class="num" style="font-size:28px;font-weight:700;letter-spacing:-.02em;color:${C.ink}">+$48,612.40</div><div style="font-size:13px;color:${C.ink}">Helix Health Distribution</div></div>
</div>`;
const remitCard = `
<div id="${p}-rcard" class="a-card" style="position:absolute;left:680px;top:178px;width:372px;height:150px;padding:16px 18px;display:flex;gap:14px;box-shadow:0 2px 4px rgba(70,50,30,.08),0 18px 40px rgba(70,50,30,.16)">
  <div style="width:62px;height:80px;border-radius:6px;background:#FFFEFB;border:1px solid ${C.line};position:relative;flex:none"><div style="position:absolute;left:8px;right:8px;top:12px;height:5px;background:#E6DFD3;border-radius:2px"></div><div style="position:absolute;left:8px;right:16px;top:24px;height:4px;background:#EEE9E1;border-radius:2px"></div><div style="position:absolute;left:8px;right:12px;top:34px;height:4px;background:#EEE9E1;border-radius:2px"></div><div style="position:absolute;left:8px;right:18px;top:44px;height:4px;background:#EEE9E1;border-radius:2px"></div><div style="position:absolute;left:6px;bottom:6px;font-size:9px;font-weight:700;color:#fff;background:${C.red};padding:1px 4px;border-radius:3px">PDF</div></div>
  <div><div style="font-size:13px;color:${C.ink2}">Remittance from ap@helixhealth.com</div><div style="font-size:19px;font-weight:700;margin:2px 0 8px">Remittance #HX-778120</div><div style="font-size:13px;line-height:1.55;color:${C.ink}">64 payment lines<br>3 related customer accounts</div></div>
</div>`;

const src = (id, label, inner, style) => `<div id="${p}-${id}" class="abs" style="${style}"><div class="label" style="margin-bottom:12px">${label}</div>${inner}</div>`;
const bank = src("bank", "Bank deposit", `
<div class="card" style="position:relative;width:460px;padding:26px 28px">
  <div style="display:flex;justify-content:space-between;align-items:center"><span class="chip chip-grey" style="font-size:16px">First Harbor Bank ••4417</span><span style="font-size:18px;color:${C.ink2}">Oct 27 · ACH</span></div>
  <div class="num" style="font-size:56px;font-weight:700;letter-spacing:-.03em;margin-top:16px;color:${C.ink}">+$48,612.40</div>
  <div style="font-size:19px;color:${C.ink2};margin-top:4px">Ref: HELIX HLTH DIST PMT</div>
</div>`, "left:1290px;top:150px");
const erpInv = (id, acct, amt, y) => `<div class="card" style="position:absolute;left:0;top:${y}px;width:460px;padding:18px 24px;display:flex;justify-content:space-between;align-items:center"><div><div style="font-size:24px;font-weight:700">${id}</div><div style="font-size:17px;color:${C.ink2}">Open · ${acct}</div></div><div class="num" style="font-size:26px;font-weight:700">${amt}</div></div>`;
const erp = src("erp", "ERP · open invoices", `<div style="position:relative;width:460px;height:330px">
  ${erpInv("INV-55141", "HX-1001", "$12,480.00", 0)}${erpInv("INV-55188", "HX-1002", "$4,250.00", 104)}${erpInv("INV-55203", "HX-1003", "$1,000.00", 208)}
  <div style="position:absolute;top:304px;left:24px;font-size:18px;color:${C.ink2}">+ 56 more open invoices</div></div>`, "left:1290px;top:520px");

export default {
  id: p,
  dur: 8,
  css: `
${screenCSS(p)}
#${p}-row{opacity:0}
#${p}-pw{background:${C.ivory};opacity:0}
#${p}-pdf{left:700px;top:150px;width:480px;height:800px;transform-origin:0 0}
#${p}-t1{left:110px;top:150px}
#${p}-t2{left:110px;top:240px}
#${p}-t3{left:110px;top:360px;width:520px}
#${p}-ws{left:300px;top:250px;width:1320px;height:640px;opacity:0;background:#FFFFFF;border-radius:28px;border:1px solid ${C.line};box-shadow:0 30px 80px rgba(70,50,30,.10)}
`,
  html: `
${worldLayer(p, appShell("Dashboard", dashContent(p).replace(`id="${p}-newrow"`, `id="${p}-row"`) + toast + remitCard))}
<div id="${p}-pw" class="full">
  <div id="${p}-ws" class="abs">
    <div style="position:absolute;left:36px;top:28px;display:flex;align-items:center;gap:14px;font-size:24px;font-weight:700">
      <svg width="30" height="30" viewBox="0 0 40 40" fill="none"><circle cx="18" cy="21" r="12" stroke="${C.bronze}" stroke-width="3.4"/><path d="M30 9 V33 M8 27 H30" stroke="${C.bronze}" stroke-width="3.4" stroke-linecap="round"/></svg>
      Cash application workspace <span style="font-weight:400;color:${C.ink2}">· Helix Health Distribution</span></div>
  </div>
  ${bank}
  ${erp}
  <div id="${p}-pdf" class="abs">${remitPDF(p)}</div>
  <div id="${p}-t1" class="t-hero">${words("One payment.", p + "-w1")}</div>
  <div id="${p}-t2" class="t-hero" style="color:${C.bronze}">${words("64 lines.", p + "-w2")}</div>
  <div id="${p}-t3" class="t-sub">${words("More complexity than meets the eye.", p + "-w3")}</div>
</div>
${cursorSVG(p + "-cur")}
`,
  js: `
${BACK_POSE(p)}
${CAM(p)}
drive(tl, 8, (t) => {
  seated(t, { head: 4 + 1.2 * Math.sin(t * 0.9) + key(t, [[1.0, 0], [1.3, 2.5], [2.2, 2.5], [2.6, 0]]) });
  const s = key(t, [[0, 3.25], [1.05, 6.0], [3.15, 6.0], [3.7, 11]], (u) => eio(u));
  const cx = key(t, [[0, 1234], [1.05, 1291], [3.15, 1291], [3.7, 1333]]);
  const cy = key(t, [[0, 538], [1.05, 520], [3.15, 520], [3.7, 500]]);
  cam(s, cx, cy);
});
// payment lands on the dashboard
tl.fromTo('#${p}-toast', { opacity: 0, x: 60 }, { opacity: 1, x: 0, duration: 0.6, ease: 'expo.out' }, 1.0);
tl.fromTo('#${p}-row', { opacity: 0, x: 120 }, { opacity: 1, x: 0, duration: 0.6, ease: 'expo.out' }, 1.25);
tl.fromTo('#${p}-rcard', { opacity: 0, x: 80 }, { opacity: 1, x: 0, duration: 0.6, ease: 'expo.out' }, 1.85);
// Maya opens the remittance
tl.fromTo('#${p}-cur', { x: 1500, y: 1100, opacity: 1 }, { x: 1370, y: 380, duration: 0.7, ease: 'power3.out' }, 2.35);
tl.to('#${p}-cur', { scale: 0.86, duration: 0.07, yoyo: true, repeat: 1, transformOrigin: '10px 6px' }, 3.05);
tl.to('#${p}-cur', { opacity: 0, duration: 0.15 }, 3.3);
// the card becomes the document (screen-space handoff)
tl.fromTo('#${p}-pw', { opacity: 0 }, { opacity: 1, duration: 0.3, ease: 'power2.inOut' }, 3.35);
tl.fromTo('#${p}-world', { opacity: 1 }, { opacity: 0, duration: 0.05 }, 3.66);
tl.fromTo('#${p}-pdf', { x: 1210 - 700, y: 309 - 150, scaleX: 0.8, scaleY: 0.22 }, { x: 0, y: 0, scaleX: 1, scaleY: 1, duration: 0.75, ease: 'expo.out' }, 3.35);
tl.fromTo('.${p}-prow, .${p}-pbar', { opacity: 0 }, { opacity: 1, duration: 0.3, stagger: 0.025 }, 3.7);
// three sources, fragmented
tl.fromTo('#${p}-bank', { opacity: 0, x: 80, rotation: 0 }, { opacity: 1, x: 0, rotation: -1.6, duration: 0.8, ease: 'expo.out' }, 4.15);
tl.fromTo('#${p}-erp', { opacity: 0, x: 80, rotation: 0 }, { opacity: 1, x: 0, rotation: 1.4, duration: 0.8, ease: 'expo.out' }, 4.4);
tl.to('#${p}-pdf', { rotation: -1.2, duration: 0.8, ease: 'power2.out' }, 4.2);
// titles
revealWords(tl, '.${p}-w1', 3.95);
revealWords(tl, '.${p}-w2', 4.6);
revealWords(tl, '.${p}-w3', 5.25, 0.06);
hideWords(tl, '.${p}-w1, .${p}-w2, .${p}-w3', 6.62, 0.02);
// everything gathers into one workspace
tl.fromTo('#${p}-ws', { opacity: 0, scale: 0.96 }, { opacity: 1, scale: 1, duration: 0.6, ease: 'power3.out' }, 6.65);
tl.to('#${p}-bank', { x: 360 - 1290, y: 400 - 150, scale: 0.72, rotation: 0, duration: 0.75, ease: 'power3.inOut' }, 6.7);
tl.to('#${p}-erp', { x: 1210 - 1290, y: 360 - 520, scale: 0.72, rotation: 0, duration: 0.75, ease: 'power3.inOut' }, 6.75);
tl.to('#${p}-pdf', { x: 790 - 700, y: 330 - 150, scale: 0.62, rotation: 0, duration: 0.75, ease: 'power3.inOut' }, 6.72);
// follow the remittance
tl.to('#${p}-ws, #${p}-bank, #${p}-erp', { opacity: 0, filter: 'blur(10px)', duration: 0.3, ease: 'power2.in' }, 7.55);
tl.to('#${p}-pdf', { x: 0, y: 0, scale: 1, duration: 0.45, ease: 'power3.inOut' }, 7.55);
`,
};
