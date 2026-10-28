// 0:07–0:15 — The complex payment. Jake's question; the deposit arrives from the bank, the remittance by
// email; Maya opens it; the three sources sit fragmented, then gather into one workspace.
import { C, appShell, dashContent, worldLayer, screenCSS, CAM, BACK_POSE, remitPDF, words, cursorSVG, icon } from "../shared.mjs";

const p = "s2";
// ---- the app: dashboard (left) and a live inbox column (right) ----
// inbox column in app coordinates: x 872..1250 (app-main left 192 + padding 30 + 650)
const shadow = "box-shadow:0 2px 4px rgba(70,50,30,.08),0 14px 30px rgba(70,50,30,.12)";
const inbox = `
<div style="position:absolute;left:680px;top:58px;width:378px;font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:${C.ink2};display:flex;justify-content:space-between"><span>Inbox</span><span style="color:${C.bronze}">● live</span></div>
<div id="${p}-dev" class="a-card" style="position:absolute;left:680px;top:84px;width:378px;height:124px;padding:14px 16px;display:flex;gap:12px;${shadow}">
  <div style="width:38px;height:38px;border-radius:50%;background:${C.blueL};color:${C.blueD};font-weight:700;font-size:14px;display:flex;align-items:center;justify-content:center;flex:none">JM</div>
  <div><div style="font-size:12.5px;color:${C.ink2}"><b style="color:${C.ink}">Jake Miller</b> · Sales · now</div><div style="font-size:17px;line-height:1.35;margin-top:3px;color:${C.ink}">Helix says they <b>paid in full</b>. Can you confirm before my 10:00 call?</div></div>
</div>
<div id="${p}-toast" class="a-card" style="position:absolute;left:680px;top:220px;width:378px;height:112px;padding:14px 16px;display:flex;gap:12px;align-items:center;${shadow}">
  <div style="width:42px;height:42px;border-radius:12px;background:${C.bronzeL};color:${C.bronzeD};display:flex;align-items:center;justify-content:center;flex:none">${icon.bank(22)}</div>
  <div><div style="font-size:12.5px;color:${C.ink2}">Bank deposit · ACH · First Harbor ••4417</div><div class="num" style="font-size:28px;font-weight:700;letter-spacing:-.02em;color:${C.ink};line-height:1.15">+$48,612.40</div><div style="font-size:12.5px;color:${C.ink}">Helix Health Distribution</div></div>
</div>
<div id="${p}-rcard" class="a-card" style="position:absolute;left:680px;top:346px;width:378px;height:146px;padding:14px 16px;display:flex;gap:14px;${shadow}">
  <div style="width:58px;height:76px;border-radius:6px;background:#FFFEFB;border:1px solid ${C.line};position:relative;flex:none"><div style="position:absolute;left:8px;right:8px;top:12px;height:5px;background:#E6DFD3;border-radius:2px"></div><div style="position:absolute;left:8px;right:16px;top:24px;height:4px;background:#EEE9E1;border-radius:2px"></div><div style="position:absolute;left:8px;right:12px;top:34px;height:4px;background:#EEE9E1;border-radius:2px"></div><div style="position:absolute;left:6px;bottom:6px;font-size:9px;font-weight:700;color:#fff;background:${C.red};padding:1px 4px;border-radius:3px;line-height:1.2">PDF</div></div>
  <div><div style="font-size:12.5px;color:${C.ink2};display:flex;align-items:center;gap:6px">${icon.mail(14)} ap@helixhealth.com</div><div style="font-size:18px;font-weight:700;margin:3px 0 6px">Remittance #HX-778120</div><div style="font-size:13px;line-height:1.5;color:${C.ink}">64 payment lines<br>3 related customer accounts</div></div>
</div>`;
const dash = `<div style="width:640px">${dashContent(p).replace(`id="${p}-newrow"`, `id="${p}-row"`)}</div>${inbox}`;

// ---- screen-space "incoming" layer: bank transfer from the left, email from the right ----
// app → screen at the push-in camera: sx = 294 + 1.0406·ax, sy = 132 + 1.0406·ay
const SX = (ax) => 294 + 1.0406 * ax, SY = (ay) => 132 + 1.0406 * ay;
const bankDock = [SX(872), SY(276)], mailDock = [SX(1250), SY(419)];
const bankPath = `M268 236 C 560 40, 1000 60, ${bankDock[0]} ${bankDock[1]}`;
const mailPath = `M1676 640 C 1690 520, 1660 420, ${mailDock[0]} ${mailDock[1]}`;
const incoming = `
<div id="${p}-inc" class="full" style="pointer-events:none">
  <svg class="full" viewBox="0 0 1920 1080" style="position:absolute;left:0;top:0">
    <path id="${p}-bpath" d="${bankPath}" stroke="${C.bronze}" stroke-width="3.5" stroke-dasharray="2 12" stroke-linecap="round" fill="none" opacity="0"/>
    <path id="${p}-mpath" d="${mailPath}" stroke="${C.blue}" stroke-width="3.5" stroke-dasharray="2 12" stroke-linecap="round" fill="none" opacity="0"/>
  </svg>
  <div id="${p}-bsrc" class="card" style="left:36px;top:170px;width:232px;padding:16px 18px;opacity:0">
    <div style="display:flex;align-items:center;gap:10px;color:${C.bronzeD}">${icon.bank(30, C.bronzeD)}<span class="label" style="color:${C.bronzeD};font-size:15px">Bank</span></div>
    <div style="font-size:20px;font-weight:700;margin-top:8px">First Harbor Bank</div><div style="font-size:16px;color:${C.ink2}">ACH transfer in</div>
  </div>
  <div id="${p}-msrc" class="card" style="left:1652px;top:640px;width:236px;padding:16px 18px;opacity:0">
    <div style="position:relative;width:64px;height:46px;margin-bottom:8px">
      <div id="${p}-paper" style="position:absolute;left:10px;top:4px;width:44px;height:38px;border-radius:3px;background:#FFFEFB;border:1px solid ${C.line}"><div style="margin:7px 6px 0;height:3px;background:#E2DACD"></div><div style="margin:4px 6px 0;height:3px;background:#ECE6DC"></div><div style="margin:4px 6px 0;height:3px;background:#ECE6DC"></div></div>
      <div style="position:absolute;left:0;top:14px;width:64px;height:32px;border-radius:4px;background:${C.blueL};border:1.5px solid ${C.blue}"></div>
      <svg id="${p}-flap" width="64" height="24" viewBox="0 0 64 24" style="position:absolute;left:0;top:14px;transform-origin:50% 0"><path d="M1 1 L32 20 L63 1 Z" fill="#D2DEEB" stroke="${C.blue}" stroke-width="1.5" stroke-linejoin="round"/></svg>
    </div>
    <div style="font-size:20px;font-weight:700">Email</div><div style="font-size:16px;color:${C.ink2}">ap@helixhealth.com</div>
  </div>
  <div id="${p}-coin" class="abs" style="left:0;top:0;opacity:0"><span class="chip chip-bronze" style="font-size:20px;box-shadow:0 8px 20px rgba(70,50,30,.18);transform:translate(-50%,-50%);border:1.5px solid ${C.bronze}">${icon.bank(18, C.bronzeD)} +$48,612.40</span></div>
  <div id="${p}-doc" class="abs" style="left:0;top:0;opacity:0"><div style="transform:translate(-50%,-50%);width:46px;height:58px;border-radius:4px;background:#FFFEFB;border:1.5px solid ${C.blue};box-shadow:0 8px 20px rgba(70,50,30,.18);position:relative"><div style="margin:9px 7px 0;height:3px;background:#E2DACD"></div><div style="margin:4px 7px 0;height:3px;background:#ECE6DC"></div><div style="margin:4px 7px 0;height:3px;background:#ECE6DC"></div><div style="position:absolute;left:5px;bottom:5px;font:700 8px Inter;color:#fff;background:${C.red};padding:1px 3px;border-radius:2px">PDF</div></div></div>
</div>`;

// ---- the fragmented sources ----
const src = (id, ic, label, inner, style) => `<div id="${p}-${id}" class="abs" style="${style}"><div class="label" style="margin-bottom:12px;display:flex;align-items:center;gap:8px">${ic}${label}</div>${inner}</div>`;
const bank = src("bank", icon.bank(20), "Bank deposit", `
<div class="card" style="position:relative;width:460px;padding:26px 28px">
  <div style="display:flex;justify-content:space-between;align-items:center"><span class="chip chip-grey" style="font-size:16px">First Harbor Bank ••4417</span><span style="font-size:18px;color:${C.ink2}">Oct 27 · ACH</span></div>
  <div class="num" style="font-size:56px;font-weight:700;letter-spacing:-.03em;margin-top:16px;color:${C.ink};line-height:1.1">+$48,612.40</div>
  <div style="font-size:19px;color:${C.ink2};margin-top:6px">Ref: HELIX HLTH DIST PMT</div>
</div>`, "left:1290px;top:150px;transform-origin:0 0");
const erpInv = (id, acct, amt, y) => `<div class="card" style="position:absolute;left:0;top:${y}px;width:460px;padding:18px 24px;display:flex;justify-content:space-between;align-items:center"><div><div style="font-size:24px;font-weight:700">${id}</div><div style="font-size:17px;color:${C.ink2}">Open · ${acct}</div></div><div class="num" style="font-size:26px;font-weight:700">${amt}</div></div>`;
const erp = src("erp", icon.erp(20), "ERP · open invoices", `<div style="position:relative;width:460px;height:330px">
  ${erpInv("INV-55141", "HX-1001", "$12,480.00", 0)}${erpInv("INV-55188", "HX-1002", "$4,250.00", 104)}${erpInv("INV-55203", "HX-1003", "$1,000.00", 208)}
  <div style="position:absolute;top:304px;left:24px;font-size:18px;color:${C.ink2}">+ 56 more open invoices</div></div>`, "left:1290px;top:520px;transform-origin:0 0");

export default {
  id: p,
  dur: 8,
  css: `
${screenCSS(p)}
#${p}-row{opacity:0}
#${p}-pw{background:var(--ground);opacity:0}
#${p}-pdf{left:700px;top:150px;width:480px;height:800px;transform-origin:0 0}
#${p}-pdfl{position:absolute;left:0;top:-40px;display:flex;align-items:center;gap:8px}
#${p}-t1{left:110px;top:150px}
#${p}-t2{left:110px;top:240px}
#${p}-t3{left:110px;top:360px;width:520px}
#${p}-ws{left:300px;top:250px;width:1320px;height:640px;opacity:0;background:#FFFFFF;border-radius:28px;border:1px solid ${C.line};box-shadow:0 30px 80px rgba(70,50,30,.10)}
`,
  html: `
${worldLayer(p, appShell("Dashboard", dash))}
${incoming}
<div id="${p}-pw" class="full">
  <div id="${p}-ws" class="abs">
    <div style="position:absolute;left:36px;top:28px;display:flex;align-items:center;gap:14px;font-size:24px;font-weight:700;white-space:nowrap">Cash application workspace <span style="font-weight:400;color:${C.ink2}">· Helix Health Distribution</span></div>
  </div>
  ${bank}
  ${erp}
  <div id="${p}-pdf" class="abs"><div id="${p}-pdfl" class="label">${icon.mail(20)}Customer remittance · email</div>${remitPDF(p)}</div>
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
  seated(t, { head: 4 + 1.2 * Math.sin(t * 0.9) + key(t, [[0.5, 0], [0.8, 2.5], [1.6, 2.5], [2.0, 0]]) });
  const s = key(t, [[0, 3.25], [1.0, 6.0], [3.35, 6.0], [3.9, 11]], (u) => eio(u));
  const cx = key(t, [[0, 1234], [1.0, 1291], [3.35, 1291], [3.9, 1333]]);
  const cy = key(t, [[0, 538], [1.0, 520], [3.35, 520], [3.9, 500]]);
  cam(s, cx, cy);
});
// a token that travels along an SVG path
const travel = (sel, pathSel, t0, dur) => {
  const path = $(pathSel); const L = path.getTotalLength(); const o = { u: 0 };
  tl.fromTo(o, { u: 0 }, { u: 1, duration: dur, ease: 'power2.inOut', onUpdate: () => { const pt = path.getPointAtLength(o.u * L); const e = $(sel); e.style.transform = 'translate(' + pt.x.toFixed(1) + 'px,' + pt.y.toFixed(1) + 'px)'; } }, t0);
  const pt0 = path.getPointAtLength(0); $(sel).style.transform = 'translate(' + pt0.x + 'px,' + pt0.y + 'px)';
};
// Jake's question
tl.fromTo('#${p}-dev', { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.55, ease: 'expo.out' }, 0.35);
// the deposit comes in from the bank
tl.fromTo('#${p}-bsrc', { opacity: 0, x: -60 }, { opacity: 1, x: 0, duration: 0.5, ease: 'expo.out' }, 0.9);
tl.fromTo('#${p}-bpath', { opacity: 0, strokeDashoffset: 60 }, { opacity: 0.9, strokeDashoffset: 0, duration: 0.35 }, 1.05);
tl.fromTo('#${p}-coin', { opacity: 0 }, { opacity: 1, duration: 0.15 }, 1.15);
travel('#${p}-coin', '#${p}-bpath', 1.15, 0.65);
tl.to('#${p}-coin', { opacity: 0, duration: 0.12 }, 1.8);
tl.fromTo('#${p}-toast', { opacity: 0, scale: 0.94 }, { opacity: 1, scale: 1, duration: 0.45, ease: 'back.out(1.4)' }, 1.78);
tl.fromTo('#${p}-row', { opacity: 0, x: 100 }, { opacity: 1, x: 0, duration: 0.55, ease: 'expo.out' }, 1.92);
// the remittance arrives by email
tl.fromTo('#${p}-msrc', { opacity: 0, x: 60 }, { opacity: 1, x: 0, duration: 0.5, ease: 'expo.out' }, 1.75);
tl.fromTo('#${p}-flap', { scaleY: 1 }, { scaleY: -1, duration: 0.3, ease: 'power2.inOut' }, 2.05);
tl.fromTo('#${p}-paper', { y: 0 }, { y: -20, duration: 0.3, ease: 'power2.out' }, 2.25);
tl.fromTo('#${p}-mpath', { opacity: 0, strokeDashoffset: 60 }, { opacity: 0.9, strokeDashoffset: 0, duration: 0.3 }, 2.3);
tl.fromTo('#${p}-doc', { opacity: 0 }, { opacity: 1, duration: 0.12 }, 2.4);
travel('#${p}-doc', '#${p}-mpath', 2.4, 0.55);
tl.to('#${p}-doc', { opacity: 0, duration: 0.12 }, 2.95);
tl.fromTo('#${p}-rcard', { opacity: 0, scale: 0.94 }, { opacity: 1, scale: 1, duration: 0.45, ease: 'back.out(1.4)' }, 2.93);
// Maya opens the remittance
tl.to('#${p}-bsrc, #${p}-msrc, #${p}-bpath, #${p}-mpath', { opacity: 0, duration: 0.3, ease: 'power2.in' }, 3.05);
tl.fromTo('#${p}-cur', { x: 1600, y: 1100, opacity: 1 }, { x: ${(SX(1061) - 10).toFixed(0)}, y: ${(SY(419) - 6).toFixed(0)}, duration: 0.55, ease: 'power3.out' }, 2.95);
tl.to('#${p}-cur', { scale: 0.86, duration: 0.07, yoyo: true, repeat: 1, transformOrigin: '10px 6px' }, 3.5);
tl.to('#${p}-cur', { opacity: 0, duration: 0.15 }, 3.62);
// the card becomes the document (screen-space handoff)
tl.fromTo('#${p}-pw', { opacity: 0 }, { opacity: 1, duration: 0.3, ease: 'power2.inOut' }, 3.6);
tl.fromTo('#${p}-world', { opacity: 1 }, { opacity: 0, duration: 0.05 }, 3.92);
tl.fromTo('#${p}-pdf', { x: ${(SX(872) - 700).toFixed(0)}, y: ${(SY(346) - 150).toFixed(0)}, scaleX: ${(378 * 1.0406 / 480).toFixed(3)}, scaleY: ${(146 * 1.0406 / 800).toFixed(3)} }, { x: 0, y: 0, scaleX: 1, scaleY: 1, duration: 0.7, ease: 'expo.out' }, 3.6);
tl.fromTo('.${p}-prow, .${p}-pbar, #${p}-pdfl', { opacity: 0 }, { opacity: 1, duration: 0.3, stagger: 0.02 }, 3.95);
// three sources, fragmented
tl.fromTo('#${p}-bank', { opacity: 0, x: 80, rotation: 0 }, { opacity: 1, x: 0, rotation: -1.6, duration: 0.8, ease: 'expo.out' }, 4.35);
tl.fromTo('#${p}-erp', { opacity: 0, x: 80, rotation: 0 }, { opacity: 1, x: 0, rotation: 1.4, duration: 0.8, ease: 'expo.out' }, 4.55);
tl.to('#${p}-pdf', { rotation: -1.2, duration: 0.8, ease: 'power2.out' }, 4.4);
// titles
revealWords(tl, '.${p}-w1', 4.15);
revealWords(tl, '.${p}-w2', 4.75);
revealWords(tl, '.${p}-w3', 5.35, 0.06);
hideWords(tl, '.${p}-w1, .${p}-w2, .${p}-w3', 6.72, 0.02);
// everything gathers into one workspace: bank | remittance | ERP, no overlaps
tl.fromTo('#${p}-ws', { opacity: 0, scale: 0.96 }, { opacity: 1, scale: 1, duration: 0.55, ease: 'power3.out' }, 6.8);
tl.to('#${p}-bank', { x: 350 - 1290, y: 400 - 150, scale: 0.7, rotation: 0, duration: 0.7, ease: 'power3.inOut' }, 6.82);
tl.to('#${p}-pdf', { x: 840 - 700, y: 370 - 150, scale: 0.5, rotation: 0, duration: 0.7, ease: 'power3.inOut' }, 6.84);
tl.to('#${p}-erp', { x: 1240 - 1290, y: 370 - 520, scale: 0.7, rotation: 0, duration: 0.7, ease: 'power3.inOut' }, 6.86);
// follow the remittance
tl.to('#${p}-ws, #${p}-bank, #${p}-erp, #${p}-pdfl', { opacity: 0, filter: 'blur(10px)', duration: 0.28, ease: 'power2.in' }, 7.58);
tl.to('#${p}-pdf', { x: 0, y: 0, scale: 1, duration: 0.42, ease: 'power3.inOut' }, 7.58);
`,
};
