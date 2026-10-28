// 0:30–0:44 — The intelligent workflow. Six decisions, each showing its rule and source.
import { C, words } from "../shared.mjs";

const p = "s5";
const check = (col = C.bronzeD, sz = 22) => `<svg width="${sz}" height="${sz}" viewBox="0 0 22 22"><path d="M4 11.5 L9 16 L18 6" stroke="${col}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const row = (id, x, y, ref, acct, amt, opts = {}) => `
<div id="${id}" class="card" style="left:${x}px;top:${y}px;width:${opts.w ?? 540}px;height:${opts.h ?? 86}px;display:flex;align-items:center;padding:0 26px;gap:16px;${opts.style ?? ""}">
  <span style="font-size:26px;font-weight:700">${ref}</span><span class="chip chip-white" style="font-size:17px;padding:4px 12px">${acct}</span>
  <span class="num" style="margin-left:auto;font-size:28px;font-weight:700;color:${opts.col ?? C.ink}">${amt}</span>${opts.tail ?? ""}
</div>`;
const ruleBar = (id, rule, src) => `<div id="${id}" class="abs ${p}-rb" style="left:960px;top:270px;transform:translateX(-50%);display:flex;gap:12px;white-space:nowrap;opacity:0">
  <span class="chip chip-white" style="font-size:20px"><span class="dot" style="background:${C.bronze}"></span>Rule · ${rule}</span>
  <span class="chip chip-blue" style="font-size:20px">Confidence · High</span>
  <span class="chip chip-grey" style="font-size:20px">Source · ${src}</span></div>`;
const label = (id, txt, cls = "chip-bronze") => `<div id="${id}" class="abs" style="left:960px;top:830px;transform:translateX(-50%);opacity:0"><span class="chip ${cls}" style="font-size:28px;padding:12px 26px">${txt}</span></div>`;

// Beat A data
const A = [["INV-55141", "HX-1001", "$12,230.40"], ["INV-55188", "HX-1002", "$4,165.00"], ["INV-55167", "HX-1003", "$2,906.10"], ["INV-55172", "HX-1001", "$1,418.00"]];
const AY = (i) => 370 + i * 108;

const LOG = [
  ["✓ 59 invoices matched", "chip-bronze"],
  ["✓ Discounts verified", "chip-bronze"],
  ["Chargebacks → $0 invoice", "chip-amber"],
  ["✓ Recovery $212.75", "chip-bronze"],
  ["INV-55203 kept open", "chip-amber"],
  ["Duplicate held", "chip-red"],
];

export default {
  id: p,
  dur: 13.5,
  css: `
#${p}-t1{left:110px;top:64px;font-size:60px}
#${p}-t2{left:110px;top:134px;font-size:60px}
#${p}-log{left:110px;top:968px;display:flex;gap:12px;align-items:center}
#${p}-log .chip{font-size:17px;padding:6px 12px;opacity:0}
.${p}-col{position:absolute;top:322px;font-size:17px}
`,
  html: `
<div id="${p}-ask" class="card" style="left:1400px;top:56px;width:410px;padding:16px 20px;display:flex;gap:14px;align-items:center;opacity:0">
  <div style="width:46px;height:46px;border-radius:50%;background:${C.blueL};color:${C.blueD};font-weight:700;font-size:17px;display:flex;align-items:center;justify-content:center;flex:none">DP</div>
  <div><div style="font-size:16px;color:${C.ink2}"><b style="color:${C.ink}">Dev</b> asked</div><div style="font-size:23px;font-weight:700;letter-spacing:-.01em">“Did Helix pay in full?”</div></div>
</div>
<div id="${p}-t1" class="t-hero">${words("Every line understood.", p + "-w1")}</div>
<div id="${p}-t2" class="t-hero">${words("Every decision", p + "-w2")} <span class="w ${p}-w2 serif">explained.</span></div>

<!-- A: invoice matching -->
<div id="${p}-A" class="full">
  <div class="label ${p}-col" style="left:260px">Remittance lines</div><div class="label ${p}-col" style="left:1120px">ERP invoices</div>
  <svg class="full" viewBox="0 0 1920 1080" style="position:absolute;left:0;top:0">
    ${A.map((a, i) => `<path id="${p}-al${i}" d="M800 ${AY(i) + 43} L1120 ${AY(i) + 43}" stroke="${C.bronze}" stroke-width="2.5" fill="none" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/>`).join("")}
  </svg>
  ${A.map((a, i) => row(`${p}-ar${i}`, 260, AY(i), a[0], a[1], a[2])).join("")}
  ${A.map((a, i) => row(`${p}-ae${i}`, 1120, AY(i), a[0], a[1], "", { tail: `<span id="${p}-as${i}" class="chip chip-grey" style="font-size:17px">Open</span><span id="${p}-ak${i}" class="chip chip-bronze" style="font-size:17px;position:absolute;right:24px;opacity:0">${check(C.bronzeD, 18)}Matched</span>` })).join("")}
  ${label(`${p}-aL`, "Matched across related accounts")}
</div>
${ruleBar(`${p}-aR`, "invoice no. + amount", "remittance lines 1–59")}

<!-- B: early payment discount -->
<div id="${p}-B" class="card" style="left:500px;top:350px;width:920px;height:420px;padding:34px 40px;opacity:0">
  <div style="display:flex;align-items:center;gap:14px"><span style="font-size:36px;font-weight:700">INV-55188</span><span class="chip chip-white" style="font-size:18px">HX-1002</span><span class="chip chip-blue" style="font-size:18px;margin-left:auto">Terms 2/10 net 30 · paid in 8 days</span></div>
  <div style="margin-top:34px;font-size:28px;display:flex;justify-content:space-between;padding:14px 0;border-bottom:1px solid #F0EAE0"><span style="color:${C.ink2}">Gross</span><span class="num" style="font-weight:700">$4,250.00</span></div>
  <div style="font-size:28px;display:flex;justify-content:space-between;align-items:center;padding:14px 0;border-bottom:1px solid #F0EAE0"><span style="color:${C.ink2}">Early-payment discount · 2%</span><span id="${p}-bslot" style="width:230px;height:46px;border-radius:10px;border:2px dashed ${C.line};display:inline-flex;align-items:center;justify-content:center"><span id="${p}-bslotT" class="num" style="font-size:22px;font-weight:700;color:${C.bronze};opacity:0">−$85.00</span></span></div>
  <div style="font-size:28px;display:flex;justify-content:space-between;padding:14px 0"><span style="color:${C.ink2}">Net paid</span><span class="num" style="font-weight:700">$4,165.00</span></div>
</div>
<div id="${p}-bcol" class="card" style="left:1500px;top:430px;width:300px;padding:22px 24px;opacity:0"><div class="label">Discounts</div><div style="height:58px"></div><div style="font-size:17px;color:${C.ink2}">Booked to 4910 · Sales discounts</div></div>
<div id="${p}-bstrip" class="abs" style="left:1150px;top:540px;width:230px;height:46px;border-radius:10px;background:${C.bronze};color:#fff;font-size:24px;font-weight:700;display:flex;align-items:center;justify-content:center;opacity:0" class="num">−$85.00</div>
${ruleBar(`${p}-bR`, "early-payment terms", "remittance line 2 · ERP terms")}
${label(`${p}-bL`, "2% discount verified")}

<!-- C: chargebacks -->
<div id="${p}-C" class="card" style="left:900px;top:350px;width:820px;height:420px;padding:30px 36px;opacity:0;border:2px solid ${C.redL}">
  <div style="position:relative;height:90px">
    <div id="${p}-ch1" style="position:absolute;inset:0"><div class="label" style="color:${C.redD}">Exception</div><div style="font-size:36px;font-weight:700;margin-top:6px">Chargeback handling</div></div>
    <div id="${p}-ch2" style="position:absolute;inset:0;opacity:0"><div class="label" style="color:${C.amberD}">Proposed action</div><div style="font-size:36px;font-weight:700;margin-top:6px">Create $0 invoice</div></div>
    <span id="${p}-cbadge" class="chip chip-amber" style="position:absolute;right:0;top:4px;font-size:18px;opacity:0">Awaiting approval</span>
  </div>
  <div style="height:200px"></div>
  <div style="display:flex;justify-content:space-between;font-size:26px;border-top:1px solid #F0EAE0;padding-top:16px"><span style="color:${C.ink2}">Offsets 2 deductions</span><span class="num" style="font-weight:700;color:${C.redD}">−$515.75</span></div>
</div>
${row(`${p}-c1`, 200, 420, "CB-40112", "HX-1002", "−$326.50", { col: C.redD, style: `border-color:${C.redL};opacity:0` })}
${row(`${p}-c2`, 200, 530, "CB-40113", "HX-1003", "−$189.25", { col: C.redD, style: `border-color:${C.redL};opacity:0` })}
${ruleBar(`${p}-cR`, "chargeback handling", "remittance lines 5–6")}
${label(`${p}-cL`, "Chargeback handling", "chip-amber")}

<!-- D: chargeback paid back -->
<div id="${p}-D" class="card" style="left:410px;top:400px;width:1100px;height:240px;padding:32px 40px;opacity:0">
  <div style="display:flex;align-items:center;gap:14px"><span style="font-size:36px;font-weight:700">CB-39870</span><span class="chip chip-white" style="font-size:18px">HX-1001</span></div>
  <div style="font-size:22px;color:${C.ink2};margin-top:12px">Deducted on remittance HX-771904 (Sep 18) · repaid on HX-778120</div>
  <div style="position:absolute;right:40px;top:40px;width:330px;height:68px">
    <div id="${p}-dneg" class="chip chip-red" style="position:absolute;right:0;top:0;font-size:30px;padding:12px 22px"><svg id="${p}-darr" width="30" height="30" viewBox="0 0 30 30"><path d="M24 15 H6 M12 8 L5 15 L12 22" stroke="currentColor" stroke-width="3.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg><span class="num">−$212.75</span></div>
    <div id="${p}-dpos" class="chip chip-bronze" style="position:absolute;right:0;top:0;font-size:30px;padding:12px 22px;opacity:0"><svg width="30" height="30" viewBox="0 0 30 30"><path d="M6 15 H24 M18 8 L25 15 L18 22" stroke="currentColor" stroke-width="3.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg><span class="num">+$212.75</span></div>
  </div>
  <div style="position:absolute;left:40px;bottom:30px;font-size:22px"><span id="${p}-dst" style="color:${C.redD};font-weight:700">Previously deducted</span></div>
</div>
${ruleBar(`${p}-dR`, "prior deduction lookup", "remittance line 7 · Sep 18 history")}
${label(`${p}-dL`, "Chargeback recovery identified")}

<!-- E: short payment -->
<div id="${p}-E" class="card" style="left:410px;top:360px;width:1100px;height:380px;padding:32px 40px;opacity:0">
  <div style="display:flex;align-items:center;gap:14px"><span style="font-size:36px;font-weight:700">INV-55203</span><span class="chip chip-white" style="font-size:18px">HX-1003</span><span id="${p}-est" class="chip chip-amber" style="margin-left:auto;font-size:20px">Open</span></div>
  <div style="display:flex;gap:60px;margin-top:30px;font-size:24px;color:${C.ink2}"><div>Invoice<div class="num" style="font-size:40px;font-weight:700;color:${C.ink}">$1,000.00</div></div><div>Paid<div class="num" style="font-size:40px;font-weight:700;color:${C.ink}">$881.60</div></div><div>Remaining<div class="num" style="font-size:40px;font-weight:700;color:${C.amberD}">$118.40</div></div></div>
  <div style="position:relative;margin-top:36px;height:28px;border-radius:14px;background:${C.amberL};overflow:hidden">
    <div id="${p}-efill" style="position:absolute;left:0;top:0;bottom:0;width:88.16%;background:${C.bronze};transform-origin:0 50%"></div>
    <div id="${p}-erem" style="position:absolute;right:0;top:0;bottom:0;width:11.84%;background:${C.amber};opacity:0"></div>
  </div>
</div>
${ruleBar(`${p}-eR`, "short payment", "remittance line 3")}
${label(`${p}-eL`, "Keep remaining balance open", "chip-amber")}

<!-- F: duplicate protection -->
<div id="${p}-F" class="full" style="opacity:0">
  <div id="${p}-fbar" class="abs" style="left:1060px;top:390px;width:6px;height:230px;border-radius:3px;background:${C.red};opacity:0;box-shadow:0 0 0 10px rgba(181,97,90,.10)"></div>
  <div id="${p}-fbl" class="abs" style="left:1000px;top:350px;opacity:0"><span class="chip chip-red" style="font-size:18px">Already paid</span></div>
  ${row(`${p}-finv`, 1100, 460, "INV-54877", "HX-1001", "", { tail: `<span class="chip chip-grey" style="font-size:18px">Paid Oct 03 · R-88214</span>`, w: 640, style: "background:#F7F4EE" })}
  ${row(`${p}-frow`, 200, 460, "INV-54877", "line 4", "$640.00")}
  <div id="${p}-fex" class="card" style="left:560px;top:640px;width:800px;padding:20px 28px;display:flex;align-items:center;gap:16px;opacity:0;border:2px solid ${C.redL}"><span class="label" style="color:${C.redD}">Exceptions</span><span style="font-size:24px;font-weight:700">Unapplied credit</span><span class="num" style="margin-left:auto;font-size:28px;font-weight:700">$640.00</span><span class="chip chip-grey" style="font-size:16px">Held for customer</span></div>
</div>
${ruleBar(`${p}-fR`, "duplicate check", "remittance line 4 · ERP receipts")}
${label(`${p}-fL`, "Duplicate payment prevented", "chip-red")}

<div id="${p}-log" class="abs"><span class="label" style="margin-right:6px">Decisions</span>${LOG.map((l, i) => `<span id="${p}-lg${i}" class="chip ${l[1]}">${l[0]}</span>`).join("")}<span id="${p}-need" class="chip chip-amber" style="font-size:17px;opacity:0;border:2px solid ${C.amber}">Needs review · 2</span></div>
`,
  js: `
const pop = (sel, t) => tl.fromTo(sel, { opacity: 0, scale: 0.85 }, { opacity: 1, scale: 1, duration: 0.35, ease: 'back.out(1.6)' }, t);
const inOut = (sel, a, b) => { tl.fromTo(sel, { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.35, ease: 'expo.out' }, a); tl.to(sel, { opacity: 0, duration: 0.2, ease: 'power2.in' }, b); };
revealWords(tl, '.${p}-w1', 0.15);
tl.fromTo('#${p}-ask', { opacity: 0, x: 40 }, { opacity: 1, x: 0, duration: 0.5, ease: 'expo.out' }, 0.25);
revealWords(tl, '.${p}-w2', 6.6, 0.1);

// ---- A: invoice matching (0.3 – 2.7)
tl.fromTo('.${p}-col', { opacity: 0 }, { opacity: 1, duration: 0.3 }, 0.3);
${A.map((a, i) => `tl.fromTo('#${p}-ar${i}', { opacity: 0, x: -60 }, { opacity: 1, x: 0, duration: 0.5, ease: 'expo.out' }, ${(0.3 + i * 0.06).toFixed(2)});
tl.fromTo('#${p}-ae${i}', { opacity: 0, x: 220, y: ${[-8, 10, -6, 8][i]} }, { opacity: 0.85, x: 150, y: ${[-8, 10, -6, 8][i]}, duration: 0.5, ease: 'expo.out' }, ${(0.45 + i * 0.06).toFixed(2)});
tl.to('#${p}-ae${i}', { y: 0, x: 0, opacity: 1, duration: 0.6, ease: 'power3.inOut' }, ${(1.0 + i * 0.06).toFixed(2)});
tl.to('#${p}-al${i}', { strokeDashoffset: 0, duration: 0.35, ease: 'power2.out' }, ${(1.5 + i * 0.07).toFixed(2)});
tl.to('#${p}-as${i}', { opacity: 0, duration: 0.15 }, ${(1.75 + i * 0.07).toFixed(2)});
pop('#${p}-ak${i}', ${(1.75 + i * 0.07).toFixed(2)});`).join("\n")}
inOut('#${p}-aR', 0.5, 2.55);
inOut('#${p}-aL', 1.95, 2.55);
pop('#${p}-lg0', 2.25);
tl.to('#${p}-A', { opacity: 0, x: -80, duration: 0.3, ease: 'power3.in' }, 2.6);

// ---- B: early-payment discount (2.75 – 4.8)
tl.fromTo('#${p}-B', { opacity: 0, x: 260 - 500, y: ${AY(1)} - 350, scale: 0.55, transformOrigin: '0 0' }, { opacity: 1, x: 0, y: 0, scale: 1, duration: 0.55, ease: 'expo.out' }, 2.7);
tl.fromTo('#${p}-bstrip', { opacity: 0 }, { opacity: 1, duration: 0.25 }, 3.15);
inOut('#${p}-bR', 2.95, 4.65);
tl.fromTo('#${p}-bcol', { opacity: 0, x: 40 }, { opacity: 1, x: 0, duration: 0.4, ease: 'expo.out' }, 3.45);
tl.to('#${p}-bstrip', { x: 1524 - 1150, y: 478 - 540, scale: 0.9, transformOrigin: '0 0', duration: 0.6, ease: 'power3.inOut' }, 3.65);
inOut('#${p}-bL', 3.95, 4.65);
tl.fromTo('#${p}-bslotT', { opacity: 0 }, { opacity: 0.7, duration: 0.3 }, 4.1);
pop('#${p}-lg1', 4.3);
tl.to('#${p}-B, #${p}-bstrip, #${p}-bcol', { opacity: 0, y: -30, duration: 0.25, ease: 'power3.in' }, 4.65);

// ---- C: chargebacks (4.85 – 7.0)
tl.fromTo('#${p}-c1', { opacity: 0, x: -80 }, { opacity: 1, x: 0, duration: 0.45, ease: 'expo.out' }, 4.85);
tl.fromTo('#${p}-c2', { opacity: 0, x: -80 }, { opacity: 1, x: 0, duration: 0.45, ease: 'expo.out' }, 4.93);
inOut('#${p}-cR', 5.0, 6.85);
tl.fromTo('#${p}-C', { opacity: 0, scale: 0.94 }, { opacity: 1, scale: 1, duration: 0.45, ease: 'expo.out' }, 5.2);
tl.to('#${p}-c1', { x: 936 - 200, y: 480 - 420, scale: 0.9, transformOrigin: '0 0', duration: 0.55, ease: 'power3.inOut' }, 5.35);
tl.to('#${p}-c2', { x: 936 - 200, y: 576 - 530, scale: 0.9, transformOrigin: '0 0', duration: 0.55, ease: 'power3.inOut' }, 5.42);
tl.to('#${p}-ch1', { opacity: 0, y: -8, duration: 0.15, ease: 'power2.in' }, 6.0);
tl.fromTo('#${p}-ch2', { opacity: 0, y: 8 }, { opacity: 1, y: 0, duration: 0.35, ease: 'expo.out' }, 6.17);
tl.fromTo('#${p}-C', { borderColor: '${C.redL}' }, { borderColor: '${C.amber}', duration: 0.3 }, 6.05);
pop('#${p}-cbadge', 6.25);
inOut('#${p}-cL', 5.9, 6.85);
pop('#${p}-lg2', 6.4);
tl.to('#${p}-C, #${p}-c1, #${p}-c2', { opacity: 0, y: -30, duration: 0.25, ease: 'power3.in' }, 6.8);

// ---- D: chargeback paid back (7.0 – 8.85)
tl.fromTo('#${p}-D', { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.45, ease: 'expo.out' }, 7.0);
inOut('#${p}-dR', 7.1, 8.7);
tl.to('#${p}-darr', { rotation: 180, transformOrigin: '50% 50%', duration: 0.4, ease: 'power3.inOut' }, 7.6);
tl.to('#${p}-dneg', { opacity: 0, x: 30, duration: 0.3, ease: 'power2.in' }, 7.9);
tl.fromTo('#${p}-dpos', { opacity: 0, x: -30 }, { opacity: 1, x: 0, duration: 0.4, ease: 'expo.out' }, 8.0);
tl.fromTo('#${p}-dst', { color: '${C.redD}' }, { color: '${C.bronzeD}', duration: 0.3 }, 8.0);
tl.call(() => {}, null, 8.0);
tl.fromTo('#${p}-dst', { opacity: 1 }, { opacity: 1, duration: 0.01, onStart: () => { $('#${p}-dst').textContent = 'Recovered · applied to HX-1001'; }, onReverseComplete: () => { $('#${p}-dst').textContent = 'Previously deducted'; } }, 8.0);
inOut('#${p}-dL', 8.05, 8.7);
pop('#${p}-lg3', 8.3);
tl.to('#${p}-D', { opacity: 0, y: -30, duration: 0.25, ease: 'power3.in' }, 8.7);

// ---- E: short payment (8.9 – 10.7)
tl.fromTo('#${p}-E', { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.45, ease: 'expo.out' }, 8.9);
inOut('#${p}-eR', 9.0, 10.55);
tl.fromTo('#${p}-efill', { scaleX: 0 }, { scaleX: 1, duration: 0.6, ease: 'power3.out' }, 9.2);
tl.fromTo('#${p}-erem', { opacity: 0 }, { opacity: 1, duration: 0.3 }, 9.75);
tl.fromTo('#${p}-est', { opacity: 1 }, { opacity: 1, duration: 0.01, onStart: () => { $('#${p}-est').textContent = 'Kept open · $118.40'; }, onReverseComplete: () => { $('#${p}-est').textContent = 'Open'; } }, 9.95);
tl.fromTo('#${p}-est', { scale: 1 }, { scale: 1.08, duration: 0.15, yoyo: true, repeat: 1 }, 9.95);
inOut('#${p}-eL', 9.85, 10.55);
pop('#${p}-lg4', 10.15);
tl.to('#${p}-E', { opacity: 0, y: -30, duration: 0.25, ease: 'power3.in' }, 10.55);

// ---- F: duplicate protection (10.75 – 12.8)
tl.fromTo('#${p}-F', { opacity: 0 }, { opacity: 1, duration: 0.3 }, 10.75);
inOut('#${p}-fR', 10.85, 12.65);
tl.fromTo('#${p}-frow', { x: 0 }, { x: 500 - 200, duration: 0.9, ease: 'power4.out' }, 11.0);
tl.fromTo('#${p}-fbar', { opacity: 0, scaleY: 0.2 }, { opacity: 1, scaleY: 1, duration: 0.3, ease: 'expo.out' }, 11.35);
tl.fromTo('#${p}-fbl', { opacity: 0 }, { opacity: 1, duration: 0.25 }, 11.45);
tl.to('#${p}-frow', { x: 450 - 200, duration: 0.25, ease: 'power2.out' }, 11.9);
tl.to('#${p}-frow', { x: 560 - 200, y: 640 - 460, opacity: 0, duration: 0.5, ease: 'power3.inOut' }, 12.16);
tl.fromTo('#${p}-fex', { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.4, ease: 'expo.out' }, 12.3);
inOut('#${p}-fL', 11.9, 12.65);
pop('#${p}-lg5', 12.4);
tl.to('#${p}-F', { opacity: 0, y: -30, duration: 0.25, ease: 'power3.in' }, 12.7);

// ---- outro: two decisions need Maya
hideWords(tl, '.${p}-w1, .${p}-w2', 12.75, 0.02);
tl.to('#${p}-log', { y: 540 - 968 - 20, x: -20, scale: 1.0, transformOrigin: '0 50%', duration: 0.55, ease: 'power3.inOut' }, 12.75);
tl.to('#${p}-lg0, #${p}-lg1, #${p}-lg3, #${p}-lg5', { opacity: 0.35, duration: 0.3 }, 12.95);
pop('#${p}-need', 13.1);
tl.to('#${p}-ask', { x: -560, y: 300, duration: 0.55, ease: 'power3.inOut' }, 12.75);
`,
};
