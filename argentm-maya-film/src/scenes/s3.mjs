// 0:15–0:23 — AI reads and understands. The PDF's rows become a structured table.
import { C, remitPDF, words, brandMark } from "../shared.mjs";

const p = "s3";
const ROWS = [
  { ref: "INV-55141", acct: "HX-1001", g: "12,480.00", d: "249.60", n: "12,230.40", type: ["Invoice", "chip-blue"] },
  { ref: "INV-55188", acct: "HX-1002", g: "4,250.00", d: "85.00", n: "4,165.00", type: ["Invoice", "chip-blue"] },
  { ref: "INV-55203", acct: "HX-1003", g: "1,000.00", d: "—", n: "881.60", type: ["Invoice", "chip-blue"] },
  { ref: "INV-54877", acct: "HX-1001", g: "640.00", d: "—", n: "640.00", type: ["Invoice", "chip-blue"] },
  { ref: "CB-40112", acct: "HX-1002", g: "—", d: "—", n: "−326.50", type: ["Deduction", "chip-red"], col: C.redD },
  { ref: "CB-40113", acct: "HX-1003", g: "—", d: "—", n: "−189.25", type: ["Deduction", "chip-red"], col: C.redD },
  { ref: "CB-39870", acct: "HX-1001", g: "—", d: "—", n: "+212.75", type: ["Repayment", "chip-bronze"], col: C.bronzeD },
  { ref: "+ 56 invoice lines", acct: "3 accounts", g: "31,076.40", d: "78.00", n: "30,998.40", type: ["Invoice ×56", "chip-blue"], muted: true },
];
const TX = 520, TY = 470, RH = 56;
// columns: [left, width, align]
const COLS = [[0, 250, "left"], [250, 200, "left"], [450, 230, "right"], [690, 180, "right"], [880, 210, "right"], [1110, 150, "left"]];
const cell = (i, txt, extra = "") => `<span style="position:absolute;left:${COLS[i][0]}px;width:${COLS[i][1]}px;text-align:${COLS[i][2]};${extra}">${txt}</span>`;
const jit = (i, k) => [6, -9, 12, -4, 8, -12, 5, -7, 10][(i * 3 + k) % 9];

const rowsHTML = ROWS.map((r, i) => {
  const y = TY + i * RH;
  const raw = `<div class="${p}-raw abs" id="${p}-raw${i}" style="left:${TX}px;top:${y}px;width:1250px;height:${RH}px;font-family:'IBM Plex Mono',monospace;font-size:19px;color:#6F685F;line-height:${RH}px">
    ${cell(0, r.ref, `margin-left:${jit(i, 0)}px`)}${cell(2, r.g, `margin-left:${jit(i, 1)}px`)}${cell(3, r.d, `margin-left:${jit(i, 2)}px`)}${cell(4, r.n, `margin-left:${jit(i, 3)}px`)}</div>`;
  const clean = `<div class="${p}-clean abs" id="${p}-clean${i}" style="left:${TX}px;top:${y}px;width:1250px;height:${RH}px;font-size:22px;line-height:${RH}px;color:${r.muted ? C.ink2 : C.ink};border-bottom:1px solid #F0EAE0">
    ${cell(0, r.ref, "font-weight:700")}${cell(1, `<span class="chip chip-white ${p}-ac${i}" style="font-size:17px;padding:4px 12px">${r.acct}</span>`)}${cell(2, r.g, "font-variant-numeric:tabular-nums")}${cell(3, r.d, "font-variant-numeric:tabular-nums")}${cell(4, r.n, `font-variant-numeric:tabular-nums;font-weight:700;color:${r.col ?? "inherit"}`)}${cell(5, `<span class="chip ${r.type[1]}" style="font-size:16px;padding:4px 12px">${r.type[0]}</span>`)}</div>`;
  return raw + clean;
}).join("");
const totY = TY + ROWS.length * RH + 8;

export default {
  id: p,
  dur: 8,
  css: `
#${p}-pdf{left:700px;top:150px;width:480px;height:800px;transform-origin:0 0}
#${p}-win{left:110px;top:190px;width:1700px;height:850px;opacity:0}
#${p}-title{left:110px;top:60px}
.${p}-clean{opacity:0}
#${p}-scan{left:${TX - 20}px;top:0;width:1290px;height:4px;background:${C.blue};border-radius:2px;box-shadow:0 0 0 6px rgba(91,127,166,.12),0 0 30px rgba(91,127,166,.35);opacity:0}
#${p}-total{left:${TX}px;top:${totY}px;width:1250px;height:60px;border-radius:12px;background:#F7F3EC;font-size:24px;line-height:60px;font-weight:700}
#${p}-dock{left:1150px;top:290px;width:620px;height:110px;border:2px dashed ${C.line};border-radius:14px;opacity:0}
.${p}-lift{position:absolute;opacity:0;transform-origin:50% 50%}
`,
  html: `
<div id="${p}-win" class="card">
  <div style="position:absolute;left:0;right:0;top:0;height:76px;border-bottom:1px solid ${C.line};display:flex;align-items:center;padding:0 32px;gap:16px;font-size:22px">
    ${brandMark(30)}<b>Remittances</b><span style="color:${C.ink2}">/ HX-778120 · Helix Health Distribution</span>
    <span id="${p}-status" class="chip chip-blue" style="margin-left:auto;font-size:18px"><span class="dot" style="background:${C.blue}"></span><span id="${p}-count">Reading 0 / 64 lines</span></span>
  </div>
  <div class="label abs" style="left:40px;top:104px">Source</div>
  <div class="label abs" style="left:${TX - 110}px;top:104px">Extracted lines</div>
</div>
<div id="${p}-bankbox" class="card" style="left:${TX}px;top:290px;width:600px;height:110px;opacity:0;display:flex;align-items:center;gap:20px;padding:0 26px">
  <span class="chip chip-grey" style="font-size:16px">Bank deposit</span><span class="num" style="font-size:36px;font-weight:700;letter-spacing:-.02em">+$48,612.40</span>
</div>
<div id="${p}-dock" class="abs"></div>
<div id="${p}-dtotal" class="card" style="left:1150px;top:290px;width:620px;height:110px;opacity:0;display:flex;align-items:center;gap:20px;padding:0 26px">
  <span class="chip chip-blue" style="font-size:16px">Remittance total</span><span class="num" style="font-size:36px;font-weight:700;letter-spacing:-.02em">$48,612.40</span>
</div>
<div id="${p}-thead" class="abs label" style="left:${TX}px;top:${TY - 36}px;width:1250px;height:30px;opacity:0">
  ${cell(0, "Invoice")}${cell(1, "Account")}${cell(2, "Gross")}${cell(3, "Discount")}${cell(4, "Net")}${cell(5, "Read as")}
</div>
${rowsHTML}
<div id="${p}-total" class="abs" style="opacity:0">
  <span style="position:absolute;left:24px">TOTAL</span>
  <span id="${p}-sumchip" class="chip chip-blue" style="position:absolute;left:180px;top:11px;font-size:17px;padding:6px 14px">Summary row · not an invoice</span>
  <span class="num" style="position:absolute;right:${1250 - 1090}px">48,612.40</span>
</div>
<div id="${p}-scan" class="abs"></div>
<div id="${p}-pdf" class="abs">${remitPDF(p)}</div>
<div id="${p}-title" class="t-hero">${words("From documents to", p + "-w")} <span class="w ${p}-w serif">understanding.</span></div>
<span id="${p}-l1" data-layout-allow-overlap class="chip chip-white ${p}-lift" style="font-size:40px;padding:14px 30px">HX-1001</span>
<span id="${p}-l2" data-layout-allow-overlap class="chip chip-white ${p}-lift" style="font-size:40px;padding:14px 30px">HX-1002</span>
<span id="${p}-l3" data-layout-allow-overlap class="chip chip-white ${p}-lift" style="font-size:40px;padding:14px 30px">HX-1003</span>
`,
  js: `
// the document settles into the source pane while the app builds around it
tl.fromTo('#${p}-win', { opacity: 0, scale: 1.03 }, { opacity: 1, scale: 1, duration: 0.7, ease: 'power3.out' }, 0.3);
tl.fromTo('#${p}-pdf', { x: 0, y: 0, scale: 1 }, { x: 150 - 700, y: 300 - 150, scale: 0.6, duration: 0.85, ease: 'power3.inOut' }, 0.15);
tl.fromTo('#${p}-thead', { opacity: 0 }, { opacity: 1, duration: 0.4 }, 0.9);
// raw lines lift off the page into the table
${ROWS.map((r, i) => `tl.fromTo('#${p}-raw${i}', { opacity: 0, x: ${150 + 20 - TX}, y: ${300 + 190 + i * 14 - (TY + i * RH)}, scale: 0.55, transformOrigin: '0 0' }, { opacity: 1, x: 0, y: 0, scale: 1, duration: 0.7, ease: 'expo.out' }, ${(1.0 + i * 0.07).toFixed(2)});`).join("\n")}
tl.fromTo('#${p}-total', { opacity: 0, y: -380 }, { opacity: 1, y: 0, duration: 0.7, ease: 'expo.out' }, 1.6);
// the reading pass
tl.fromTo('#${p}-scan', { opacity: 0, y: ${TY - 6} }, { opacity: 1, y: ${TY - 6}, duration: 0.2 }, 2.1);
tl.to('#${p}-scan', { y: ${totY + 62}, duration: 2.0, ease: 'none' }, 2.3);
tl.to('#${p}-scan', { opacity: 0, duration: 0.25 }, 4.3);
${ROWS.map((r, i) => {
  const ti = 2.3 + ((i * RH + RH * 0.6) / (totY + 68 - TY)) * 2.0;
  return `tl.to('#${p}-raw${i}', { opacity: 0, duration: 0.25 }, ${ti.toFixed(2)});
tl.fromTo('#${p}-clean${i}', { opacity: 0 }, { opacity: 1, duration: 0.3, ease: 'power2.out' }, ${ti.toFixed(2)});`;
}).join("\n")}
countTo(tl, '#${p}-count', 0, 64, 2.3, 2.0, (v) => 'Reading ' + Math.round(v) + ' / 64 lines');
tl.call(() => {}, null, 4.3);
tl.fromTo('#${p}-status', { backgroundColor: '${C.blueL}', color: '${C.blueD}' }, { backgroundColor: '${C.bronzeL}', color: '${C.bronzeD}', duration: 0.3 }, 4.35);
tl.fromTo('#${p}-count', { opacity: 1 }, { opacity: 1, duration: 0.01, onComplete: () => { const e = $('#${p}-count'); if (e) e.textContent = '64 lines processed'; }, onReverseComplete: () => {} }, 4.35);
// TOTAL is recognised as a summary and docks beside the bank amount
tl.fromTo('#${p}-sumchip', { opacity: 0, x: -10 }, { opacity: 1, x: 0, duration: 0.35, ease: 'expo.out' }, 4.25);
tl.fromTo('#${p}-bankbox', { opacity: 0, y: 12 }, { opacity: 1, y: 0, duration: 0.5, ease: 'expo.out' }, 4.4);
tl.fromTo('#${p}-dock', { opacity: 0 }, { opacity: 1, duration: 0.3 }, 4.5);
tl.to('#${p}-total', { x: 1150 - ${TX}, y: ${290 + 25 - totY}, opacity: 0, duration: 0.7, ease: 'power3.inOut' }, 4.75);
tl.fromTo('#${p}-dtotal', { opacity: 0, y: 60, scale: 0.96 }, { opacity: 1, y: 0, scale: 1, duration: 0.6, ease: 'expo.out' }, 5.05);
tl.to('#${p}-sumchip', { opacity: 0, duration: 0.2 }, 4.7);
tl.to('#${p}-dock', { opacity: 0, duration: 0.3 }, 5.4);
// titles
revealWords(tl, '.${p}-w', 2.5, 0.1);
hideWords(tl, '.${p}-w', 6.05, 0.03);
// the three account ids lift out of the table
const lifts = [['#${p}-l1', '.${p}-ac0', 460], ['#${p}-l2', '.${p}-ac1', 960], ['#${p}-l3', '.${p}-ac2', 1460]];
const startPos = [[${TX + 250}, ${TY + 10}], [${TX + 250}, ${TY + RH + 10}], [${TX + 250}, ${TY + 2 * RH + 10}]];
lifts.forEach(([sel, src, fx], i) => {
  const e = $(sel); const w = e.offsetWidth, h = e.offsetHeight;
  tl.fromTo(sel, { opacity: 0, x: startPos[i][0] - w / 2 + 50, y: startPos[i][1] - h / 2 + 18, scale: 0.42 },
    { opacity: 1, x: startPos[i][0] - w / 2 + 50, y: startPos[i][1] - h / 2 + 18, scale: 0.42, duration: 0.15 }, 6.0 + i * 0.08);
  tl.to(sel, { x: fx - w / 2, y: 575 - h / 2, scale: 1, duration: 0.95, ease: 'power3.inOut' }, 6.2 + i * 0.08);
});
tl.to('#${p}-win, #${p}-pdf, .${p}-clean, #${p}-thead, #${p}-total, #${p}-bankbox, #${p}-dtotal', { opacity: 0, filter: 'blur(8px)', duration: 0.5, ease: 'power2.in' }, 6.4);
`,
};
