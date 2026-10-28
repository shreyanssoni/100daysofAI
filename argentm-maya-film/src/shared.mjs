import { mayaSide, mayaBack, dev, mayaFaceSVG, P } from "./characters.mjs";
export { mayaSide, mayaBack };
export const colleague = dev;
// Shared building blocks for every scene: palette, type, the office, Maya and the app UI.
// Scenes are generated from these so the room, the character and the product stay identical.

export const C = {
  ivory: "#FBF8F2",
  paper: "#F5F0E7",
  line: "#E6DFD3",
  ink: "#2B2A28",
  ink2: "#6B665E",
  bronze: "#A9773F",
  bronzeD: "#8A5E2E",
  bronzeL: "#F1E5D6",
  blue: "#5B7FA6",
  blueD: "#3F6489",
  blueL: "#E3EBF3",
  amber: "#C99A3B",
  amberD: "#8E6A22",
  amberL: "#F6EAD0",
  red: "#B5615A",
  redD: "#9A4E47",
  redL: "#F3E0DC",
  grey: "#ECE8E1",
  skin: "#E7B793",
  skinS: "#D49C78",
  hair: "#2B211C",
  sweater: "#F0E4D0",
  sweaterS: "#DFCDB2",
  trousers: "#3A3940",
  trousersB: "#2F2E35",
  shoe: "#26231F",
};

// ---------- CSS shared by all scenes ----------
export const CSS = `
#root{--ground:radial-gradient(1400px 900px at 68% 22%, #FFFDF9 0%, #FAF5EC 55%, #F3EBDE 100%);position:absolute;inset:0;overflow:hidden;background:var(--ground);font-family:Inter,sans-serif;color:${C.ink};-webkit-font-smoothing:antialiased}
.abs{position:absolute}
.full{position:absolute;left:0;top:0;width:1920px;height:1080px}
.world{position:absolute;left:0;top:0;width:1920px;height:1080px;transform-origin:0 0}
.num{font-variant-numeric:tabular-nums}
.serif{font-family:"EB Garamond",serif;font-style:italic;font-weight:400;letter-spacing:-.01em}
.t-hero{position:absolute;font-weight:700;font-size:72px;line-height:1.06;letter-spacing:-.035em;color:${C.ink}}
.t-hero .serif{font-size:1.12em;color:${C.bronze}}
.t-sub{position:absolute;font-weight:400;font-size:40px;line-height:1.15;letter-spacing:-.02em;color:${C.ink2}}
.w{display:inline-block;white-space:pre}
.card{position:absolute;background:#fff;border:1px solid #EAE3D7;border-radius:20px;box-shadow:0 1px 2px rgba(70,50,30,.05),0 6px 14px rgba(70,50,30,.05),0 28px 60px rgba(70,50,30,.10)}
.chip{display:inline-flex;align-items:center;gap:8px;padding:7px 14px;border-radius:999px;font-weight:700;font-size:20px;line-height:1.15;letter-spacing:-.005em;white-space:nowrap;vertical-align:middle}
.chip-blue{background:${C.blueL};color:${C.blueD}}
.chip-amber{background:${C.amberL};color:${C.amberD}}
.chip-red{background:${C.redL};color:${C.redD}}
.chip-bronze{background:${C.bronzeL};color:${C.bronzeD}}
.chip-grey{background:${C.grey};color:${C.ink2}}
.chip-white{background:#fff;color:${C.ink};border:1px solid ${C.line}}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.label{font-size:18px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:${C.ink2}}
.btn{display:inline-flex;align-items:center;justify-content:center;border-radius:12px;font-weight:700;white-space:nowrap}
.btn-bronze{background:${C.bronze};color:#fff}
.btn-ghost{background:#fff;color:${C.ink};border:1.5px solid ${C.line}}
.cursor{position:absolute;width:64px;height:64px;left:0;top:0;z-index:50}
/* ----- the app, authored at 1280x720 and scaled where it is shown ----- */
.app{position:absolute;left:0;top:0;width:1280px;height:720px;background:#FBF9F5;display:flex;font-family:Inter,sans-serif;color:${C.ink};overflow:hidden;transform-origin:0 0}
.app-nav{width:192px;flex:none;background:#F3EEE5;border-right:1px solid ${C.line};padding:22px 14px}
.brand{display:flex;align-items:center;gap:9px;font-weight:700;font-size:19px;letter-spacing:-.02em;margin:0 6px 26px}
.nav-i{font-size:14px;padding:9px 12px;border-radius:9px;color:${C.ink2};margin-bottom:3px;display:flex;align-items:center;gap:10px}
.nav-i i{width:14px;height:14px;border-radius:4px;border:1.6px solid currentColor;display:inline-block;opacity:.7}
.nav-i.on{background:#fff;color:${C.ink};font-weight:700;box-shadow:0 1px 2px rgba(70,50,30,.08)}
.nav-i.on i{border-color:${C.bronze};background:${C.bronzeL};opacity:1}
.app-main{flex:1;position:relative;padding:22px 30px}
.topbar{display:flex;justify-content:space-between;align-items:center;font-size:13px;color:${C.ink2};margin-bottom:18px}
.avatar{width:28px;height:28px;border-radius:50%;background:${C.bronzeL};color:${C.bronzeD};font-weight:700;font-size:11px;display:inline-flex;align-items:center;justify-content:center}
.a-title{font-size:30px;font-weight:700;letter-spacing:-.03em;margin-bottom:4px}
.a-sub{font-size:15px;color:${C.ink2}}
.a-card{background:#fff;border:1px solid #ECE5D9;border-radius:14px;box-shadow:0 1px 2px rgba(70,50,30,.05),0 8px 20px rgba(70,50,30,.06)}
.a-row{display:flex;align-items:center;font-size:15px;padding:12px 18px;border-top:1px solid #F0EAE0}
.a-pill{font-size:12px;font-weight:700;padding:4px 10px;border-radius:999px;line-height:1.2;white-space:nowrap;display:inline-block}
`;

// ---------- small pieces ----------
// No logo marks anywhere in the film (client request).
export const brandMark = () => "";

export const cursorSVG = (id) => `
<svg id="${id}" class="cursor" viewBox="0 0 64 64">
  <path d="M10 6 L10 50 L21 39 L29 57 L37 53.5 L29 36 L45 36 Z" fill="${C.ink}" stroke="#fff" stroke-width="3" stroke-linejoin="round"/>
</svg>`;

// split a line into word spans for per-word reveals
export const words = (text, idp) =>
  text
    .split(" ")
    .map((w, i) => `<span class="w ${idp}" >${w}${i < text.split(" ").length - 1 ? " " : ""}</span>`)
    .join("");

// ---------- the app UI (1280x720) ----------
const NAV = ["Dashboard", "Remittances", "Customers", "Matching", "Exceptions", "Posting & Audit"];
export const appShell = (active, content, extra = "") => `
<div class="app" ${extra}>
  <div class="app-nav">
    <div class="brand" style="font-size:16px;color:#4A463F">AR Workspace</div>
    ${NAV.map((n) => `<div class="nav-i ${n === active ? "on" : ""}"><i></i>${n}</div>`).join("")}
  </div>
  <div class="app-main">
    <div class="topbar"><span>Cash application &nbsp;/&nbsp; <b style="color:${C.ink}">${active}</b></span><span style="display:flex;gap:14px;align-items:center">Tue, Oct 27 · 9:12 AM <span class="avatar">MC</span></span></div>
    ${content}
  </div>
</div>`;

// Dashboard used on Maya's monitor in the opening.
export const dashContent = (p) => `
  <div class="a-title">Good morning, Maya</div>
  <div class="a-sub" style="margin-bottom:18px">Here's what came in overnight.</div>
  <div style="display:flex;gap:14px;margin-bottom:16px">
    <div class="a-card" style="flex:1;padding:16px 18px"><div style="font-size:13px;color:${C.ink2}">Remittances today</div><div style="font-size:34px;font-weight:700;letter-spacing:-.03em" class="num">141</div></div>
    <div class="a-card" style="flex:1;padding:16px 18px"><div style="font-size:13px;color:${C.ink2}">Applied without a touch</div><div style="font-size:34px;font-weight:700;letter-spacing:-.03em" class="num">138</div></div>
    <div class="a-card" style="flex:1;padding:16px 18px"><div style="font-size:13px;color:${C.ink2}">Awaiting review</div><div style="font-size:34px;font-weight:700;letter-spacing:-.03em" class="num">0</div></div>
  </div>
  <div class="a-card" style="overflow:hidden">
    <div style="display:flex;justify-content:space-between;padding:14px 18px 10px;font-weight:700;font-size:15px">Incoming payments<span style="color:${C.ink2};font-weight:400">Bank feed · live</span></div>
    <div style="display:grid;grid-template-columns:1.7fr .7fr 1fr 92px;align-items:center;column-gap:12px;padding:0 18px;height:30px;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#A39C92;border-top:1px solid #F0EAE0"><span>Customer</span><span>Method</span><span style="text-align:right">Amount</span><span style="text-align:right">Status</span></div>
    <div id="${p}-newrow" class="a-row" style="display:grid;grid-template-columns:1.7fr .7fr 1fr 92px;align-items:center;column-gap:12px;padding:0 18px;height:46px;border-top:1px solid #F0EAE0;background:#FFFCF6"><span style="font-weight:700;white-space:nowrap">Helix Health Distribution</span><span style="color:${C.ink2}">ACH</span><span class="num" style="text-align:right;font-weight:700;color:${C.bronzeD}">+$48,612.40</span><span class="a-pill" style="background:${C.blueL};color:${C.blueD};justify-self:end">Reading</span></div>
    <div class="a-row" style="display:grid;grid-template-columns:1.7fr .7fr 1fr 92px;align-items:center;column-gap:12px;padding:0 18px;height:46px;border-top:1px solid #F0EAE0"><span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Northwind Clinics</span><span style="color:${C.ink2}">ACH</span><span class="num" style="text-align:right">$12,904.00</span><span class="a-pill" style="background:#EEF2EA;color:#4F6B45;justify-self:end">Applied</span></div>
    <div class="a-row" style="display:grid;grid-template-columns:1.7fr .7fr 1fr 92px;align-items:center;column-gap:12px;padding:0 18px;height:46px;border-top:1px solid #F0EAE0"><span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Calder Medical Supply</span><span style="color:${C.ink2}">Wire</span><span class="num" style="text-align:right">$7,218.55</span><span class="a-pill" style="background:#EEF2EA;color:#4F6B45;justify-self:end">Applied</span></div>
    <div class="a-row" style="display:grid;grid-template-columns:1.7fr .7fr 1fr 92px;align-items:center;column-gap:12px;padding:0 18px;height:46px;border-top:1px solid #F0EAE0"><span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Ridgeway Health Partners</span><span style="color:${C.ink2}">ACH</span><span class="num" style="text-align:right">$5,880.00</span><span class="a-pill" style="background:#EEF2EA;color:#4F6B45;justify-self:end">Applied</span></div>
    <div class="a-row" style="display:grid;grid-template-columns:1.7fr .7fr 1fr 92px;align-items:center;column-gap:12px;padding:0 18px;height:46px;border-top:1px solid #F0EAE0"><span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Bayview Pharmacy Group</span><span style="color:${C.ink2}">Check</span><span class="num" style="text-align:right">$3,460.20</span><span class="a-pill" style="background:#EEF2EA;color:#4F6B45;justify-self:end">Applied</span></div>
    <div class="a-row" style="display:grid;grid-template-columns:1.7fr .7fr 1fr 92px;align-items:center;column-gap:12px;padding:0 18px;height:46px;border-top:1px solid #F0EAE0"><span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Lakeside Surgical</span><span style="color:${C.ink2}">Wire</span><span class="num" style="text-align:right">$2,145.90</span><span class="a-pill" style="background:#EEF2EA;color:#4F6B45;justify-self:end">Applied</span></div>
  </div>`;

// ---------- the office (world coordinates 1920x1080) ----------
// Monitor screen rect in world space: x 1180..1402, y 452..577 (222 x 125)
export const SCREEN = { x: 1180, y: 452, w: 222, h: 125 };
export const CHAIR_X = 1095;

export const office = (p, opts = {}) => `
<svg class="full" viewBox="0 0 1920 1080" style="position:absolute;left:0;top:0">
  <defs>
    <linearGradient id="${p}-wall" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F8F3EB"/><stop offset="1" stop-color="#F1E9DD"/></linearGradient>
    <linearGradient id="${p}-floor" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#E9DFCF"/><stop offset="1" stop-color="#E2D6C3"/></linearGradient>
    <linearGradient id="${p}-glass" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#EAF0F1"/><stop offset="1" stop-color="#F7F4EC"/></linearGradient>
    <linearGradient id="${p}-sun" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFDF7" stop-opacity=".85"/><stop offset="1" stop-color="#FFFDF7" stop-opacity="0"/></linearGradient>
    <radialGradient id="${p}-sh"><stop offset="0" stop-color="#4A3A2A" stop-opacity=".22"/><stop offset="1" stop-color="#4A3A2A" stop-opacity="0"/></radialGradient>
  </defs>
  <!-- ceiling, side walls, back wall, floor -->
  <polygon points="0,0 1920,0 1770,30 150,30" fill="#FAF6EF"/>
  <polygon points="0,0 150,30 150,640 0,770" fill="#EFE7DB"/>
  <polygon points="1920,0 1770,30 1770,640 1920,770" fill="#F2EBE0"/>
  <rect x="150" y="30" width="1620" height="610" fill="url(#${p}-wall)"/>
  <polygon points="0,770 150,640 1770,640 1920,770 1920,1080 0,1080" fill="url(#${p}-floor)"/>
  <rect x="150" y="628" width="1620" height="12" fill="#E4D9C9"/>
  <!-- floor boards, subtle -->
  ${[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((i) => { const xb = 150 + i * 135; const xf = (xb - 960) * 2.2 + 960; return `<line x1="${xb}" y1="640" x2="${xf}" y2="1080" stroke="#DCCFBB" stroke-width="1.2" opacity=".55"/>`; }).join("")}
  <!-- windows -->
  ${[330, 700, 1070].map((x) => `
  <g>
    <rect x="${x - 8}" y="72" width="336" height="496" rx="4" fill="#E7DED0"/>
    <rect x="${x}" y="80" width="320" height="480" fill="url(#${p}-glass)"/>
    <path d="M${x} 470 L${x + 40} 430 L${x + 70} 450 L${x + 70} 380 L${x + 120} 380 L${x + 120} 440 L${x + 170} 410 L${x + 200} 430 L${x + 200} 350 L${x + 250} 350 L${x + 250} 450 L${x + 320} 420 L${x + 320} 560 L${x} 560 Z" fill="#DDE5E8" opacity=".75"/>
    <ellipse cx="${x + 70}" cy="520" rx="90" ry="60" fill="#D7DECB" opacity=".7"/>
    <ellipse cx="${x + 260}" cy="530" rx="80" ry="50" fill="#D3DBC6" opacity=".6"/>
    <rect x="${x + 157}" y="80" width="6" height="480" fill="#E7DED0"/>
    <rect x="${x}" y="230" width="320" height="6" fill="#E7DED0"/>
    <rect x="${x - 14}" y="560" width="348" height="12" rx="3" fill="#EAE2D5"/>
  </g>`).join("")}
  <!-- sunlight on the floor -->
  <g id="${p}-sun" opacity="${opts.sun ?? 0.55}">
    ${[330, 700, 1070].map((x) => `<polygon points="${x + 30},642 ${x + 330},642 ${x + 560},1000 ${x + 180},1000" fill="url(#${p}-sun)"/>`).join("")}
  </g>
  <!-- rug under Maya's desk -->
  <polygon points="850,716 1610,716 1720,900 760,900" fill="#E8DCCB"/>
  <polygon points="872,726 1592,726 1690,890 790,890" fill="none" stroke="#DCCBB4" stroke-width="3"/>
  <!-- wall clock: shows the time of day -->
  <g id="${p}-clock">
    <circle cx="1444" cy="182" r="31" fill="#FFFDF8" stroke="#3A3631" stroke-width="3.5"/>
    ${[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((i) => `<line x1="1444" y1="${182 - 26}" x2="1444" y2="${182 - (i % 3 ? 23 : 21)}" stroke="#3A3631" stroke-width="${i % 3 ? 1.4 : 2.4}" transform="rotate(${i * 30} 1444 182)"/>`).join("")}
    <line id="${p}-clkH" x1="1444" y1="182" x2="1444" y2="166" stroke="#2B2A28" stroke-width="3.4" stroke-linecap="round" transform="rotate(${((opts.clock ?? [8, 58])[0] % 12 + (opts.clock ?? [8, 58])[1] / 60) * 30} 1444 182)"/>
    <line id="${p}-clkM" x1="1444" y1="182" x2="1444" y2="159" stroke="#2B2A28" stroke-width="2.2" stroke-linecap="round" transform="rotate(${(opts.clock ?? [8, 58])[1] * 6} 1444 182)"/>
    <circle cx="1444" cy="182" r="2.6" fill="${C.bronze}"/>
  </g>
  <!-- pendant lamps -->
  ${[760, 1290].map((x) => `<line x1="${x}" y1="30" x2="${x}" y2="150" stroke="#8C8378" stroke-width="2"/><path d="M${x - 34} 176 Q${x} 132 ${x + 34} 176 Z" fill="${C.bronze}"/><ellipse cx="${x}" cy="176" rx="34" ry="5" fill="#E9D3B4"/>`).join("")}
  <!-- tall plant -->
  <g>
    <ellipse cx="236" cy="712" rx="60" ry="10" fill="url(#${p}-sh)"/>
    <path d="M206 640 L266 640 L258 710 L214 710 Z" fill="#D9CDBB"/>
    ${[[-60, 420], [-30, 380], [0, 360], [30, 390], [55, 430], [-45, 500], [45, 510], [-15, 450], [20, 470]].map(([dx, y], i) => `<path d="M236 640 Q${236 + dx * 0.4} ${(640 + y) / 2} ${236 + dx} ${y} Q${236 + dx * 1.4} ${y + 30} ${236 + dx * 0.6} ${y + 60} Z" fill="${i % 2 ? "#8FA287" : "#7D9276"}" opacity=".95"/>`).join("")}
  </g>
  <!-- ERP archive shelf (right of the desk) -->
  <g id="${p}-shelf">
    <ellipse cx="1612" cy="764" rx="150" ry="12" fill="url(#${p}-sh)"/>
    <rect x="1490" y="300" width="244" height="460" rx="4" fill="#EEE6D9" stroke="#E0D5C4" stroke-width="2"/>
    ${[300, 450, 600, 750].map((y) => `<rect x="1490" y="${y - 6}" width="244" height="10" fill="#E2D7C6"/>`).join("")}
    ${[[318, ["#E8D8C2", "#D9C7AE", "#E1E5E8", "#CFD8E1", "#E8D8C2", "#E6DFD3"]], [468, ["#D9C7AE", "#E6DFD3", "#CFD8E1", "#E8D8C2", "#D9C7AE", "#E1E5E8"]], [618, ["#E6DFD3", "#E8D8C2", "#D9C7AE", "#CFD8E1", "#E6DFD3", "#E8D8C2"]]].map(([y, cols], r) => cols.map((c, i) => `<rect id="${p}-bind-${r}-${i}" x="${1508 + i * 36}" y="${y}" width="30" height="${126 - (i % 3) * 6}" transform="translate(0 ${(i % 3) * 6})" rx="3" fill="${c}" stroke="#CDBFA9" stroke-width="1"/><rect x="${1514 + i * 36}" y="${y + 18 + (i % 3) * 6}" width="18" height="6" rx="1.5" fill="#fff" opacity=".8"/>`).join("")).join("")}
    <rect x="1574" y="270" width="40" height="30" rx="4" fill="#D9CDBB"/>
    <path d="M1594 270 Q1570 230 1556 236 Q1580 250 1590 270 M1594 270 Q1612 222 1630 228 Q1606 246 1598 270 M1594 270 Q1596 236 1590 214 Q1604 240 1598 270" fill="#7D9276" stroke="#7D9276" stroke-width="5" stroke-linejoin="round"/>
    <text x="1612" y="290" font-family="Inter" font-size="11" font-weight="700" fill="#A3978A" text-anchor="middle" letter-spacing="2" opacity="0">ERP</text>
  </g>
  <!-- colleague's desk (background left) -->
  <g id="${p}-coldesk">
    <ellipse cx="540" cy="726" rx="170" ry="10" fill="url(#${p}-sh)"/>
    <polygon points="420,612 700,612 688,598 432,598" fill="#F4EDE2"/>
    <rect x="420" y="612" width="280" height="9" fill="#E3D9CA"/>
    <rect x="432" y="621" width="6" height="103" fill="#4A4744"/><rect x="682" y="621" width="6" height="103" fill="#4A4744"/>
    <polygon points="560,598 640,598 652,574 572,574" fill="#C9C4BC"/><polygon points="560,598 640,598 636,602 556,602" fill="#B2ACA2"/>
    <rect x="600" y="586" width="44" height="12" rx="2" fill="#E9E1D3"/>
    <path d="M444 576 L462 576 L460 598 L446 598 Z" fill="#7F96B1"/><path d="M462 582 q8 0 7 8 q-1 6 -8 5" stroke="#7F96B1" stroke-width="2.5" fill="none"/>
    <path d="M488 598 L506 598 L503 584 L491 584 Z" fill="#D9CDBB"/>
    <path d="M497 584 Q486 568 480 570 Q490 576 494 584 M497 584 Q508 564 516 566 Q504 574 500 584 M497 584 Q498 566 494 556 Q502 568 500 584" fill="#7D9276" stroke="#7D9276" stroke-width="3" stroke-linejoin="round"/>
  </g>
  ${opts.colleague ?? ""}
  <!-- Maya's desk -->
  <g id="${p}-desk">
    <ellipse cx="1225" cy="766" rx="300" ry="14" fill="url(#${p}-sh)"/>
    <rect x="975" y="608" width="7" height="152" fill="#4A4744"/><rect x="1468" y="608" width="7" height="152" fill="#4A4744"/>
    <rect x="1000" y="606" width="5" height="138" fill="#5A5653"/><rect x="1446" y="606" width="5" height="138" fill="#5A5653"/>
    <polygon points="980,578 1470,578 1490,598 960,598" fill="#F5EEE3"/>
    <rect x="960" y="598" width="530" height="11" fill="#E4D9C9"/>
    <!-- monitor -->
    <ellipse cx="1291" cy="590" rx="44" ry="5" fill="#4A3A2A" opacity=".18"/>
    <rect x="1284" y="560" width="14" height="26" fill="#BDB6AC"/>
    <path d="M1258 590 L1324 590 L1316 582 L1266 582 Z" fill="#CFC8BD"/>
    <rect x="${SCREEN.x - 5}" y="${SCREEN.y - 5}" width="${SCREEN.w + 10}" height="${SCREEN.h + 10}" rx="5" fill="#34312E"/>
    <!-- keyboard, mouse, notebook, notes, papers -->
    <polygon points="1214,586 1350,586 1356,594 1208,594" fill="#E9E5DE" stroke="#D6D0C6" stroke-width="1"/>
    <ellipse cx="1384" cy="590" rx="9" ry="5" fill="#E9E5DE" stroke="#D6D0C6" stroke-width="1"/>
    <polygon points="1100,584 1170,584 1176,594 1094,594" fill="${C.bronze}" opacity=".85"/>
    <polygon points="1104,583 1168,583 1172,588 1100,588" fill="#F7F1E6"/>
    <polygon points="1414,580 1470,580 1476,594 1408,594" fill="#FFFFFF" stroke="#E2DACD" stroke-width="1"/>
    <polygon points="1418,576 1468,576 1472,590 1412,590" fill="#FDFBF7" stroke="#E2DACD" stroke-width="1" transform="rotate(-3 1440 584)"/>
    <rect x="1420" y="570" width="18" height="16" fill="${C.amber}" opacity=".75" transform="rotate(-6 1429 578)"/>
    <rect x="1446" y="572" width="16" height="15" fill="#E9D7A8" transform="rotate(5 1454 580)"/>
    <polygon points="985,584 1015,584 1018,594 980,594" fill="#FDFBF7" stroke="#E2DACD" stroke-width="1"/>
    <rect x="1150" y="560" width="16" height="16" fill="#F0DFA9" transform="rotate(-4 1158 568)"/>
    <rect x="1153" y="540" width="15" height="15" fill="${C.amber}" opacity=".7" transform="rotate(6 1160 548)"/>
    <!-- desk lamp and a small plant -->
    <ellipse cx="1478" cy="579" rx="11" ry="3" fill="#3A3631"/>
    <path d="M1478 578 L1470 536 L1446 526" stroke="#3A3631" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M1432 534 L1447 520 L1457 530 Z" fill="${C.bronze}"/>
    <g transform="translate(-190 0)"><path d="M1186 594 L1204 594 L1202 580 L1188 580 Z" fill="#EDE5D8" stroke="#D8CDBD" stroke-width="1"/>
    <path d="M1195 580 Q1184 566 1178 568 Q1188 572 1192 580 M1195 580 Q1206 562 1212 565 Q1202 572 1198 580 M1195 580 Q1196 562 1192 554 Q1200 566 1198 580" fill="#7D9276" stroke="#7D9276" stroke-width="2.5" stroke-linejoin="round"/></g>
    <!-- coffee placed on the desk -->
    <g id="${p}-deskcup" opacity="${opts.cupOnDesk ? 1 : 0}">
      <ellipse cx="1040" cy="596" rx="16" ry="3.5" fill="#4A3A2A" opacity=".2"/>
      <path d="M1028 570 L1052 570 L1049 595 L1031 595 Z" fill="#FFFFFF" stroke="#D8CFC2" stroke-width="1.2"/>
      <rect x="1027" y="577" width="26" height="9" fill="${C.bronze}"/>
      <path d="M1035 566 q3 -6 0 -12 M1043 566 q3 -6 0 -12" stroke="#CFC6B9" stroke-width="1.5" fill="none" opacity=".7"/>
    </g>
  </g>
</svg>`;

// office chair seen from behind
export const chair = (p) => `
<g id="${p}-chair" transform="translate(${CHAIR_X} 798)">
  <ellipse cx="0" cy="2" rx="70" ry="9" fill="#4A3A2A" opacity=".16"/>
  <path d="M-52 -4 L52 -4 M0 -4 L-30 -10 M0 -4 L30 -10" stroke="#4A4744" stroke-width="6" stroke-linecap="round"/>
  <circle cx="-52" cy="-1" r="5" fill="#33302D"/><circle cx="52" cy="-1" r="5" fill="#33302D"/><circle cx="0" cy="-1" r="5" fill="#33302D"/>
  <rect x="-5" y="-82" width="10" height="78" fill="#5A5653"/>
  <g id="${p}-chair-seat">
    <rect x="-64" y="-98" width="128" height="18" rx="9" fill="#55514D"/>
    <rect x="-8" y="-130" width="16" height="40" fill="#4A4744"/>
    <rect x="-56" y="-180" width="112" height="92" rx="22" fill="#5C5853"/>
    <rect x="-50" y="-174" width="100" height="80" rx="18" fill="#67625C"/>
    <path d="M-40 -164 Q0 -172 40 -164" stroke="#7A746D" stroke-width="2" fill="none"/>
  </g>
</g>`;

// ---------- Maya close-up, 3/4 profile facing right (for reactions) ----------
export const mayaFace = (p) => `
<svg class="full" viewBox="0 0 1920 1080" style="position:absolute;left:0;top:0">
  <defs>
    <filter id="${p}-blur" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="14"/></filter>
    <linearGradient id="${p}-glow" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#FFFFFF" stop-opacity="0"/><stop offset="1" stop-color="#F3F7FB" stop-opacity=".55"/></linearGradient>
    <linearGradient id="${p}-bgw" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F7F1E8"/><stop offset="1" stop-color="#EEE5D7"/></linearGradient>
  </defs>
  <rect width="1920" height="1080" fill="url(#${p}-bgw)"/>
  <g filter="url(#${p}-blur)">
    <rect x="60" y="90" width="420" height="620" fill="#EEF2F2"/><rect x="262" y="90" width="12" height="620" fill="#E4DACB"/>
    <rect x="560" y="90" width="420" height="620" fill="#EFF2F1"/>
    <path d="M120 820 Q180 640 230 560 Q260 640 300 820 Z" fill="#8FA287" opacity=".8"/>
    <rect x="0" y="790" width="1920" height="290" fill="#E6DCCB"/>
    <ellipse cx="760" cy="240" rx="60" ry="22" fill="${C.bronze}" opacity=".7"/>
  </g>
  <!-- monitor edge on the right, lit screen -->
  <polygon points="1500,170 1920,120 1920,860 1500,800" fill="#F7F4EE"/>
  <polygon points="1470,166 1500,170 1500,800 1470,806" fill="#34312E"/>
  <rect x="1470" y="160" width="450" height="10" fill="#34312E" transform="rotate(-6.8 1470 160)"/>
  <g id="${p}-screenUI" opacity=".9">
    <rect x="1540" y="240" width="300" height="26" rx="6" fill="#ECE5D9" transform="skewY(-6.8)"/>
  </g>
  ${mayaFaceSVG(p)}
</svg>`;

// ---------- runtime helpers injected into each scene script ----------
export const HELPERS = `
gsap.config({force3D:false});
const $ = (s) => document.querySelector(s);
const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
const lerp = (a,b,t) => a + (b-a)*t;
const sm = (t) => { t = clamp(t,0,1); return t*t*(3-2*t); };
const eio = (t) => { t = clamp(t,0,1); return t<.5 ? 4*t*t*t : 1-Math.pow(-2*t+2,3)/2; };
const eout = (t) => { t = clamp(t,0,1); return 1-Math.pow(1-t,3); };
// keyed values: keys = [[t, v], ...] eased in-out between keys
const key = (t, keys, ease = eio) => {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    if (t <= keys[i][0]) { const [t0,v0] = keys[i-1], [t1,v1] = keys[i]; return lerp(v0, v1, ease((t-t0)/(t1-t0))); }
  }
  return keys[keys.length-1][1];
};
const rot = (id, a, cx, cy) => { const e = document.getElementById(id); if (e) e.setAttribute('transform', 'rotate(' + a.toFixed(3) + ' ' + cx + ' ' + cy + ')'); };
const setT = (id, s) => { const e = document.getElementById(id); if (e) e.setAttribute('transform', s); };
const setO = (id, o) => { const e = document.getElementById(id); if (e) e.setAttribute('opacity', o); };
// camera: put world point (cx,cy) at screen (sx,sy) with scale s
const camTo = (s, cx, cy, sx = 960, sy = 540) => ({ scale: s, x: sx - cx*s, y: sy - cy*s });
// per-word blur reveal
const revealWords = (tl, sel, t, gap = 0.08, dur = 0.5) => {
  tl.fromTo(sel, { opacity: 0, filter: 'blur(10px)', y: 14 }, { opacity: 1, filter: 'blur(0px)', y: 0, duration: dur, ease: 'expo.out', stagger: gap }, t);
};
const hideWords = (tl, sel, t, gap = 0.04) => {
  tl.to(sel, { opacity: 0, filter: 'blur(8px)', duration: 0.22, ease: 'power2.in', stagger: gap }, t);
};
// a clock that calls fn(t) for every frame of the scene, deterministic under seeking
const drive = (tl, dur, fn) => { const c = { t: 0 }; tl.fromTo(c, { t: 0 }, { t: dur, duration: dur, ease: 'none', onUpdate: () => fn(c.t) }, 0); fn(0); };
// number counter
const countTo = (tl, sel, from, to, t, dur, fmt) => { const o = { v: from }; tl.fromTo(o, { v: from }, { v: to, duration: dur, ease: 'power2.out', onUpdate: () => { const e = $(sel); if (e) e.textContent = fmt(o.v); } }, t); };
const money = (v) => '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
`;

// Maya side-view pose (walk cycle) – p = prefix
export const SIDE_POSE = (p) => `
const poseSide = (o) => {
  const ph = o.phase, a = o.amp;
  const s = Math.sin(ph), c = Math.cos(ph);
  const bob = 5*a*Math.abs(s);
  setT('${p}-ms', 'translate(' + o.x.toFixed(2) + ' ' + (o.y ?? 800) + ') scale(' + (o.sc ?? 1) + ')');
  setT('${p}-ms-body', 'translate(0 ' + bob.toFixed(2) + ') rotate(' + (2*a) + ' 0 -190)');
  rot('${p}-ms-legF', -21*a*s, 0, -196);
  rot('${p}-ms-legB', 21*a*s, 0, -196);
  rot('${p}-ms-legF-shin', a*(3 + 26*Math.pow(Math.max(0, c), 2)), 0, -104);
  rot('${p}-ms-legB-shin', a*(3 + 26*Math.pow(Math.max(0, -c), 2)), 0, -104);
  rot('${p}-ms-armB', -7*a*s + (o.armB ?? 4), 1, -295);
  rot('${p}-ms-foreB', -10, 0.5, -229);
  rot('${p}-ms-bag', 2.5*a*Math.sin(ph*2 + 0.6), -6, -296);
  const ua = (o.armF ?? -14) + 5*a*s, fa = (o.foreF ?? -72);
  rot('${p}-ms-armF', ua, 1, -295);
  rot('${p}-ms-foreF', fa, 0.5, -229);
  // cup follows the front hand and stays upright
  const r1 = ua*Math.PI/180, r2 = (ua+fa)*Math.PI/180;
  const ex = 1 - 66*Math.sin(r1), ey = -295 + 66*Math.cos(r1);
  const hx = ex - 66*Math.sin(r2), hy = ey + 66*Math.cos(r2);
  setT('${p}-ms-cup', 'translate(' + (hx+5+3).toFixed(2) + ' ' + (hy-5).toFixed(2) + ')');
  setO('${p}-ms-cup', o.cup ?? 1);
  rot('${p}-ms-head', (o.head ?? 0) + 1.2*a*Math.sin(ph*2), 5, -318);
  setO('${p}-ms', o.op ?? 1);
};`;

// Maya back-view pose. sit: 0 standing .. 1 seated. Arms in degrees.
export const BACK_POSE = (p) => `
const poseBack = (o) => {
  const sit = o.sit ?? 0;
  setT('${p}-mb', 'translate(' + o.x.toFixed(2) + ' 798)');
  setO('${p}-mb', o.op ?? 1);
  setT('${p}-mb-up', 'translate(0 ' + (88*sit + (o.bob ?? 0)).toFixed(2) + ') rotate(' + (o.lean ?? 0) + ' 0 -165)');
  setT('${p}-mb-legs', 'translate(0 ' + (40*sit).toFixed(2) + ') scale(1 ' + (1 - 0.55*sit).toFixed(3) + ')');
  setO('${p}-mb-legs', Math.max(0, 1 - sit*2.6));
  rot('${p}-mb-armL', o.armL ?? 4, -34, -294);
  rot('${p}-mb-foreL', o.foreL ?? 0, -34, -229);
  rot('${p}-mb-armR', o.armR ?? -4, 34, -294);
  rot('${p}-mb-foreR', o.foreR ?? 0, 34, -229);
  setO('${p}-mb-cupR', o.cup ?? 0);
  setO('${p}-mb-foreL', o.foreLVis ?? (1 - sit));
  setO('${p}-mb-foreR', o.foreRVis ?? (1 - sit));
  setO('${p}-mb-cupL', o.cupL ?? 0);
  setT('${p}-mb-head', 'rotate(' + (o.head ?? 0).toFixed(2) + ' 0 -318) translate(' + (o.headX ?? 0) + ' 0)');
};`;

// Out-of-focus foreground leaves: depth for the wide shots (off-frame in close shots).
export const foreground = (p) => `
<svg class="full" viewBox="0 0 1920 1080" style="position:absolute;left:0;top:0;pointer-events:none">
  <defs><filter id="${p}-fgblur" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="9"/></filter></defs>
  <g filter="url(#${p}-fgblur)" opacity=".92">
    ${[[-30, 1110, -10, 830, 60], [40, 1120, 120, 860, 40], [100, 1130, 230, 900, 20], [-60, 1100, 60, 760, 75], [150, 1140, 300, 960, 10]].map(([x1, y1, x2, y2, w], i) => `<path d="M${x1} ${y1} Q${(x1 + x2) / 2 - w} ${(y1 + y2) / 2} ${x2} ${y2} Q${(x1 + x2) / 2 + w} ${(y1 + y2) / 2 + 20} ${x1 + 40} ${y1} Z" fill="${i % 2 ? "#6E8467" : "#5F7559"}"/>`).join("")}
    ${[[1960, 1120, 1840, 880, -50], [1900, 1130, 1760, 930, -30], [1990, 1100, 1900, 820, -60]].map(([x1, y1, x2, y2, w], i) => `<path d="M${x1} ${y1} Q${(x1 + x2) / 2 - w} ${(y1 + y2) / 2} ${x2} ${y2} Q${(x1 + x2) / 2 + w} ${(y1 + y2) / 2 + 20} ${x1 - 40} ${y1} Z" fill="${i % 2 ? "#6E8467" : "#5F7559"}"/>`).join("")}
  </g>
</svg>`;
// The office "world" with Maya seated at her desk and the monitor showing screenHTML (1280x720 app).
export const screenCSS = (p) => `
#${p}-screen{left:${SCREEN.x}px;top:${SCREEN.y}px;width:${SCREEN.w}px;height:${SCREEN.h}px;overflow:hidden;background:#FBF9F5}
#${p}-screen .app{transform:scale(${SCREEN.w / 1280})}`;
export const worldLayer = (p, screenHTML, opts = {}) => `
<div id="${p}-world" class="world">
  ${office(p, { colleague: colleague(p), cupOnDesk: opts.cupOnDesk ?? true, sun: opts.sun, clock: opts.clock })}
  ${opts.overSVG ?? ""}
  <svg class="full" viewBox="0 0 1920 1080" style="position:absolute;left:0;top:0">
    ${mayaBack(p)}
    ${chair(p)}
    ${opts.side ? mayaSide(p) : ""}
  </svg>
  <div id="${p}-screen" class="abs">${screenHTML}</div>
  ${opts.extra ?? ""}
  ${opts.fg === false ? "" : foreground(p)}
</div>`;
// camera helper (runtime): sets the world transform so world point (cx,cy) sits at frame centre
export const CAM = (p) => `
const world = document.getElementById('${p}-world');
const cam = (s, cx, cy, sx = 960, sy = 540) => { cx = clamp(cx, sx / s, 1920 - (1920 - sx) / s); cy = clamp(cy, sy / s, 1080 - (1080 - sy) / s); world.style.transform = 'translate(' + (sx - cx * s).toFixed(2) + 'px,' + (sy - cy * s).toFixed(2) + 'px) scale(' + s.toFixed(4) + ')'; };
// seated, typing pose
const seated = (t, extra = {}) => poseBack(Object.assign({ x: ${CHAIR_X}, op: 1, sit: 1, armR: -12, foreR: -18, armL: 10, foreL: 14, head: 4 + 1.2 * Math.sin(t * 0.9) }, extra));
`;

// The customer's remittance advice as a paper document (480 x 800)
export const REMIT_ROWS = [
  ["INV-55141", "12,480.00", "249.60", "12,230.40"],
  ["INV-55188", "4,250.00", "85.00", "4,165.00"],
  ["INV-55203", "1,000.00", "0.00", "881.60"],
  ["INV-54877", "640.00", "0.00", "640.00"],
  ["CB-40112", "", "", "-326.50"],
  ["CB-40113", "", "", "-189.25"],
  ["CB-39870 RPY", "", "", "212.75"],
];
export const remitPDF = (p) => `
<div style="position:absolute;inset:0;background:#FFFEFB;border-radius:6px;box-shadow:0 2px 4px rgba(70,50,30,.08),0 24px 60px rgba(70,50,30,.14);padding:38px 36px;font-family:'IBM Plex Mono',monospace;color:#5E5952;overflow:hidden">
  <div style="font-family:Inter;font-weight:700;font-size:13px;letter-spacing:.14em;color:#8E877D">HELIX HEALTH DISTRIBUTION</div>
  <div style="font-family:Inter;font-weight:700;font-size:30px;letter-spacing:-.02em;color:${C.ink};margin-top:14px">Remittance advice</div>
  <div style="font-family:Inter;font-size:19px;color:${C.ink};margin-top:4px">#HX-778120 &nbsp;·&nbsp; Oct 26, 2026</div>
  <div id="${p}-pdfmeta" style="font-family:Inter;font-size:16px;margin-top:16px;padding:10px 12px;border-radius:8px;background:${C.bronzeL};color:${C.bronzeD};font-weight:700;line-height:1.5">64 payment lines<br>3 related customer accounts</div>
  <div style="font-size:12.5px;margin-top:20px;display:flex;color:#A39C92;border-bottom:1px solid #EEE8DE;padding-bottom:6px"><span style="flex:1.3">REF</span><span style="flex:1;text-align:right">GROSS</span><span style="flex:.8;text-align:right">DISC</span><span style="flex:1;text-align:right">NET</span></div>
  ${REMIT_ROWS.map((r, i) => `<div class="${p}-prow" style="font-size:13.5px;display:flex;padding:7px 0;border-bottom:1px dashed #F0EBE2"><span style="flex:1.3">${r[0]}</span><span style="flex:1;text-align:right">${r[1]}</span><span style="flex:.8;text-align:right">${r[2]}</span><span style="flex:1;text-align:right">${r[3]}</span></div>`).join("")}
  ${[0, 1, 2, 3, 4, 5, 6, 7].map((i) => `<div class="${p}-pbar" style="display:flex;gap:10px;padding:9px 0;border-bottom:1px dashed #F0EBE2"><span style="flex:1.3;height:8px;border-radius:3px;background:#EEE9E1"></span><span style="flex:1;height:8px;border-radius:3px;background:#F2EEE7"></span><span style="flex:.8;height:8px;border-radius:3px;background:#F2EEE7"></span><span style="flex:1;height:8px;border-radius:3px;background:#EEE9E1"></span></div>`).join("")}
  <div style="font-size:12.5px;color:#A39C92;padding:8px 0">… 56 more lines</div>
  <div class="${p}-ptotal" style="font-size:15px;display:flex;padding:9px 0;border-top:1.5px solid #D9D2C6;color:${C.ink};font-weight:600"><span style="flex:1">TOTAL</span><span style="text-align:right">48,612.40</span></div>
</div>`;

// ---------- small line icons for sources (no brand marks) ----------
export const icon = {
  bank: (s = 24, c = "currentColor") => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none"><path d="M3 9.5 L12 4 L21 9.5 Z" fill="${c}"/><rect x="5" y="11" width="2.6" height="6.5" rx=".6" fill="${c}"/><rect x="10.7" y="11" width="2.6" height="6.5" rx=".6" fill="${c}"/><rect x="16.4" y="11" width="2.6" height="6.5" rx=".6" fill="${c}"/><rect x="3" y="18.6" width="18" height="2.2" rx=".8" fill="${c}"/></svg>`,
  mail: (s = 24, c = "currentColor") => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none"><rect x="2.5" y="5.5" width="19" height="13" rx="2.2" stroke="${c}" stroke-width="2"/><path d="M3.5 7 L12 13 L20.5 7" stroke="${c}" stroke-width="2" stroke-linejoin="round"/></svg>`,
  erp: (s = 24, c = "currentColor") => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none"><ellipse cx="12" cy="6" rx="8" ry="3" stroke="${c}" stroke-width="2"/><path d="M4 6 V18 C4 19.7 7.6 21 12 21 C16.4 21 20 19.7 20 18 V6 M4 12 C4 13.7 7.6 15 12 15 C16.4 15 20 13.7 20 12" stroke="${c}" stroke-width="2"/></svg>`,
};
