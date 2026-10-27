// 0:55.6–1:00 — Resolution and end card.
import { C, appShell, worldLayer, screenCSS, CAM, BACK_POSE } from "../shared.mjs";

const p = "s8";
const summary = appShell("Dashboard", `
  <div class="a-sub" style="font-size:18px">Today · Cash application</div>
  <div style="font-size:84px;font-weight:700;letter-spacing:-.04em;margin-top:30px" class="num">142</div>
  <div style="font-size:28px;color:${C.ink2};margin-top:-6px">remittances today</div>
  <div style="font-size:84px;font-weight:700;letter-spacing:-.04em;margin-top:30px;color:${C.bronze}" class="num">97%</div>
  <div style="font-size:28px;color:${C.ink2};margin-top:-6px">applied without a touch</div>
  <div style="position:absolute;right:30px;bottom:20px;font-size:13px;color:#A39C92">Illustrative figures</div>`);

export default {
  id: p,
  dur: 4.4,
  css: `
${screenCSS(p)}
#${p}-glow{background:radial-gradient(ellipse at 30% 30%, rgba(255,236,200,.55), rgba(255,236,200,0) 60%);opacity:0;pointer-events:none}
#${p}-end{background:${C.ivory};opacity:0}
#${p}-lock{left:960px;top:330px;transform:translateX(-50%);display:flex;align-items:center;gap:30px}
#${p}-wm{font-size:104px;font-weight:700;letter-spacing:-.045em;color:${C.ink};white-space:nowrap}
#${p}-tag{left:0;width:1920px;top:500px;text-align:center;font-size:26px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:${C.ink2}}
#${p}-line{left:0;width:1920px;top:580px;text-align:center;font-size:56px;font-weight:700;letter-spacing:-.03em;color:${C.ink}}
#${p}-cta{left:960px;top:700px;transform:translateX(-50%)}
#${p}-note{left:0;width:1920px;top:1020px;text-align:center;font-size:16px;color:#A39C92}
`,
  html: `
${worldLayer(p, summary, { sun: 0.55 })}
<div id="${p}-glow" class="full"></div>
<div id="${p}-end" class="full">
  <div id="${p}-lock" class="abs">
    <svg id="${p}-mark" width="150" height="150" viewBox="0 0 40 40" fill="none">
      <circle id="${p}-ring" cx="18" cy="21" r="12" stroke="${C.bronze}" stroke-width="2.6" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1" transform="rotate(-90 18 21)"/>
      <path id="${p}-stem" d="M30 9 V33" stroke="${C.bronze}" stroke-width="2.6" stroke-linecap="round" pathLength="1" stroke-dasharray="1 1.2" stroke-dashoffset="1.1"/>
      <path id="${p}-bar" d="M8 27 H30" stroke="${C.bronze}" stroke-width="2.6" stroke-linecap="round" pathLength="1" stroke-dasharray="1 1.2" stroke-dashoffset="1.1"/>
    </svg>
    <div id="${p}-wm">argentm<span style="color:${C.bronze}">.ai</span></div>
  </div>
  <div id="${p}-tag" class="abs">Finance AI for FinOps</div>
  <div id="${p}-line" class="abs">Complex cash. <span class="serif" style="color:${C.bronze};font-size:1.12em">Clear decisions.</span></div>
  <div id="${p}-cta" class="abs"><div class="btn btn-bronze" style="height:84px;padding:0 54px;font-size:32px">Book a demo</div></div>
  <div id="${p}-note" class="abs">Product data and metrics shown are illustrative.</div>
</div>
`,
  js: `
${BACK_POSE(p)}
${CAM(p)}
drive(tl, 4.4, (t) => {
  // Maya reaches for her coffee and lifts it
  const reach = key(t, [[0.15, 0], [0.55, 1]]);
  const lift = key(t, [[0.65, 0], [1.15, 1]]);
  seated(t, {
    armL: lerp(lerp(10, 18, reach), 22, lift), foreL: lerp(lerp(14, 70, reach), 150, lift),
    cupL: t >= 0.55 ? 1 : 0, head: 4 + key(t, [[0.8, 0], [1.2, -3]]),
  });
  setO('${p}-deskcup', t >= 0.55 ? 0 : 1);
  cam(key(t, [[0, 2.4], [1.7, 1.04]]), key(t, [[0, 1250], [1.7, 960]]), key(t, [[0, 520], [1.7, 540]]));
});
tl.fromTo('#${p}-sun', { opacity: 0.55 }, { opacity: 1, duration: 1.4, ease: 'power2.inOut' }, 0.1);
tl.fromTo('#${p}-glow', { opacity: 0 }, { opacity: 1, duration: 1.4, ease: 'power2.inOut' }, 0.1);
// warm-white flood, then the mark draws
tl.fromTo('#${p}-end', { opacity: 0 }, { opacity: 1, duration: 0.45, ease: 'power2.inOut' }, 1.5);
tl.to('#${p}-ring', { strokeDashoffset: 0, duration: 0.6, ease: 'power2.inOut' }, 1.75);
tl.to('#${p}-stem', { strokeDashoffset: 0, duration: 0.3, ease: 'power2.out' }, 2.15);
tl.to('#${p}-bar', { strokeDashoffset: 0, duration: 0.3, ease: 'power2.out' }, 2.3);
tl.fromTo('#${p}-wm', { clipPath: 'inset(0% 100% 0% 0%)', x: -20 }, { clipPath: 'inset(0% 0% 0% 0%)', x: 0, duration: 0.6, ease: 'power3.out' }, 2.25);
tl.fromTo('#${p}-tag', { opacity: 0, scale: 1.06 }, { opacity: 1, scale: 1, duration: 0.6, ease: 'expo.out' }, 2.55);
tl.fromTo('#${p}-line', { opacity: 0, filter: 'blur(8px)' }, { opacity: 1, filter: 'blur(0px)', duration: 0.4, ease: 'expo.out' }, 2.7);
tl.fromTo('#${p}-cta', { opacity: 0, y: 16 }, { opacity: 1, y: 0, duration: 0.5, ease: 'expo.out' }, 2.9);
tl.fromTo('#${p}-note', { opacity: 0 }, { opacity: 1, duration: 0.4 }, 3.0);
`,
};
