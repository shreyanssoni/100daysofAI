// 0:54.6–1:00 — Resolution. Maya answers Dev; the office is calm; end card (no logo).
import { C, appShell, worldLayer, screenCSS, CAM, BACK_POSE } from "../shared.mjs";

const p = "s8";
const screen = appShell("Dashboard", `
  <div style="display:flex;gap:26px">
    <div style="width:330px">
      <div class="a-sub" style="font-size:16px">Today · cash application</div>
      <div style="font-size:78px;font-weight:700;letter-spacing:-.04em;margin-top:18px" class="num">142</div>
      <div style="font-size:22px;color:${C.ink2};margin-top:-4px">remittances today</div>
      <div style="font-size:78px;font-weight:700;letter-spacing:-.04em;margin-top:22px;color:${C.bronze}" class="num">97%</div>
      <div style="font-size:22px;color:${C.ink2};margin-top:-4px">applied without a touch</div>
      <div style="font-size:12px;color:#A39C92;margin-top:26px">Illustrative figures</div>
    </div>
    <div class="a-card" style="flex:1;padding:18px 20px;height:560px;position:relative">
      <div style="font-size:15px;font-weight:700;padding-bottom:12px;border-bottom:1px solid #F0EAE0">Dev Patel · Sales</div>
      <div style="margin-top:16px;max-width:400px;background:#F3EFE8;border-radius:16px 16px 16px 4px;padding:14px 16px;font-size:18px;line-height:1.4">Helix says they paid in full. Can you confirm before my 10:00 call?</div>
      <div id="${p}-reply" style="margin:16px 0 0 auto;max-width:470px;background:${C.bronze};color:#fff;border-radius:16px 16px 4px 16px;padding:16px 18px;font-size:19px;line-height:1.45">
        Confirmed: <b>$48,612.40</b> received and applied across HX-1001–1003.<br>Not quite in full: <b>$118.40</b> still open on INV-55203, and a <b>$640</b> duplicate is held as credit.
      </div>
      <div id="${p}-seen" style="text-align:right;font-size:13px;color:${C.ink2};margin-top:6px">Sent 9:14 AM</div>
    </div>
  </div>`);

export default {
  id: p,
  dur: 5.4,
  css: `
${screenCSS(p)}
#${p}-glow{background:radial-gradient(ellipse at 30% 30%, rgba(255,236,200,.55), rgba(255,236,200,0) 60%);opacity:0;pointer-events:none}
#${p}-bub{left:250px;top:300px;opacity:0}
#${p}-end{background:var(--ground);opacity:0}
#${p}-line{left:0;width:1920px;top:400px;text-align:center;font-size:96px;font-weight:700;letter-spacing:-.04em;color:${C.ink}}
#${p}-tag{left:0;width:1920px;top:540px;text-align:center;font-size:30px;color:${C.ink2};letter-spacing:-.01em}
#${p}-cta{left:960px;top:640px;transform:translateX(-50%)}
#${p}-note{left:0;width:1920px;top:1020px;text-align:center;font-size:16px;color:#A39C92}
`,
  html: `
${worldLayer(p, screen, { sun: 0.55, clock: [9, 14] })}
<div id="${p}-glow" class="full"></div>
<div id="${p}-bub" class="abs">
  <div class="card" style="position:relative;padding:18px 22px;width:440px;font-size:24px;line-height:1.35;border-radius:20px 20px 20px 6px">
    <div style="font-size:15px;color:${C.ink2};margin-bottom:4px"><b style="color:${C.ink}">Dev</b> · Sales</div>Perfect — I'll raise the <b>$118.40</b> on the call.
  </div>
</div>
<div id="${p}-end" class="full">
  <div id="${p}-line" class="abs">Complex cash. <span class="serif" style="color:${C.bronze};font-size:1.1em">Clear decisions.</span></div>
  <div id="${p}-tag" class="abs">argentm.ai · Finance AI for FinOps</div>
  <div id="${p}-cta" class="abs"><div class="btn btn-bronze" style="height:84px;padding:0 54px;font-size:32px">Book a demo</div></div>
  <div id="${p}-note" class="abs">Product data, customers and metrics shown are illustrative.</div>
</div>
`,
  js: `
${BACK_POSE(p)}
${CAM(p)}
drive(tl, 5.4, (t) => {
  // Maya sends the reply, then reaches for her coffee
  const reach = key(t, [[1.55, 0], [1.9, 1]]);
  const lift = key(t, [[1.95, 0], [2.45, 1]]);
  const vis = key(t, [[1.5, 0], [1.6, 1]]);
  seated(t, {
    armL: lerp(lerp(10, 14, reach), 18, lift), foreL: lerp(lerp(14, 70, reach), 150, lift),
    foreLVis: vis, cupL: t >= 1.9 ? 1 : 0, head: 4 + key(t, [[2.0, 0], [2.4, -3]]) + key(t, [[2.5, 0], [2.8, -8]]),
  });
  setO('${p}-deskcup', t >= 1.9 ? 0 : 1);
  cam(key(t, [[0, 5.6], [1.15, 5.6], [2.2, 1.05]]), key(t, [[0, 1291], [1.15, 1291], [2.2, 960]]), key(t, [[0, 525], [1.15, 525], [2.2, 560]]));
  // Dev reads it, looks over and gives a thumbs-up
  const look = key(t, [[1.9, 0], [2.2, 1]]);
  const up = key(t, [[2.05, 0], [2.35, 1]]);
  rot('${p}-col-head', -12 * look, 4, -226);
  rot('${p}-col-arm', -62 * up, 0, -204);
  rot('${p}-col-fore', -40 * up, 0, -146);
  setO('${p}-col-thumb', up > 0.6 ? 1 : 0);
});
tl.fromTo('#${p}-reply', { opacity: 0, y: 20, scale: 0.96, transformOrigin: '100% 100%' }, { opacity: 1, y: 0, scale: 1, duration: 0.45, ease: 'expo.out' }, 0.5);
tl.fromTo('#${p}-seen', { opacity: 0 }, { opacity: 1, duration: 0.3 }, 0.8);
tl.fromTo('#${p}-sun', { opacity: 0.55 }, { opacity: 1, duration: 1.4, ease: 'power2.inOut' }, 1.2);
tl.fromTo('#${p}-glow', { opacity: 0 }, { opacity: 1, duration: 1.4, ease: 'power2.inOut' }, 1.2);
tl.fromTo('#${p}-bub', { opacity: 0, y: 16, scale: 0.94, transformOrigin: '0 100%' }, { opacity: 1, y: 0, scale: 1, duration: 0.45, ease: 'expo.out' }, 2.15);
// end card
tl.fromTo('#${p}-end', { opacity: 0 }, { opacity: 1, duration: 0.45, ease: 'power2.inOut' }, 3.0);
tl.fromTo('#${p}-world', { opacity: 1 }, { opacity: 0, duration: 0.05 }, 3.46);
tl.to('#${p}-bub', { opacity: 0, duration: 0.05 }, 3.46);
tl.fromTo('#${p}-line', { opacity: 0, filter: 'blur(10px)', y: 12 }, { opacity: 1, filter: 'blur(0px)', y: 0, duration: 0.6, ease: 'expo.out' }, 3.25);
tl.fromTo('#${p}-tag', { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.5, ease: 'expo.out' }, 3.55);
tl.fromTo('#${p}-cta', { opacity: 0, y: 16 }, { opacity: 1, y: 0, duration: 0.5, ease: 'expo.out' }, 3.75);
tl.fromTo('#${p}-note', { opacity: 0 }, { opacity: 1, duration: 0.4 }, 3.9);
`,
};
