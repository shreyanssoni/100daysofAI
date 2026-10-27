// Dev-only: renders the character rigs large for visual review.  node src/harness.mjs <out.png>
import { mayaSide, mayaBack, dev, mayaFaceSVG } from "./characters.mjs";
import { createRequire } from "node:module";
const { chromium } = createRequire("/opt/node22/lib/node_modules/")("playwright");
const out = process.argv[2];
const html = `<html><body style="margin:0;background:#F3ECE1">
<svg width="1920" height="1080" viewBox="0 0 1920 1080">
  <g transform="translate(150 -650) scale(2.1)">${mayaSide("a").replace('translate(-200 800)', 'translate(60 800)')}</g>
  <g transform="translate(-1500 -900) scale(2.1)">${mayaBack("b").replace('opacity="0"', '')}</g>
  <g transform="translate(-300 -750) scale(2.1)">${dev("c").replace('translate(500 724) scale(.9)','translate(560 760) scale(.9)')}</g>
  <g transform="translate(1180 520) scale(.5)">${mayaFaceSVG("d")}</g>
</svg></body></html>`;
const b = await chromium.launch(); const pg = await b.newPage({ viewport: { width: 1920, height: 1080 } });
await pg.setContent(html); await pg.screenshot({ path: out }); await b.close();
