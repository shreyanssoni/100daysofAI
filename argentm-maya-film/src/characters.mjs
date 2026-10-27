// Characters: Maya Chen (AR manager) and Dev Patel (Sales). Hand-built SVG rigs.
// Proportions: Maya ~370 world px tall (~7.6 heads). All rigs share one palette so she reads
// the same in the walk, the over-the-shoulder shots and the close-ups.

export const P = {
  skin: "#E2B08E", skinS: "#CC9474", skinH: "#EFC6A7", lip: "#B8645A", blush: "#E58F7C",
  hair: "#2A1F1A", hairH: "#4A3A31",
  blazer: "#C49A6C", blazerS: "#A87F55", blazerH: "#D3AE84",
  knit: "#F4EEE4", knitS: "#E2D8C8",
  trousers: "#33333B", trousersS: "#26262D", trousersH: "#41414A",
  shoe: "#1F1B18", sole: "#5A4A3E",
  bag: "#8A5A3B", bagS: "#6F462D",
  cup: "#FFFFFF", sleeve: "#A9773F",
  // Dev
  dSkin: "#B9845F", dSkinS: "#9E6C4B", dHair: "#1E1915", shirt: "#9DB4CC", shirtS: "#839CB6", chino: "#3D4A5C", chinoS: "#323D4D",
};

// ---------------- Maya, 3/4 walking view facing right. Origin at feet. ----------------
const armShape = (sleeve, sleeveS, skin, cuff) => `
  <path d="M-7.5 -297 C-10 -272 -9 -250 -7 -228 L8.5 -228 C10.5 -250 11.5 -272 10.5 -297 C6 -301 -3 -301 -7.5 -297 Z" fill="${sleeve}"/>
  <path d="M-7.5 -297 C-10 -272 -9 -250 -7 -228 L-1 -228 C-3 -250 -3 -275 -1 -299 Z" fill="${sleeveS}" opacity=".7"/>
  <g class="fore">
    <path d="M-7 -231 L8.5 -231 L7.2 -180 L-5.6 -180 Z" fill="${sleeve}"/>
    <path d="M-7 -231 L-1.5 -231 L-0.5 -180 L-5.6 -180 Z" fill="${sleeveS}" opacity=".7"/>
    <rect x="-5.8" y="-181" width="13.2" height="6" rx="2" fill="${cuff}"/>
    <path d="M-5 -175.5 C-6.5 -166 -5 -157 0.5 -154 C5.5 -153 8.5 -158 8 -165 L7 -175.5 Z" fill="${skin}"/>
    <path d="M6.5 -170 C9.5 -168 10.5 -164 9 -161" stroke="${skin}" stroke-width="3.4" stroke-linecap="round" fill="none"/>
  </g>`;

const legShape = (tr, trS, id) => `
  <g id="${id}">
    <path d="M-16 -200 C-16 -170 -15 -135 -13.5 -102 L14 -102 C15.5 -135 16.5 -170 16 -200 Z" fill="${tr}"/>
    <path d="M6 -198 C7 -165 8 -130 8 -102 L14 -102 C15.5 -135 16.5 -170 16 -200 Z" fill="${trS}" opacity=".55"/>
    <g id="${id}-shin">
      <path d="M-13.5 -106 L14 -106 C16 -72 18.5 -40 19.5 -9 L-17 -9 C-16.5 -40 -15 -72 -13.5 -106 Z" fill="${tr}"/>
      <path d="M2 -104 C2.5 -70 2.5 -40 2 -10" stroke="${P.trousersH}" stroke-width="1.4" opacity=".6"/>
      <path d="M-9 -9 C-4 -12 18 -11 29 -4.5 C32.5 -2.5 31.5 1.5 27.5 1.5 L-9.5 1.5 C-12.5 1.5 -12 -7 -9 -9 Z" fill="${P.shoe}"/>
      <path d="M-9.5 1.5 L27.5 1.5" stroke="${P.sole}" stroke-width="1.6"/>
    </g>
  </g>`;

export const mayaSide = (p) => `
<g id="${p}-ms" transform="translate(-200 800)">
  <ellipse cx="2" cy="2" rx="52" ry="7" fill="#4A3A2A" opacity=".16"/>
  <g id="${p}-ms-body">
    <!-- tote bag on the far shoulder -->
    <g id="${p}-ms-bag">
      <path d="M-6 -296 C-20 -270 -30 -240 -34 -214" stroke="${P.bagS}" stroke-width="3.2" fill="none" stroke-linecap="round"/>
      <path d="M-56 -216 L-14 -216 L-11 -170 C-11 -163 -15 -160 -21 -160 L-49 -160 C-55 -160 -59 -164 -58 -171 Z" fill="${P.bag}"/>
      <path d="M-56 -216 L-44 -216 L-46 -160 L-49 -160 C-55 -160 -59 -164 -58 -171 Z" fill="${P.bagS}"/>
      <rect x="-50" y="-223" width="32" height="8" rx="1.5" fill="#C9C6C0"/>
    </g>
    <g transform="translate(-6 0)"><g id="${p}-ms-armB">${armShape(P.blazerS, "#92704B", P.skinS, P.knitS).replace('class="fore"', `id="${p}-ms-foreB"`)}</g></g>
    ${legShape(P.trousersS, "#1E1E24", `${p}-ms-legB`)}
    <!-- hips / waistband -->
    <path d="M-22 -204 L24 -204 L23 -186 L-22 -186 Z" fill="${P.trousers}"/>
    ${legShape(P.trousers, P.trousersS, `${p}-ms-legF`)}
    <!-- blazer -->
    <path d="M-21 -300 C-9 -307 14 -307 26 -299 C28 -280 27 -262 25 -246 C27 -226 31 -205 33 -184 C14 -180 -8 -180 -26 -184 C-25 -205 -22 -226 -20 -246 C-23 -264 -25 -284 -21 -300 Z" fill="${P.blazer}"/>
    <path d="M-21 -300 C-25 -284 -23 -264 -20 -246 C-22 -226 -25 -205 -26 -184 C-20 -182.5 -14 -181.5 -9 -181 C-11 -226 -10 -268 -7 -305 Z" fill="${P.blazerS}"/>
    <path d="M12 -303 C20 -298 24 -290 24 -280 L22 -262 C17 -270 13 -284 11 -300 Z" fill="${P.knit}"/>
    <path d="M11.5 -301 L21 -264 L15.5 -250" stroke="${P.blazerS}" stroke-width="2" fill="none" stroke-linejoin="round"/>
    <path d="M24 -276 L33 -184" stroke="${P.blazerS}" stroke-width="1.2" opacity=".8"/>
    <path d="M-8 -232 L10 -232" stroke="${P.blazerS}" stroke-width="1.4" opacity=".6"/>
    <!-- neck and collar -->
    <path d="M-0.5 -322 L10.5 -322 L12 -302 L-2 -302 Z" fill="${P.skinS}"/>
    <path d="M-6 -309 C1 -314 13 -314 17 -307 L17 -299 C9 -303 -1 -303 -7 -299 Z" fill="${P.knit}"/>
    <path d="M-6 -305 C2 -309 12 -309 17 -303" stroke="${P.knitS}" stroke-width="1.2" fill="none"/>
    <!-- head -->
    <g id="${p}-ms-head"><g transform="translate(5 -316) scale(.86) translate(-5 316)">
      <path d="M-14 -350 C-16 -364 2 -373 15 -367 C24 -362 27 -351 26 -343 L28.8 -336 C29.6 -334 28.4 -332.6 26.4 -332.4 C26.6 -328 25 -322.5 20.5 -319 C14 -314.5 5 -314.5 -1 -319 C-9 -324.5 -13.5 -335 -14 -350 Z" fill="${P.skin}"/>
      <path d="M19 -330 C21 -325 19 -320 15 -317.5" stroke="${P.skinS}" stroke-width="1.5" fill="none" opacity=".7"/>
      <ellipse cx="18" cy="-333" rx="4" ry="2.4" fill="${P.blush}" opacity=".35"/>
      <!-- hair: back volume, low bun, side-swept fringe -->
      <path d="M-17 -334 C-25 -358 -8 -378 12 -374 C27 -371 32 -357 28 -347 C20 -357 7 -359 -1 -352 C-6 -346 -8 -337 -9 -326 C-12 -324 -15 -328 -17 -334 Z" fill="${P.hair}"/>
      <path d="M1 -369 C14 -370 25 -362 28 -348 C21 -354 12 -357 3 -356 Z" fill="${P.hair}"/>
      <path d="M-8 -366 C2 -371 14 -370 22 -364" stroke="${P.hairH}" stroke-width="2" fill="none" stroke-linecap="round" opacity=".9"/>
      <circle cx="-19" cy="-326" r="10.5" fill="${P.hair}"/>
      <path d="M-26 -329 C-21 -333 -14 -331 -12 -325" stroke="${P.hairH}" stroke-width="1.6" fill="none" opacity=".8"/>
      <path d="M-5 -341 C-7 -337 -7 -332 -4 -329 C-1 -331 -1 -338 -5 -341 Z" fill="${P.skinS}"/>
      <circle cx="-4" cy="-327.5" r="1.6" fill="#C9A24A"/>
      <!-- features -->
      <path d="M8 -349 C11 -351 16 -351 19 -349.5" stroke="${P.hair}" stroke-width="1.9" fill="none" stroke-linecap="round"/>
      <path id="${p}-ms-eye" d="M10.5 -342.5 C12.5 -344.6 16.2 -344.6 18 -342.6 C16 -341.2 12.6 -341 10.5 -342.5 Z" fill="#2B211C"/>
      <path d="M24.5 -343 C25.5 -344 27 -344 27.4 -343" stroke="#2B211C" stroke-width="1.3" fill="none" stroke-linecap="round"/>
      <path d="M26 -341 C27.5 -337.5 28.6 -335 26.8 -333.4" stroke="${P.skinS}" stroke-width="1.3" fill="none" stroke-linecap="round"/>
      <path d="M18.5 -326.6 C21 -327.8 23.6 -327.5 25 -326.4 C23.4 -324.2 20.6 -324 18.5 -326.6 Z" fill="${P.lip}"/>
    </g></g>
    <g transform="translate(5 0)"><g id="${p}-ms-armF">${armShape(P.blazer, P.blazerS, P.skin, P.knit).replace('class="fore"', `id="${p}-ms-foreF"`)}</g></g>
    <g id="${p}-ms-cup">
      <path d="M-10.5 -15 L10.5 -15 L8.2 11 L-8.2 11 Z" fill="${P.cup}" stroke="#D7CEC1" stroke-width="1"/>
      <rect x="-10" y="-7" width="20" height="9" fill="${P.sleeve}"/>
      <path d="M-11.5 -18.5 L11.5 -18.5 L11 -14.5 L-11 -14.5 Z" fill="#E9E2D6"/>
    </g>
  </g>
</g>`;

// ---------------- Maya, back view (standing / seated). Origin at feet. ----------------
const backArm = (id, side, sleeve, sleeveS, skin) => {
  const s = side; // -1 left, +1 right
  const x0 = 34 * s;
  return `<g id="${id}">
    <path d="M${x0 - 9} -296 C${x0 - 11} -270 ${x0 - 10} -250 ${x0 - 8} -228 L${x0 + 8} -228 C${x0 + 10} -250 ${x0 + 11} -272 ${x0 + 9} -296 Z" fill="${sleeve}"/>
    <path d="M${x0 + 3 * s} -296 C${x0 + 6 * s} -270 ${x0 + 6 * s} -250 ${x0 + 5 * s} -228 L${x0 + 8 * s} -228 C${x0 + 10 * s} -250 ${x0 + 11 * s} -272 ${x0 + 9 * s} -296 Z" fill="${sleeveS}" opacity=".7"/>
    <g id="${id.replace("arm", "fore")}">
      <path d="M${x0 - 8} -231 L${x0 + 8} -231 L${x0 + 6.5} -180 L${x0 - 6.5} -180 Z" fill="${sleeve}"/>
      <rect x="${x0 - 6.5}" y="-181" width="13" height="6" rx="2" fill="${P.knit}"/>
      <path d="M${x0 - 6} -175.5 C${x0 - 7} -165 ${x0 - 4} -156 ${x0} -155 C${x0 + 4} -156 ${x0 + 7} -165 ${x0 + 6} -175.5 Z" fill="${skin}"/>
      <g id="${id.replace("arm", "cup")}" opacity="0" transform="translate(${x0} -158)"><path d="M-10 -14 L10 -14 L8 10 L-8 10 Z" fill="#fff" stroke="#D7CEC1" stroke-width="1"/><rect x="-9.6" y="-7" width="19.2" height="8" fill="${P.sleeve}"/></g>
    </g></g>`;
};
export const mayaBack = (p) => `
<g id="${p}-mb" transform="translate(1095 798)" opacity="0">
  <ellipse cx="0" cy="0" rx="50" ry="7" fill="#4A3A2A" opacity=".16"/>
  <g id="${p}-mb-legs">
    <path d="M-25 -196 C-25 -130 -24 -60 -26 -8 L-3 -8 C-2 -60 -2 -130 -1 -196 Z" fill="${P.trousers}"/>
    <path d="M1 -196 C2 -130 2 -60 3 -8 L26 -8 C24 -60 25 -130 25 -196 Z" fill="${P.trousersS}"/>
    <path d="M-24 -8 L-4 -8 L-4 2 C-10 4 -18 4 -24 2 Z" fill="${P.shoe}"/><path d="M4 -8 L24 -8 L24 2 C18 4 10 4 4 2 Z" fill="${P.shoe}"/>
  </g>
  <g id="${p}-mb-up">
    ${backArm(`${p}-mb-armL`, -1, P.blazerS, "#92704B", P.skinS)}
    ${backArm(`${p}-mb-armR`, 1, P.blazer, P.blazerS, P.skin)}
    <path d="M-38 -296 C-30 -304 30 -304 38 -296 C40 -270 34 -240 33 -184 C10 -180 -10 -180 -33 -184 C-34 -240 -40 -270 -38 -296 Z" fill="${P.blazer}"/>
    <path d="M-38 -296 C-40 -270 -34 -240 -33 -184 C-26 -182.5 -20 -181.5 -14 -181 C-18 -230 -22 -270 -20 -301 Z" fill="${P.blazerS}" opacity=".85"/>
    <path d="M0 -296 L0 -186" stroke="${P.blazerS}" stroke-width="1.6" opacity=".8"/>
    <path d="M-24 -290 C-28 -270 -27 -255 -24 -246 M24 -290 C28 -270 27 -255 24 -246" stroke="${P.blazerS}" stroke-width="1.3" fill="none" opacity=".7"/>
    <path d="M-7 -322 L7 -322 L8 -300 L-8 -300 Z" fill="${P.skinS}"/>
    <path d="M-16 -308 C-8 -313 8 -313 16 -308 L18 -298 C8 -302 -8 -302 -18 -298 Z" fill="${P.knit}"/>
    <path d="M-20 -300 C-8 -296 8 -296 20 -300 L24 -294 C8 -289 -8 -289 -24 -294 Z" fill="${P.blazerS}"/>
    <g id="${p}-mb-head"><g transform="translate(0 -318) scale(.86) translate(0 318)">
      <ellipse cx="-22.5" cy="-339" rx="4" ry="7" fill="${P.skinS}"/><ellipse cx="22.5" cy="-339" rx="4" ry="7" fill="${P.skinS}"/>
      <circle cx="-22.5" cy="-331" r="1.7" fill="#C9A24A"/><circle cx="22.5" cy="-331" r="1.7" fill="#C9A24A"/>
      <path d="M-22 -340 C-24 -364 -12 -374 0 -374 C12 -374 24 -364 22 -340 C21 -330 14 -322 0 -320 C-14 -322 -21 -330 -22 -340 Z" fill="${P.hair}"/>
      <path d="M-12 -368 C-4 -372 8 -371 14 -366 M-17 -356 C-12 -362 -4 -364 2 -362 M4 -366 C10 -364 16 -358 18 -350" stroke="${P.hairH}" stroke-width="1.8" fill="none" stroke-linecap="round" opacity=".85"/>
      <ellipse cx="0" cy="-324" rx="13" ry="11.5" fill="${P.hair}"/>
      <path d="M-10 -327 C-4 -333 6 -332 11 -325 M-8 -320 C-2 -324 5 -323 9 -318" stroke="${P.hairH}" stroke-width="1.6" fill="none" stroke-linecap="round" opacity=".85"/>
    </g></g>
  </g>
</g>`;

// ---------------- Dev Patel, seated side view facing right (background desk). ----------------
export const dev = (p) => `
<g id="${p}-col" transform="translate(500 724) scale(.9)">
  <ellipse cx="10" cy="2" rx="62" ry="6" fill="#4A3A2A" opacity=".14"/>
  <!-- chair -->
  <path d="M-30 -4 L40 -4 M5 -4 L5 -64" stroke="#55514D" stroke-width="5" stroke-linecap="round"/>
  <circle cx="-30" cy="-1" r="4.5" fill="#33302D"/><circle cx="40" cy="-1" r="4.5" fill="#33302D"/>
  <rect x="-30" y="-150" width="16" height="90" rx="7" fill="#5C5853"/>
  <rect x="-28" y="-72" width="74" height="12" rx="6" fill="#55514D"/>
  <!-- legs -->
  <path d="M-8 -98 C10 -102 40 -102 60 -96 C66 -92 66 -80 60 -76 C40 -76 10 -76 -8 -72 Z" fill="${P.chino}"/>
  <path d="M46 -86 C48 -60 49 -30 49 -8 L63 -8 C64 -32 64 -62 64 -88 Z" fill="${P.chinoS}"/>
  <path d="M46 -9 C54 -11 66 -10 74 -4 C76 -1 74 2 70 2 L46 2 Z" fill="#2A2622"/>
  <!-- torso -->
  <path d="M-18 -210 C-2 -216 18 -214 26 -206 C30 -170 30 -120 26 -92 L-14 -92 C-22 -130 -24 -175 -18 -210 Z" fill="${P.shirt}"/>
  <path d="M-18 -210 C-24 -175 -22 -130 -14 -92 L-4 -92 C-8 -130 -8 -176 -4 -213 Z" fill="${P.shirtS}"/>
  <path d="M10 -208 L12 -100" stroke="${P.shirtS}" stroke-width="1.3"/>
  <circle cx="12" cy="-180" r="1.4" fill="#fff"/><circle cx="12" cy="-158" r="1.4" fill="#fff"/><circle cx="12" cy="-136" r="1.4" fill="#fff"/>
  <path d="M-3 -226 L9 -226 L11 -208 L-5 -208 Z" fill="${P.dSkinS}"/>
  <path d="M-8 -214 L3 -205 L15 -214 L13 -208 L3 -200 L-6 -208 Z" fill="${P.shirtS}"/>
  <g id="${p}-col-head"><g transform="translate(4 -228) scale(.84) translate(-4 228)">
    <path d="M-14 -258 C-15 -276 0 -284 14 -279 C23 -275 26 -264 25 -256 L28 -249 C28.6 -247 27.4 -246 25.6 -246 C25.6 -240 24 -234 19 -231 C12 -226 2 -226 -4 -231 C-11 -237 -14 -246 -14 -258 Z" fill="${P.dSkin}"/>
    <path d="M-4 -231 C2 -226 12 -226 19 -231 C24 -234 25.6 -240 25.6 -246 L22 -246 C20 -240 14 -236 6 -236 C2 -238 -2 -240 -4 -242 Z" fill="${P.dHair}" opacity=".85"/>
    <path d="M-16 -256 C-20 -278 -2 -290 14 -284 C24 -281 29 -272 27 -262 C20 -270 10 -272 2 -268 C-4 -262 -8 -254 -12 -246 Z" fill="${P.dHair}"/>
    <path d="M-5 -255 C-7 -251 -7 -246 -4 -243 C-1 -245 -1 -252 -5 -255 Z" fill="${P.dSkinS}"/>
    <path d="M9 -262 C12 -264 17 -264 20 -262" stroke="${P.dHair}" stroke-width="2.4" fill="none" stroke-linecap="round"/>
    <ellipse cx="15" cy="-256" rx="2.2" ry="2.4" fill="#1E1915"/>
    <path d="M25 -254 C27 -251 28 -248 26 -247" stroke="${P.dSkinS}" stroke-width="1.4" fill="none"/>
    <path id="${p}-col-mouth" d="M16 -239 C19 -238 22 -238 24 -239.5" stroke="#5A3A2E" stroke-width="1.6" fill="none" stroke-linecap="round"/>
  </g></g>
  <!-- near arm -->
  <g id="${p}-col-arm">
    <path d="M-8 -206 C-12 -180 -10 -160 -6 -142 L8 -142 C10 -160 10 -184 8 -206 Z" fill="${P.shirt}"/>
    <g id="${p}-col-fore">
      <path d="M-5 -146 L8 -146 L44 -136 L42 -124 Z" fill="${P.dSkin}"/>
      <path d="M-6 -150 L9 -150 L12 -138 L-3 -136 Z" fill="${P.shirt}"/>
      <path d="M42 -138 C50 -138 54 -134 52 -128 C48 -124 42 -124 40 -127 Z" fill="${P.dSkin}"/>
      <path id="${p}-col-thumb" d="M48 -136 L50 -146" stroke="${P.dSkin}" stroke-width="4.5" stroke-linecap="round" opacity="0"/>
    </g>
  </g>
</g>`;

// ---------------- Maya close-up, 3/4 view facing right ----------------
export const mayaFaceSVG = (p) => `
  <g id="${p}-fig">
    <g id="${p}-torso">
      <!-- blazer shoulders -->
      <path d="M370 1240 L380 1080 C390 930 470 820 600 786 C650 772 700 752 726 726 L900 726 C930 752 990 770 1050 784 C1180 816 1250 930 1260 1080 L1270 1240 Z" fill="${P.blazer}"/>
      <path d="M380 1080 C390 930 470 820 600 786 C650 772 700 752 726 726 L760 732 C700 790 640 860 610 1080 Z" fill="${P.blazerS}"/>
      <path d="M1050 784 C1180 816 1250 930 1260 1080 L1180 1080 C1170 950 1120 860 1040 800 Z" fill="${P.blazerH}" opacity=".6"/>
      <!-- turtleneck front and lapels -->
      <path d="M730 728 C760 790 880 800 910 730 L970 1240 L680 1240 Z" fill="${P.knit}"/>
      <path d="M736 752 C770 810 870 816 904 754" stroke="${P.knitS}" stroke-width="3" fill="none"/>
      <path d="M700 744 L760 900 L716 960 L700 1240 L630 1240 L660 900 Z" fill="${P.blazerS}"/>
      <path d="M936 744 L880 900 L928 960 L960 1240 L1030 1240 L986 900 Z" fill="${P.blazer}"/>
      <path d="M936 744 L880 900 L928 960" stroke="${P.blazerS}" stroke-width="3" fill="none"/>
    </g>
    <g id="${p}-headg"><g transform="translate(-30 50) scale(1.06)">
      <!-- neck -->
      <path d="M760 560 C764 620 758 680 744 730 C790 770 880 768 900 730 C884 690 878 630 880 560 Z" fill="${P.skinS}"/>
      <path d="M770 640 C800 664 850 666 878 650 L880 600 C850 630 800 628 770 600 Z" fill="#B98063" opacity=".45"/>
      <!-- bun and back hair -->
      <circle cx="694" cy="556" r="62" fill="${P.hair}"/>
      <path d="M650 540 C670 510 712 506 740 524 M648 570 C676 548 716 548 744 566" stroke="${P.hairH}" stroke-width="4" fill="none" opacity=".7" stroke-linecap="round"/>
      <!-- face -->
      <path id="${p}-face" d="M752 352 C800 314 878 322 898 380 C906 402 908 424 914 444 L926 490 C930 499 925 506 914 507 C920 518 920 526 914 532 C918 541 914 551 905 555 C902 574 892 594 870 606 C842 621 806 618 780 598 C756 580 744 546 742 504 Z" fill="${P.skin}"/>
      <path d="M742 504 C744 546 756 580 780 598 C806 618 842 621 870 606 C850 612 818 610 794 590 C774 572 762 540 760 500 Z" fill="${P.skinS}" opacity=".55"/>
      <path d="M898 380 C906 402 908 424 914 444 L926 490 C930 499 925 506 914 507 C922 498 920 470 908 446 C898 424 896 400 898 380 Z" fill="${P.skinH}" opacity=".6"/>
      <ellipse cx="858" cy="512" rx="30" ry="16" fill="${P.blush}" opacity=".26"/>
      <!-- ear -->
      <path d="M770 470 C752 466 744 484 748 504 C752 524 764 534 778 530 Z" fill="${P.skinS}"/>
      <path d="M766 482 C758 488 758 506 766 516" stroke="#B98063" stroke-width="3" fill="none"/>
      <circle cx="764" cy="536" r="5.5" fill="#C9A24A"/>
      <!-- hair: crown volume, side-swept fringe, tucked behind the ear -->
      <path d="M730 520 C700 420 720 320 800 296 C866 278 914 316 914 360 C890 340 852 334 820 344 C796 360 784 396 778 430 C774 456 776 476 782 492 C768 500 752 506 744 512 C738 516 734 520 730 520 Z" fill="${P.hair}"/>
      <path d="M796 302 C850 294 902 318 914 358 C896 346 868 341 842 346 C822 351 804 362 790 376 Z" fill="${P.hair}"/>
      <path d="M760 360 C780 322 830 304 880 318 M748 400 C762 362 790 338 830 330 M800 304 C846 300 890 320 906 346" stroke="${P.hairH}" stroke-width="4" fill="none" stroke-linecap="round" opacity=".75"/>
      <!-- brows -->
      <path id="${p}-browN" d="M826 416 C840 404 862 403 880 412" stroke="#3A2C24" stroke-width="4.6" fill="none" stroke-linecap="round"/>
      <path id="${p}-browF" d="M901 411 C908 406 916 407 921 412" stroke="#3A2C24" stroke-width="3.6" fill="none" stroke-linecap="round"/>
      <!-- near eye -->
      <g id="${p}-eyeN">
        <path d="M830 452 C842 438 866 436 880 448 C868 460 846 462 830 452 Z" fill="#FBF7F2"/>
        <circle id="${p}-irisN" cx="860" cy="449" r="9" fill="#4A3226"/>
        <circle cx="860" cy="449" r="4.4" fill="#1E1611"/>
        <circle cx="863" cy="446" r="2.4" fill="#fff"/>
        <path d="M826 452 C840 434 868 432 884 447" stroke="#22190F" stroke-width="3.8" fill="none" stroke-linecap="round"/>
        <path d="M882 446 L890 441 M878 441 L884 434" stroke="#22190F" stroke-width="2.2" stroke-linecap="round"/>
        <path d="M836 462 C850 468 866 466 876 458" stroke="${P.skinS}" stroke-width="2" fill="none" opacity=".7"/>
      </g>
      <!-- far eye -->
      <g id="${p}-eyeF">
        <path d="M902 446 C908 440 918 440 922 445 C916 451 908 451 902 446 Z" fill="#FBF7F2"/>
        <circle cx="914" cy="445.5" r="4.6" fill="#4A3226"/>
        <path d="M900 446 C906 438 918 437 924 444" stroke="#22190F" stroke-width="3" fill="none" stroke-linecap="round"/>
      </g>
      <!-- nose and mouth -->
      <path d="M906 452 C913 468 920 482 923 490 C923 499 917 503 908 503" stroke="${P.skinS}" stroke-width="3.2" fill="none" stroke-linecap="round"/>
      <path d="M898 500 C902 506 910 507 914 503" stroke="#B98063" stroke-width="2.4" fill="none" stroke-linecap="round"/>
      <path id="${p}-mouth" d="M872 538 C884 534 898 533 910 536 C900 546 884 548 872 538 Z" fill="${P.lip}"/>
      <path id="${p}-mouthL" d="M872 538 C886 541 898 540 910 536" stroke="#8E463E" stroke-width="2" fill="none" stroke-linecap="round"/>
      <path d="M884 556 C892 558 900 556 904 552" stroke="${P.skinS}" stroke-width="2" fill="none" opacity=".6"/>
    </g></g>
    <g id="${p}-collar" transform="translate(0 30) scale(1 1.15) translate(0 -100)">      <path d="M738 700 C770 744 876 746 906 700 L914 742 C880 788 766 788 728 744 Z" fill="${P.knit}"/>
      <path d="M728 740 C766 786 880 786 914 738 L902 836 L748 832 Z" fill="${P.knit}"/>
      <path d="M742 724 C776 764 872 766 906 726" stroke="${P.knitS}" stroke-width="2.5" fill="none" opacity=".7"/>
    </g>
  </g>`;
