// Audio placed on the root timeline. `t` is relative to the start of `scene` (absolute if no scene).
// Every UI sound has a visible cause on screen.
export const AUDIO = [
  { id: "score", file: "audio/score.m4a", t: 0, dur: 60, vol: 0.8, track: 2 },
  { id: "foley", file: "audio/fx.m4a", t: 0, dur: 60, vol: 0.9, track: 3 },
  // s2: Dev's question, the deposit, the remittance; Maya opens it
  { id: "sx-dev", scene: "s2", file: "sfx/notification.mp3", t: 0.37, dur: 2.4, vol: 0.2, track: 4 },
  { id: "sx-coin", scene: "s2", file: "sfx/whoosh-short.mp3", t: 1.2, dur: 0.57, vol: 0.16, track: 6 },
  { id: "sx-bank", scene: "s2", file: "sfx/pop.mp3", t: 1.8, dur: 0.7, vol: 0.18, track: 5 },
  { id: "sx-mail", scene: "s2", file: "sfx/whoosh-short.mp3", t: 2.42, dur: 0.57, vol: 0.13, track: 7 },
  { id: "sx-remit", scene: "s2", file: "sfx/pop.mp3", t: 2.95, dur: 0.7, vol: 0.16, track: 6 },
  { id: "sx-open", scene: "s2", file: "sfx/click-soft.mp3", t: 3.51, dur: 0.37, vol: 0.4, track: 5 },
  { id: "sx-unfold", scene: "s2", file: "sfx/whoosh-short.mp3", t: 3.62, dur: 0.57, vol: 0.18, track: 7 },
  // s3: lines lift off the page; 64 lines done; account ids lift out
  { id: "sx-lift", scene: "s3", file: "sfx/whoosh-short.mp3", t: 1.0, dur: 0.57, vol: 0.12, track: 4 },
  { id: "sx-done64", scene: "s3", file: "sfx/pop.mp3", t: 4.36, dur: 0.7, vol: 0.16, track: 5 },
  { id: "sx-ids", scene: "s3", file: "sfx/whoosh-short.mp3", t: 6.2, dur: 0.57, vol: 0.14, track: 4 },
  // s4: parent card
  { id: "sx-parent", scene: "s4", file: "sfx/pop.mp3", t: 2.26, dur: 0.7, vol: 0.14, track: 5 },
  // s5: decision log entries
  ...[2.25, 4.3, 6.4, 8.3, 10.15, 12.4].map((t, i) => ({ id: "sx-log" + i, scene: "s5", file: "sfx/click-soft.mp3", t: t + 0.02, dur: 0.37, vol: 0.18, track: 4 + (i % 2) })),
  { id: "sx-strip", scene: "s5", file: "sfx/whoosh-short.mp3", t: 3.66, dur: 0.57, vol: 0.12, track: 6 },
  { id: "sx-need", scene: "s5", file: "sfx/pop.mp3", t: 13.12, dur: 0.37, vol: 0.18, track: 6 },
  // s6: Maya's two clicks
  { id: "sx-approve", scene: "s6", file: "sfx/click.mp3", t: 2.31, dur: 0.37, vol: 0.32, track: 4 },
  { id: "sx-keep", scene: "s6", file: "sfx/click.mp3", t: 3.76, dur: 0.37, vol: 0.32, track: 5 },
  // s7: apply
  { id: "sx-apply", scene: "s7", file: "sfx/click.mp3", t: 0.81, dur: 0.37, vol: 0.38, track: 4 },
  { id: "sx-fly", scene: "s7", file: "sfx/whoosh-short.mp3", t: 1.8, dur: 0.57, vol: 0.14, track: 5 },
  // s8: Maya sends her reply; Dev's thumbs-up
  { id: "sx-send", scene: "s8", file: "sfx/click-soft.mp3", t: 0.55, dur: 0.37, vol: 0.35, track: 4 },
  { id: "sx-dev2", scene: "s8", file: "sfx/pop.mp3", t: 2.12, dur: 0.7, vol: 0.16, track: 5 },
];
