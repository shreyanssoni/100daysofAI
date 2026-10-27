// Audio placed on the root timeline (absolute seconds). Every UI sound has a visible cause.
export const AUDIO = [
  { id: "score", file: "audio/score.m4a", t: 0, dur: 60, vol: 0.8, track: 2 },
  { id: "foley", file: "audio/fx.m4a", t: 0, dur: 60, vol: 0.9, track: 3 },
  // scene 2: the payment and the remittance arrive, Maya opens it
  { id: "sx-notify", file: "sfx/notification.mp3", t: 8.0, dur: 2.4, vol: 0.22, track: 4 },
  { id: "sx-remit", file: "sfx/pop.mp3", t: 8.86, dur: 0.7, vol: 0.18, track: 5 },
  { id: "sx-open", file: "sfx/click-soft.mp3", t: 10.06, dur: 0.37, vol: 0.4, track: 6 },
  { id: "sx-unfold", file: "sfx/whoosh-short.mp3", t: 10.36, dur: 0.57, vol: 0.18, track: 5 },
  // scene 3: lines lift off the page; summary row docks
  { id: "sx-lift", file: "sfx/whoosh-short.mp3", t: 16.0, dur: 0.57, vol: 0.12, track: 4 },
  { id: "sx-done64", file: "sfx/pop.mp3", t: 19.36, dur: 0.7, vol: 0.16, track: 5 },
  { id: "sx-ids", file: "sfx/whoosh-short.mp3", t: 21.2, dur: 0.57, vol: 0.14, track: 4 },
  // scene 4: parent card
  { id: "sx-parent", file: "sfx/pop.mp3", t: 25.26, dur: 0.7, vol: 0.14, track: 5 },
  // scene 5: decisions log entries
  ...[32.25, 34.3, 36.4, 38.3, 40.15, 42.4].map((t, i) => ({ id: "sx-log" + i, file: "sfx/click-soft.mp3", t: t + 0.02, dur: 0.37, vol: 0.18, track: 4 + (i % 2) })),
  { id: "sx-strip", file: "sfx/whoosh-short.mp3", t: 33.66, dur: 0.57, vol: 0.12, track: 4 },
  { id: "sx-need", file: "sfx/pop.mp3", t: 43.42, dur: 0.7, vol: 0.18, track: 5 },
  // scene 6: Maya's two clicks
  { id: "sx-approve", file: "sfx/click.mp3", t: 46.31, dur: 0.37, vol: 0.32, track: 4 },
  { id: "sx-keep", file: "sfx/click.mp3", t: 47.76, dur: 0.37, vol: 0.32, track: 5 },
  // scene 7: apply
  { id: "sx-apply", file: "sfx/click.mp3", t: 51.81, dur: 0.37, vol: 0.38, track: 4 },
  { id: "sx-fly", file: "sfx/whoosh-short.mp3", t: 52.8, dur: 0.57, vol: 0.14, track: 5 },
];
