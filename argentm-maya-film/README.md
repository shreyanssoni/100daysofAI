# A Morning With Maya — argentm.ai (60 s, 1920×1080, 30 fps)

2D animated product film built with [HyperFrames](https://hyperframes.heygen.com) (HTML + GSAP rendered to video).

- **Final render:** `renders/maya-60s-1080p.mp4`
- **Edit:** scenes are generated from `src/` — change a scene in `src/scenes/sN.mjs` (or the shared office, Maya rigs and UI system in `src/shared.mjs`), then `node src/build.mjs`.
- **Preview:** `npx hyperframes preview` · **Check:** `npx hyperframes check` · **Render:** `npx hyperframes render --quality looks --output renders/maya-60s-1080p.mp4`
- **Sound:** `python3 src/score.py` regenerates the original score and foley (`assets/audio/`); UI sounds in `assets/sfx/` are HyperFrames' bundled library. Cue sheet: `src/audio-cues.mjs`.

**Story:** Maya arrives at 8:58. Dev from Sales messages: *"Helix says they paid in full. Can you confirm before my 10:00 call?"* The product works through the remittance, and each complex case is part of the answer. At 9:14 Maya replies ("not quite in full: $118.40 still open, a $640 duplicate held as credit") and Dev gives a thumbs-up across the office.

| Time | Scene | File |
| --- | --- | --- |
| 0:00–0:07 | Maya arrives (8:58), Dev waves, she sits and wakes her monitor | `s1` |
| 0:07–0:15 | Dev's question; the deposit and remittance arrive; three fragmented sources | `s2` |
| 0:15–0:22.5 | The PDF becomes a structured table; TOTAL recognised as a summary | `s3` |
| 0:22.5–0:29.5 | Three accounts → one parent; Maya's recognition; exact bank match | `s4` |
| 0:29.5–0:43 | Six decisions (matching, 2% discount, chargebacks, recovery, short pay, duplicate), with Dev's question pinned | `s5` |
| 0:43–0:50 | Maya approves the $0 invoice and keeps INV-55203 open | `s6` |
| 0:50–0:54.6 | Apply & post → records filed to the ERP shelf; audit trail | `s7` |
| 0:54.6–1:00 | Maya's reply, Dev's thumbs-up, end card (no logo) | `s8` |

Characters live in `src/characters.mjs` (Maya: side walk, back/seated, close-up; Dev: seated). `node src/harness.mjs out.png` renders them large for review.

## Reconciliation (all fictional, internally consistent)

| | |
| --- | --- |
| Applied to 59 invoices (net of $412.60 discounts) | $48,275.40 |
| Chargebacks CB-40112 + CB-40113 ($0 invoice) | −$515.75 |
| Chargeback recovery CB-39870 | +$212.75 |
| Unapplied credit (duplicate of paid INV-54877) | $640.00 |
| **Bank deposit** | **$48,612.40** |

INV-55203: invoiced $1,000.00, paid $881.60, **$118.40 kept open**. 64 remittance lines = 59 invoice lines + 1 duplicate + 2 chargebacks + 1 recovery + 1 TOTAL row.
End-card metrics (142 remittances, 97% no-touch) are illustrative.

## Placeholders to swap before release
- **Logo:** none by request; the end card names argentm.ai in plain text only.
- **Product UI:** rebuilt as a plausible design, not captured from the real product — swap in real screens/labels where they differ.
- **Music:** synthesized in `src/score.py` because no music library was reachable from the build machine; a licensed or composed track can replace `assets/audio/score.m4a`.
