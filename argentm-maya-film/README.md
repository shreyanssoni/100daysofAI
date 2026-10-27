# A Morning With Maya — argentm.ai (60 s, 1920×1080, 30 fps)

2D animated product film built with [HyperFrames](https://hyperframes.heygen.com) (HTML + GSAP rendered to video).

- **Final render:** `renders/maya-60s-1080p.mp4`
- **Edit:** scenes are generated from `src/` — change a scene in `src/scenes/sN.mjs` (or the shared office, Maya rigs and UI system in `src/shared.mjs`), then `node src/build.mjs`.
- **Preview:** `npx hyperframes preview` · **Check:** `npx hyperframes check` · **Render:** `npx hyperframes render --quality looks --output renders/maya-60s-1080p.mp4`
- **Sound:** `python3 src/score.py` regenerates the original score and foley (`assets/audio/`); UI sounds in `assets/sfx/` are HyperFrames' bundled library. Cue sheet: `src/audio-cues.mjs`.

| Time | Scene | File |
| --- | --- | --- |
| 0:00–0:07 | The morning — Maya arrives, sits, wakes her monitor | `s1` |
| 0:07–0:15 | The complex payment — deposit, remittance PDF, three fragmented sources | `s2` |
| 0:15–0:23 | AI reads the PDF into a structured table; TOTAL recognised as a summary | `s3` |
| 0:23–0:30 | Three accounts → one parent; Maya's recognition; exact bank match | `s4` |
| 0:30–0:44 | Six decisions: matching, 2% discount, chargebacks, recovery, short pay, duplicate | `s5` |
| 0:44–0:51 | Maya approves the $0 invoice, keeps INV-55203 open; leans back | `s6` |
| 0:51–0:56 | Apply & post → records filed to the ERP shelf; audit trail | `s7` |
| 0:56–1:00 | Sunlit office, coffee, summary; logo end card | `s8` |

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
- **Logo:** the mark and wordmark are drawn in code — replace with the real argentm.ai logo files.
- **Product UI:** rebuilt as a plausible design, not captured from the real product — swap in real screens/labels where they differ.
- **Music:** synthesized in `src/score.py` because no music library was reachable from the build machine; a licensed or composed track can replace `assets/audio/score.m4a`.
