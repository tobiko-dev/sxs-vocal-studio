# sxs-vocal-studio

A bot for the two-drum rhythm game, driving an iPhone through macOS
**iPhone Mirroring**.

The rule it plays by: read the colour of the note at the front of the queue. Red
means tap the red drum, blue means the blue drum, both at once means both, and
the mirror ball means blue.

## How it works

**The queue is static.** This is the thing that shapes everything else, and it
isn't obvious from a screenshot. The notes do not scroll toward the drums. The
front note sits in a fixed slot until it is cleared, then the whole queue shifts
up by one over a short animation.

Measured on a 31-second gameplay recording:

- **90.5% of frames show zero vertical motion.** Nothing is moving.
- The front note stays pinned at the same position for the entire recording.
- Advances are discrete 4–8 frame animations at **irregular** intervals — driven
  by taps, not by a clock.

So there is no arrival to predict, no note velocity, and no lead time to
compensate for. The bot is a closed loop:

```
read the front slot → tap the drums it needs → wait for the slot to change → repeat
```

Waiting for the change is what absorbs mirroring latency. However slow the round
trip, the bot physically cannot run ahead of the game.

```
        ┌───────────────────────────────┐
        │      the queue (static)       │
        │   ○ blue          ● red       │
        │   ○ blue          ● red       │
        │        ┌───────────┐          │
        │        │front slot │          │  <- the only pixels read
        │        └───────────┘          │
        │   [blue drum]   [red drum]    │
        └───────────────────────────────┘
```

### Colour, not position

Which drum to press is decided by **colour**, never by lane position. Fever mode
tints the whole screen gold and, in some sections, rearranges the lanes — a
position-based detector would break there. Fever does not shift note hue, so one
set of colour gates covers every mode.

Detection is a list of **zones** sharing that one box. A zone is a box, a rule
for what counts as a hit inside it, and the drums to press:

| zone | match | taps | priority |
|---|---|---|---|
| `disco` | `blue` ≥ 1000 | blue | 20 |
| `both` | `red` **and** `blue` | red + blue | 10 |
| `red` | `red` | red | 0 |
| `blue` | `blue` | blue | 0 |

A firing zone claims the drums it taps and suppresses lower-priority zones whose
drums it already covers, so `both` never stacks with the separate `red` and
`blue` zones. Suppression needs *full* coverage: a zone tapping `red+blue` isn't
suppressed by one claiming only `red`.

The mirror ball needs no special case. Under the queue model it simply **is** the
front note when its turn comes, and it reads as pure blue there (r=0,
b=1230–2385 in the recording). That's indistinguishable by pixel count from an
ordinary blue note at 1444–1594, and doesn't need to be — both tap blue. The
`disco` zone is kept as an explicit, adjustable rule rather than a hidden
special case, and shares the front box so it costs nothing to capture.

### Detecting the advance

Colour cannot tell you the queue moved, because consecutive notes are often the
same colour. The bot keeps a small greyscale **signature** of the front slot
instead — the glyph shape, bubble position and lane all differ between notes even
when the colour doesn't.

A tap fires when the slot has been **still** and its **reading unchanged** for a
few frames, and the signature differs from the one at the last tap. Both
conditions are needed: the signature can go quiet while the outgoing note's
bubble still tints the box, which would otherwise let a transitional reading be
tapped. Measured frame-to-frame signature differences are cleanly bimodal — 87%
of frames under 3.0 (settled), 9% over 6.0 (mid-animation).

If the slot hasn't changed after `retry_ms`, the tap was lost and it fires again.

### Why it's fast

| stage | cost |
|---|---|
| full-ROI blob detection + tracking (rejected) | ~8 ms per colour |
| **zone classify, LUT + subsampled + deduplicated** | **0.074 ms** |

Four choices get it there:

1. **Only the front box is captured**, never the whole window, in one grab.
   Screen capture dominates the loop.
2. **No per-frame HSV conversion.** The gates are baked once into a 32768-entry
   lookup table keyed on the top 5 bits of each channel. Per frame it's one
   gather and one `bincount`.
3. **Subsampling by 3.** Notes are large blobs; discarding 8 of every 9 pixels
   costs nothing in separability and cuts the work ninefold.
4. **Deduplication by box.** All four zones share the front box, so those pixels
   are captured and counted once and the result fanned out.

Counts are **normalised to a reference resolution**. Raw counts scale with area,
so the same note giving 40px on the native phone gives ~10px in a 616×1336 mirror
window — without normalising, thresholds would silently stop firing whenever the
window was resized. One set of thresholds now holds at any size.

Taps run on a separate thread, so a held tap never blinds the detector.

## Setup

Requires macOS Sequoia or later, an Apple silicon Mac, and iOS 18+ — that's what
iPhone Mirroring needs.

```bash
pip install -r requirements.txt
```

Grant both permissions in **System Settings → Privacy & Security**:

- **Screen Recording** — for your terminal, to capture the box
- **Accessibility** — for your terminal, to post synthetic clicks

Then:

1. Open iPhone Mirroring and **leave the window where it is**. Coordinates are
   relative to a pinned window rect; move it and you must recalibrate.
2. Start the song so the playfield is on screen.
3. Pin the window:

   ```bash
   python -m vocalbot.cli calibrate window
   ```

   **Have the song playing when you run this.** The mirroring window has no
   title bar — it's drawn as a phone body with rounded corners and a shadow, so
   its bounds include margin that isn't screen content, and the padding varies
   with window size. Rather than guess an inset, the command finds the live
   screen by *what moves*: the game animates continuously, the desktop behind
   the window's margin doesn't. Differencing a few frames gives the content rect
   directly.

   It then prints the content aspect, which should land near **0.4600**, and the
   drum coverage, which should read 60–70% at both. If the game is paused,
   detection reports that nothing moved and falls back to the raw window bounds;
   pass `--no-auto` to skip detection entirely.

4. Place the boxes. The editor is live — it re-grabs every frame, so you can
   **drag boxes while the song plays** and watch the counts move against real
   notes:

   ```bash
   python -m vocalbot.cli calibrate zones
   python -m vocalbot.cli calibrate drums
   ```

   | | |
   |---|---|
   | drag inside a box | move it |
   | drag a handle | resize (8 handles) |
   | `TAB` | select the next zone |
   | arrows | nudge 1px |
   | `[` `]` | shrink / grow around the centre |
   | `+` `-` | adjust the thresholds this zone uses |
   | `m` / `t` | cycle match rule / tapped drums |
   | `n` / `x` | new zone / delete zone |
   | `e` | enable or disable |
   | `s` / `q` | save / quit |

   All four zones share one box, so `TAB` is how you reach the ones underneath.
   Pass `--image shot.png` to edit against a saved screenshot instead.

   If anything looks wrong, run the diagnostic before touching thresholds:

   ```bash
   python -m vocalbot.cli doctor
   ```

   See [Troubleshooting](#troubleshooting).

5. Check the signal without tapping:

   ```bash
   python -m vocalbot.cli probe
   ```

6. Dry run, then for real:

   ```bash
   python -m vocalbot.cli run --dry-run
   python -m vocalbot.cli run
   ```

## Validating against a recording

```bash
python -m vocalbot.cli replay gameplay.mp4
python -m vocalbot.cli replay gameplay.mp4 --overlay annotated.mp4
```

This runs the exact live decision path over a recording, so a config change can
be checked in seconds without touching the game. The number to tune against is
**read consistency** — how steadily the box reports one answer for the whole time
a note occupies the slot. A box reaching too far up catches the note behind and
flickers mid-note; the shipped box scores 0.997 on a signature-segmented sweep
and 0.881 measured between dispatches.

Note that a recording captures a *human* playing, so the pace in it is theirs.
Replay verifies each note is seen once and read correctly; it cannot measure how
fast the bot will go, because the queue only advances when someone taps.

## Troubleshooting

### The calibrator shows my desktop / wallpaper, not the phone

Nothing is broken — the detector is faithfully reading the wrong pixels. The
saved `window_rect` is pointing somewhere other than the mirroring window.

```bash
python -m vocalbot.cli doctor
```

That prints every iPhone Mirroring window it can see, the rect currently saved,
whether the two disagree, and saves `doctor-capture.png` showing exactly what is
being grabbed. Usual causes:

- **A placeholder rect was copied from `config.example.json`.** The example now
  ships `window_rect: null` for this reason; an invented rect looks plausible
  and captures a phone-shaped patch of desktop.
- **The window moved after calibration.** Coordinates are pinned; re-run
  `calibrate window`.
- **Calibrated while the game was paused.** Content detection needs motion. Run
  it with the song playing.
- **The wrong window was picked.** The app owns several windows, and a helper
  or menu-bar one can match first. `doctor` lists all candidates; the largest
  normal-layer window is the one used.
- **iPhone Mirroring isn't running or is behind another window.**

`calibrate window` and `doctor` both verify the grab by checking that **both
drums are visible where they should be** — a real game screen shows 60–70%
coverage at each drum target, versus under 25% for anything else. That is a much
more specific test than "does this look busy", which a detailed wallpaper passes
easily.

### The aspect ratio is wrong, or the drums read 0%

The window bounds are not the screen bounds. Don't reach for `--chrome` — there
is no title bar to trim, and trimming makes the aspect worse, not better. Re-run
`calibrate window` with the song **playing** so content detection can work.

### Zone labels overlap in the calibrator

Expected: all four zones share one box. Labels are stacked one line per zone,
and `TAB` cycles which is selected.

## Editing zones without the GUI

```bash
vocalbot zone list                          # boxes, thresholds, priorities
vocalbot zone list --image shot.png         # what each zone sees in a frame
vocalbot zone set blue --thresh-blue 250
vocalbot zone set red --rect 0.25,0.65,0.75,0.70
vocalbot zone add hold --rect 0.3,0.4,0.7,0.42 --match any --taps red+blue
vocalbot zone rm hold
```

## Known gaps

- **Never run live.** Everything is validated against stills and one recording.
  The round-trip latency, and therefore the real note rate, is unmeasured.
- **`retry_ms` is a guess.** It must sit above the true round trip or the bot
  double-taps, and below a stalled note or it hangs. 900ms is conservative;
  measure it once the bot has run.
- **No true multi-touch.** macOS has no public multi-touch injection API, so
  "both drums" is two taps `inter_tap_ms` apart. Hit windows are tens of ms wide
  so this should land; widen it first if doubles drop.
- **Window must stay pinned.** No continuous window tracking; moving the window
  invalidates the calibration. `doctor` detects the mismatch.
- **The bot doesn't know when to stop.** It has no notion of a pause, countdown
  or results screen and will keep tapping through them.
- **Only tap notes are modelled.** If any glyph is a hold or a flick, it needs a
  zone that behaves differently.

## Layout

```
vocalbot/
  config.py     zones, colour gates, geometry as screen fractions
  color.py      LUT classify, resolution normalising, slot signatures
  queuescan.py  advance detection and tap resolution
  capture.py    zone capture (union / per-zone) and mirror-window lookup
  calibrate.py  overlay rendering, count reports, the drum picker
  editor.py     draggable/resizable box editor (logic split from the GUI)
  tap.py        threaded tap dispatch via CGEventPost
  bot.py        the live loop
  replay.py     offline replay of the live decision path over a recording
  doctor.py     capture diagnostics: which window, what is actually grabbed
  cli.py        calibrate | zone | doctor | bench | probe | replay | run
tests/          85 tests: detection, queue scanner, editor geometry, diagnostics
assets/frames/  reference screenshots with known-correct answers
```

## Note

Automating input into a game will generally be against its terms of service, and
rhythm games in particular tend to flag improbable accuracy. Your call.
