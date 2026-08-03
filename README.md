# sxs-vocal-studio

A zone-based bot for the two-drum rhythm game, driving an iPhone through macOS
**iPhone Mirroring**.

The rule it plays by: read the colour of the notes coming down the runway. Red
means tap the red drum, blue means the blue drum, both at once means both, and
the mirror ball means blue.

## How it works

Notes travel down a runway toward two drums. Their **colour**, not their lane
position, decides which drum to hit — which matters because fever mode collapses
the two lanes into one centred lane, where a position-based detector would break.
Fever also tints the whole screen gold, but it does not shift note hue, so one
set of colour gates covers both modes.

Detection is a list of **zones**. A zone is a box, a rule for what counts as a
hit inside it, and the drums to press. Nothing about the layout is hardcoded —
move, retune, add or disable zones with `vocalbot calibrate zones` or
`vocalbot zone set`.

```
        ┌───────────────────────────────┐
        │        note runway            │
        │   ○ blue          ● red       │
   ═════╪═══ lane box: red / blue / both╪═══
        │       ● red                   │
        │        ┌────────┐             │
        │        │ disco  │             │   <- small box inside the ball's path
        │        └────────┘             │
        │   [blue drum]   [red drum]    │
        └───────────────────────────────┘
```

The four default zones, highest priority first:

| zone | box | match | taps | why |
|---|---|---|---|---|
| `disco` | small centre box | `blue` ≥ 300 | blue | the mirror ball |
| `both` | lane strip | `red` **and** `blue` | red + blue | simultaneous notes |
| `red` | lane strip | `red` | red | |
| `blue` | lane strip | `blue` | blue | |

### Priority resolution

Zones overlap on purpose, so a firing zone **claims the drums it taps and
suppresses lower-priority zones whose taps it already covers**. That's what stops
`both` from double-tapping alongside the separate `red` and `blue` zones, and
stops the mirror ball — which also reads as blue in the lane strip — from tapping
blue twice. A suppressed zone still advances its edge state, so it can't fire
late on the next frame.

Suppression needs *full* coverage: a zone tapping `red+blue` is not suppressed by
one that only claims `red`.

### Why the lane box sits high

Measured on the reference frames: below about 60% of screen height the frontmost
note glyphs are large enough to **overlap into one continuous run**, so two
consecutive same-colour notes merge into a single detection and the bot taps once
where it should tap twice. Higher up the glyphs separate with clean gaps.

Placing it high also buys lead time, which is what absorbs mirroring latency. The
lane box's vertical position is therefore the single latency knob — raise it to
tap earlier, lower it to tap later.

The cost is signal strength: a note is ~150–1100 glyph pixels up there versus
~2500 at the bottom, against a noise floor around 20 — which is why the
thresholds are low. The disco box is the opposite case: kept small and placed
inside the ball's path, it reads **1113 blue against at most 2** on every other
frame. Widening it to cover the ball's full extent drops that margin to 5×.

### Why it's fast

| stage | cost |
|---|---|
| full-ROI blob detection + tracking (rejected) | ~8 ms per colour |
| **zone classify, LUT + subsampled + deduplicated** | **0.074 ms** |

Four choices get it there:

1. **Only the zones are captured**, never the whole window, and in one grab
   rather than one per zone. Screen capture dominates the loop.
2. **No per-frame HSV conversion.** The HSV gates are baked once into a
   32768-entry lookup table keyed on the top 5 bits of each channel. Per frame
   it's one gather and one `bincount`.
3. **Subsampling by 3.** Notes are large blobs; discarding 8 of every 9 pixels
   costs nothing in separability and cuts the work ninefold.
4. **Deduplication by box.** Three zones share the lane strip, so those pixels
   are captured and counted once and the result fanned out — half the classify
   cost on its own.

Taps run on a separate thread, so a held tap never blinds the detector.

## Setup

Requires macOS Sequoia or later, an Apple silicon Mac, and iOS 18+ — that's what
iPhone Mirroring needs.

```bash
pip install -r requirements.txt
```

Grant both permissions in **System Settings → Privacy & Security**, or nothing
will work:

- **Screen Recording** — for your terminal, to capture the zones
- **Accessibility** — for your terminal, to post synthetic clicks

Then:

1. Open iPhone Mirroring and **leave the window where it is**. Coordinates are
   captured relative to a pinned window rect; move it and you must recalibrate.
2. Start the song so the playfield is on screen.
3. Pin the window:

   ```bash
   python -m vocalbot.cli calibrate window --chrome 28
   ```

   `--chrome` trims the title bar. The command prints the content aspect ratio —
   it should be close to **0.4600**. A big mismatch means the trim is wrong or
   the window is letterboxed. Zone rects are fractions of the phone screen, so a
   uniformly scaled window needs no other changes.

4. Place the zones by dragging boxes:

   ```bash
   python -m vocalbot.cli calibrate zones
   python -m vocalbot.cli calibrate drums
   ```

   Both accept `--image shot.png` to calibrate against a saved screenshot instead
   of a live grab, and `--only red blue` to edit a subset. Each writes
   `calibration.png` showing every box with its live counts.

5. Check the signal without tapping anything:

   ```bash
   python -m vocalbot.cli probe
   ```

   Counts should sit near zero between notes and spike as one crosses. A `*`
   marks a zone that fired.

6. Dry run, then for real:

   ```bash
   python -m vocalbot.cli run --dry-run
   python -m vocalbot.cli run
   ```

## Editing zones without the GUI

```bash
vocalbot zone list                          # boxes, thresholds, priorities
vocalbot zone list --image shot.png         # what each zone sees in a frame
vocalbot zone set blue --thresh-blue 45
vocalbot zone set red --rect 0.12,0.50,0.88,0.515
vocalbot zone set disco --disable
vocalbot zone add hold --rect 0.3,0.4,0.7,0.42 --match any --taps red+blue
vocalbot zone rm hold
```

`--match` is one of `red`, `blue`, `both`, `any`. `--taps` is `red`, `blue` or
`red+blue`. Raise `--priority` to make a zone suppress others.

## Tuning

Three things depend on motion and cannot be read off a still frame: where the
lane box sits vertically, the pixel thresholds, and the refractory period. Fit
them from a recording:

```bash
python -m vocalbot.cli tune song.mov --write
```

This runs an expensive-but-accurate blob tracker over the full playfield as
ground truth, then grid-searches the cheap zone detector until it agrees,
reporting precision and recall. Record 20–30 seconds of one song. The sweep moves
the `lane` group only — set the disco box with `vocalbot zone set disco`.

Check the loop rate any time:

```bash
python -m vocalbot.cli bench
```

`bench` times both capture modes and tells you which is faster. `union` takes one
grab covering every zone and slices it; `per_zone` takes one grab per box. Union
wins when the boxes are close together, per_zone when the union is mostly dead
space — with the default layout only 19% of the union is live pixels, so it's
worth measuring. Set `capture_mode` in `config.json`.

## Known gaps

- **No true multi-touch.** macOS has no public multi-touch injection API, so
  "both drums" is two taps `inter_tap_ms` apart rather than a simultaneous press.
  Rhythm hit windows are tens of milliseconds wide so this normally lands; widen
  it first if doubles are dropping.
- **Note speed is assumed roughly constant.** The lane box is a fixed lead
  distance, not a velocity model. A chart with large speed changes mid-song would
  need per-section offsets.
- **Window must stay pinned.** There's no continuous window tracking; moving or
  resizing the mirroring window invalidates the calibration.
- **Only tap notes are modelled.** If any glyph type turns out to be a hold or a
  flick, it needs a zone with different behaviour than a single tap.
- **The bot doesn't know when to stop.** It has no notion of a pause, countdown
  or results screen and will keep tapping through them.

## Layout

```
vocalbot/
  config.py     zones, colour gates, geometry as screen fractions
  color.py      LUT construction, the hot classify path, per-box fan-out
  scanner.py    per-zone edge state and priority resolution
  capture.py    zone capture (union / per-zone) and mirror-window lookup
  calibrate.py  overlay rendering and the interactive box/drum editors
  tap.py        threaded tap dispatch via CGEventPost
  bot.py        the live loop
  tune.py       offline blob-tracker reference and parameter sweep
  cli.py        calibrate | zone | bench | probe | tune | run
tests/          26 regression tests against the four reference frames
assets/frames/  the reference screenshots, with known-correct answers
```

## Note

Automating input into a game will generally be against its terms of service, and
rhythm games in particular tend to flag improbable accuracy. Your call.
