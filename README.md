# sxs-vocal-studio

A scanline bot for the two-drum rhythm game, driving an iPhone through macOS
**iPhone Mirroring**.

The rule it plays by: read the colour of the notes coming down the runway. Red at
the front means tap the red drum, blue means the blue drum, both at once means
both, and the mirror ball means blue.

## How it works

Notes travel down a runway toward two drums. Their colour, not their lane
position, decides which drum to hit — which matters because fever mode collapses
the two lanes into one centred lane, and a position-based detector would break
there. Colour survives the mode change.

So the detector watches a single horizontal strip across the runway and counts
red and blue *glyph* pixels. The note glyphs are far more saturated than the pale
bubbles carrying them, so a plain HSV gate separates them cleanly.

```
        ┌───────────────────────────────┐
        │        note runway            │
        │   ○ blue          ● red       │
        │                               │
   ═════╪═══════ scanline ══════════════╪═══   <- the only pixels read
        │       ● red                   │
        │                               │
        │   [blue drum]   [red drum]    │
        └───────────────────────────────┘
```

### Why the scanline sits high

Measured on the reference frames: below about 60% of screen height the frontmost
note glyphs are large enough to **overlap into one continuous run**, so two
consecutive same-colour notes merge into a single detection and the bot taps once
where it should tap twice. Higher up the glyphs separate with clean gaps between
them.

Placing the line high also buys lead time, which is what absorbs mirroring
latency. The scanline position is therefore the single latency knob — raise it to
tap earlier, lower it to tap later.

The cost is signal strength: a note is ~150–1100 glyph pixels up there versus
~2500 at the bottom, against a noise floor around 20. Still a wide margin, but it
is why the thresholds are low.

### Why it's fast

| stage | cost |
|---|---|
| full-ROI blob detection + tracking (rejected) | ~8 ms per colour |
| **scanline classify, LUT + subsampled** | **0.033 ms** |

Three choices get it there:

1. **Only the strip is captured**, never the whole window — a 910×24 grab instead
   of 910×1260. Screen capture dominates the loop, so this is the biggest win.
2. **No per-frame HSV conversion.** The HSV gates are baked once into a
   32768-entry lookup table keyed on the top 5 bits of each channel. Per frame
   it's one gather and one `bincount`.
3. **Subsampling by 3.** Notes are large blobs; discarding 8 of every 9 pixels
   costs nothing in separability and cuts the work ninefold.

Taps run on a separate thread, so a held tap never blinds the detector to the
next note.

## Setup

Requires macOS Sequoia or later, an Apple silicon Mac, and iOS 18+ — that's what
iPhone Mirroring needs.

```bash
pip install -r requirements.txt
```

Grant both permissions in **System Settings → Privacy & Security**, or nothing
will work:

- **Screen Recording** — for your terminal, to capture the strip
- **Accessibility** — for your terminal, to post synthetic clicks

Then:

1. Open iPhone Mirroring and **leave the window where it is**. All coordinates
   are captured relative to a pinned window rect; move it and you must
   recalibrate.
2. Start the song so the playfield is on screen.
3. Calibrate:

   ```bash
   python -m vocalbot.cli calibrate --chrome 28
   ```

   This writes `calibration.png`. Open it and check the green strip crosses the
   note runway and the two circles sit on the drums. If they don't, adjust
   `lane_x0`, `lane_x1`, `scan_y` or `drum_*` in `config.json` and re-run.
   `--chrome` trims the mirroring window's title bar; 28 is a starting guess.

4. Check the signal without tapping anything:

   ```bash
   python -m vocalbot.cli probe
   ```

   Watch the counts as notes cross. They should sit near zero between notes and
   spike into the hundreds as one passes.

5. Dry run — detects and logs, never taps:

   ```bash
   python -m vocalbot.cli run --dry-run
   ```

6. For real:

   ```bash
   python -m vocalbot.cli run
   ```

## Tuning

Three values depend on motion and cannot be read off a still frame: the scanline
height, the pixel thresholds, and the refractory period. Fit them from a
recording:

```bash
python -m vocalbot.cli tune song.mov --write
```

This runs an expensive-but-accurate blob tracker over the full playfield as
ground truth, then grid-searches the cheap scanline until it agrees, reporting
precision and recall. Record 20–30 seconds of one song and **include a fever
section** — see the known gap below.

Check the loop rate any time with:

```bash
python -m vocalbot.cli bench
```

## Known gaps

- **Blue notes during fever are untested.** Fever mode tints the whole screen
  gold. Red still detects cleanly under it, but the reference frames contained
  almost no blue notes during fever, so the blue gate has not been checked in
  that lighting. This is the first thing a recording should settle — if blue
  misses during fever, widen `blue.h_lo`/`h_hi` or drop `blue.s_min`.
- **No true multi-touch.** macOS has no public multi-touch injection API, so
  "both drums" is two taps `inter_tap_ms` apart rather than a simultaneous press.
  Rhythm hit windows are tens of milliseconds wide so this normally lands, but
  it's the first thing to widen if doubles are dropping.
- **Note speed is assumed roughly constant.** The scanline is a fixed lead
  distance, not a velocity model. A chart with large speed changes mid-song would
  need per-section offsets.
- **Window must stay pinned.** There's no continuous window tracking; moving or
  resizing the mirroring window invalidates the calibration.

## Layout

```
vocalbot/
  config.py    geometry as screen fractions, colour gates, tuning values
  color.py     LUT construction and the hot classify path
  scanner.py   rising-edge state machine with hysteresis and refractory
  capture.py   strip capture (mss / Quartz) and mirror-window lookup
  tap.py       threaded tap dispatch via CGEventPost
  bot.py       the live loop
  tune.py      offline blob-tracker reference and parameter sweep
  cli.py       calibrate | bench | probe | tune | run
tests/         regression tests against the four reference frames
assets/frames/ the reference screenshots, with known-correct answers
```

## Note

Automating input into a game will generally be against its terms of service, and
rhythm games in particular tend to flag improbable accuracy. Your call.
