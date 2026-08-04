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

- **Screen Recording** — for your terminal, to read the screen
- **Accessibility** — for your terminal, to click and to watch for your clicks

Then there is one command:

```bash
python -m vocalbot.cli run
```

The first run walks you through setup, one prompt at a time. Put the terminal
beside the phone window so you can read as you work. It raises iPhone Mirroring,
waits for you to press ENTER in the terminal, then asks you to point at each
target in turn:

| | |
|---|---|
| the **blue drum** | records where to press it |
| the **red drum** | records where to press it |
| the **vivid core of a blue note** | records what blue looks like on your screen |
| the **vivid core of a red note** | records what red looks like |
| a white "both" marker | optional |
| the purple of the disco ball | optional |

You pick a target by **hovering over it and holding still** for about a second —
no clicking, no keypresses. A progress bar fills as you hold; moving restarts
it, so overshooting costs nothing. Optional steps ask yes/no in the terminal
first. Ctrl-C stops at any point.

Then it shows a **live readout** for 25 seconds. Watch it against the game: a
blue note should read blue, a red note red, a pair both. It then asks whether to
start playing.

### Why hovering, and not clicking or a hotkey

Reading the global keyboard or mouse-button state on macOS needs the **Input
Monitoring** permission, which is separate from Accessibility. When it's missing
those calls don't fail — they silently report that nothing is pressed, so a
hotkey prompt just sits there forever. Reading the *cursor position* needs no
permission at all, so setup only ever asks where the pointer is.

Hovering also avoids a side effect: a click inside the mirroring window is
forwarded to the phone, so picking targets by clicking taps the game as you go.

Re-run setup any time with `run --setup`. Do that whenever the mirroring window
moves, since the coordinates are absolute.

### Why setup asks for the *vivid* part, not the darkest

Measured on the reference frames: the shadowed cores of the blue and red notes
are only **52 apart** in colour, while their vivid cores are **246 apart**. A
sample taken from a dark area makes the two notes indistinguishable. Sampling
also keys off the most strongly coloured pixels near your click rather than
averaging the patch, so clipping the edge of a note still yields the note's
colour instead of the pale bubble's — verified exact with the note covering as
little as 7% of the click.

The same reasoning is why a washed-out sample gets a warning: a pale colour
matches the bubbles and the background too, which causes phantom presses. That
is also the honest caveat on the disco step — the ball's purple is pastel, and
on the reference frames a disco sample produced 426 false matches on a frame
with no ball. Skipping it is usually better, since the ball reads as blue on
its own.

### What setup removes

Two things kept going wrong before, and both are gone:

- **No window rect.** Drum positions are absolute screen coordinates from your
  clicks, so nothing has to work out where the phone screen sits inside the
  mirroring window — a window with no title bar, variable padding, and a shadow.
- **No colour thresholds.** The colours come from your display rather than from
  reference screenshots, so nothing has to be guessed in advance or re-tuned.

The older `calibrate` / `zone` / `doctor` commands still work and still drive the
zone-based detector, which is what `replay` uses to check against a recording.
`run` uses the sampled palette once setup has been done.

## Stopping it

**Move the mouse away from the drums.** That is the way out, and it acts within
about a third of a second.

This matters more than it sounds. While playing, the bot clicks the mirroring
window many times a second, which takes focus — so your keystrokes are going to
the phone, not the terminal, and Ctrl-C can be genuinely hard to land. Reading
the cursor position needs no permission, so this check cannot silently fail the
way a hotkey can.

Backstops, in order of how fast they act:

| | |
|---|---|
| move the mouse | ~0.3s |
| wall-clock limit | `max_run_seconds`, default 150s — `--duration 0` removes it |
| Ctrl-C / SIGTERM | when the terminal is reachable |
| tap thread | daemon, and joined on exit, so nothing keeps clicking |

The bot only ever parks the cursor on one of the two drums, so a cursor
anywhere else means you took over. A single stray reading won't end a run — the
cursor has to stay away for the grace period.

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
- **Calibrated while the game was paused, or while the window was behind the
  terminal.** Content detection needs motion, and the window stops animating in
  the background. Both `calibrate window` and `doctor` raise it first; if that's
  blocked, use `--rect x,y,w,h`.
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
  wizard.py     guided setup: global click/key watching, colour sampling
  palette.py    matching against clicked colours, and the press-both fallback
  replay.py     offline replay of the live decision path over a recording
  doctor.py     capture diagnostics: which window, what is actually grabbed
  cli.py        run | calibrate | zone | doctor | bench | probe | replay
tests/          110 tests: palette, queue scanner, editor, detection, diagnostics
assets/frames/  reference screenshots with known-correct answers
```

## Note

Automating input into a game will generally be against its terms of service, and
rhythm games in particular tend to flag improbable accuracy. Your call.
