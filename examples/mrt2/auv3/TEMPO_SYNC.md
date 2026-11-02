# Host tempo / downbeat sync (`tempo-sync` fork)

This fork adds **Host Sync** to the AUv3 plugin. MRT2 follows the DAW's tempo
and puts its downbeats on the DAW's bars. The plugin reads the tempo and bar
position from the host's AU transport hooks (`musicalContextBlock`,
`transportStateBlock`). It uses no Ableton Link, no MIDI clock and no wall
clock.

The fork installs as its own plugin, **"Google: MRT2 Sync"**: bundle id
`com.google.mrt2sync.au`, subtype `MGRS`. It sits next to the upstream
"Google: MRT2" and does not replace it.

> [!IMPORTANT]
> Host Sync needs a model exported with `magenta_rt.mlx.export_tempo`; see
> [Export](#export). It does not work with the stock `mrt2_base.mlxfn`.

## How it works

**MRT2 has no tempo input.** It continues the tempo of whatever it has
just heard. A clean tempo change is therefore a **re-prime**:

1. The plugin time-stretches the last 6 s of MRT2's own output to the new
   tempo. It uses [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch),
   which keeps the pitch.
2. It rebuilds MRT2's attention memory from that audio in one parallel
   pass. This happens in `MLXEngine::reprime_from_audio` and takes about 0.3 s.

The stretched audio is never played. MRT2 continues its own music at the
new tempo from the next frame. The plugin makes each change right after one
of MRT2's beats.

**Bar lock.** The tempo export has a second output, MRT2's beat / downbeat
read-out. This is a linear probe on temporal block 11, so it only sees
generated tokens. A causal bar-pointer tracker (`core/src/tempo/beat_tracker.cpp`)
runs on this read-out to follow MRT2's beat and bar.

When MRT2's bar drifts more than 0.12 beat off the host's bar, a **phase
re-prime** fixes it. It rebuilds the memory from a context that ends where
MRT2's music was at the host's current bar position. That point is
`(-offset mod 1 bar)` back, so the correction is always a rewind: the music
repeats up to one bar. The tempo goes to the host's in the same step.

MRT2's tempo is measured from how its beat phase drifts against the host's
grid. A 1 % tempo error is 0.12 beat after 6 s, which is easy to measure;
the beat intervals alone are much noisier.

**The grid comes from the host.** Each render call reads stream sample
`cursor` (48 kHz) while the host is at beat `b`. The host places this output
`latency` earlier on its timeline (Delay Comp / PDC), so that sample carries
beat `b - latency * tempo / 60`. This is the same with or without PDC. The
mapping is sample-exact and updates on every buffer:

- A host tempo change is one phase re-prime to the new tempo, landing on the
  host's bar at the switch.
- A locate or loop jump of whole bars changes nothing.
- Any other jump re-aligns on MRT2's next beat.

While the transport is stopped, the plugin keeps the last grid running at
the host's tempo.

**Explicit CFG (Prompt Adherence) stays on by default.** Explicit guidance
pulls the music toward the prompt's own tempo. Open loop it is strong: one
re-prime to ×0.88 reached ×0.96 with explicit CFG at 3, and ×0.87 without
it. A ×1.00 re-prime drifted +3.8 % with it and 0 % without.

The sync is closed-loop, though, and the next re-prime corrects that pull.
With explicit CFG, the style and its beat stay steadier, which helps the
tracker. In the harness, runs with explicit CFG kept matched or beat runs
without it (see Results below).

**Sync Keeps CFG** (parameter 51) off leaves only the distilled CFG tokens
while synced: `neg_musiccoca` is set to the positive conditioning.

The logic ports the Python "MRT2 Conductor" (Clean re-prime tempo control,
downbeat lock, Ableton Link sync) from the
[mrt2-mech-interp](https://github.com/jesseengel/mrt2-mech-interp) project
(`rhythm/conductor/director.py`), with the same constants. The C++ beat
tracker matches the Python reference exactly on a 24-s read-out: same 40
beats and 11 downbeats, tempo within 6e-7 BPM.

## Parameters (AU addresses)

| # | id | name | default | |
|---|---|---|---|---|
| 49 | `hostsync` | Host Sync | off | Follow the host's tempo (and bars) |
| 50 | `barlock` | Bar Lock | on | Also keep MRT2's downbeats on the host's bars (off: tempo only) |
| 51 | `synckeepcfg` | Sync Keeps CFG | on | Keep the explicit musiccoca CFG while synced (off: distilled CFG tokens only) |

The UI shows these under the knobs, next to Delay Comp. It also shows the
host and MRT2 tempos, MRT2's bar offset in beats and the number of re-primes.

## Latency

A re-prime blocks the inference thread for about 0.3 s. With Host Sync on, the
ring buffer therefore keeps a 0.5 s lead instead of the 43–170 ms of the
Buffer Size setting (`RingBuffer::kCapacity` is now 32768). A re-prime only
runs when at least 0.42 s of audio is buffered; offline renders wait anyway.

The latency reported to the host stays the Buffer Size setting. The sync
maps the host's beat onto samples as they are read, so the larger lead does
not shift the alignment. It only delays your MIDI and prompt changes by
about 0.5 s while synced.

In Host Sync mode, the plugin does not drain the buffer when the transport
starts. It also does not reset the model when the transport returns to
beat 0. It keeps playing and re-aligns.

## Export

Use the fork's Python package. The MLX version must match the C++ runtime
(MLX v0.31.1, pinned in `CMakeLists.txt`):

```sh
uv venv && uv pip install "cmake<3.28" -e ".[mlx]" mlx==0.31.1 python-stretch
./.venv/bin/python -m magenta_rt.mlx.export_tempo --output-name mrt2_base_tempo --bits 8 \
    --readout readout_L11.npz
```

`readout_L11.npz` is MRT2's beat / downbeat probe at block 11, from
mrt2-mech-interp (`rhythm/conductor/models/`). The export writes these files
to `$MAGENTA_HOME/magenta-rt-v2/models/mrt2_base_tempo/`:

| file | size | contents |
|---|---|---|
| `mrt2_base_tempo.mlxfn` | 2.8 GB | the step, plus the read-out |
| `mrt2_base_tempo_reprime.mlxfn` | 2.4 GB | the re-prime |
| `mrt2_base_tempo_encoder.mlxfn` | 0.1 GB | the encoder, for exactly 150 × 1920 samples |
| `mrt2_base_tempo_state.safetensors` | | the state |
| `mrt2_base_tempo_tempo.json` | | metadata |

`--check only` compares the parallel re-prime with the engine's sequential
prefill.

## Build and install

```sh
./.venv/bin/cmake . -B build -DCMAKE_BUILD_TYPE=Release
./.venv/bin/cmake --build build --target package_mrt2_au_local -j10   # AU + UI, signed, in build/
ditto build/examples/mrt2/auv3/mrt2_au_app.app "$HOME/Applications/MRT2 Sync (AU).app"
open "$HOME/Applications/MRT2 Sync (AU).app"                         # registers the extension
```

`deploy_mrt2_au` (copy to `~/Applications` + `pluginkit`) still exists. It
is no longer part of `all`, so a build never installs anything by itself.

In the plugin:

1. Load the `mrt2_base_tempo` folder. Its `_reprime` / `_encoder` files load
   automatically.
2. Turn on **Host Sync**, then press play in the DAW.

The sandboxed plugin reads `~/Documents/Magenta/`, or any folder you pick in
its model browser.

## Offline test harness

`tempo_sync_cli` runs the engine and the sync controller exactly as the
plugin does, against a simulated DAW transport and a simulated real-time
output clock. Step times are measured. Underruns stall the stream as they do
in the AU. The ring holds `--lead` seconds.

```sh
./.venv/bin/cmake --build build --target tempo_sync_cli -j10
MAGENTA_HOME=... build/examples/tempo_sync_cli/tempo_sync_cli <model dir> \
    --prompt "deep house" --host 0:112:4 --host 28:120:4 --sync-at 8 --seconds 45 --out run
```

It writes `run.wav` and `run.json`: the per-frame read-out, host snapshots
and re-prime events. `tempo_tracker_test` checks the C++ tracker against the
Python reference.

mrt2-mech-interp's `rhythm/au_sync/` has the evaluation:
`run_cli_bench.sh`, then `eval_tempo_sync.py`. It decodes the beats with
unmodified MRT2 and compares the music with the simulated host's grid.

## Limits

- **Re-primes are not perfect.** MRT2's tempo after a re-prime is within
  about 1–2 % of the context's. Bar Lock therefore corrects about every
  5–10 s, each time with a phase re-prime that repeats up to one bar.
- **Meters.** MRT2's tracker knows 3 and 4 beats per bar. The host's time
  signature is read as `numerator * 4 / denominator` quarter notes per bar.
- **Continuous tempo automation** re-primes at most about once per beat. It
  follows ramps in steps.
- **Tested offline only.** It has been tested in the CLI harness, not yet
  in a DAW.

