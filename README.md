# MASTU_hazard

Tearing-mode hazard modelling for MAST-U — a port of the DIII-D framework in
`fdp-demo-hazard` to a spherical tokamak.

The goal is a discrete-time survival (hazard) model for tearing-mode onset:
magnetics detectors label onset events, EFIT-equivalent descriptors form the
feature set, and gradient-boosted trees fit the hazard. This repository
currently covers **Stage A only** — the rotating-mode detector and its
threshold ladder. See `docs/hazard-workflow-analysis.pdf` §9 for the full
audit and the stage-by-stage status.

## The organising idea

A detector never emits a label. It emits, per shot, the **first crossing time
as a function of a threshold ladder**. The threshold then becomes a
post-processing parameter, and a sensitivity study over thresholds costs no
recomputation over the database.

## Layout

| Path | Contents |
|---|---|
| `run_master.py` | Driver. `collect_signals`, `detect_ladder`, `build_rm_detector`, `schmitt_first_crossing` |
| `saddle_analysis.py` | Original OMAHA routine, four defects repaired. `saddle_analysis_one_shot`, `run_batch` |
| `saddle_extras.py` | Additions. `amplitude_with_coherence`, `amplitude_diiid`, `noise_floor`, `analyze_shot` |
| `flattop.py` | Flat-top windowing from Ip. `find_flattop`, `FlatTopConfig` |
| `compare_lf_hf.py` | Transfer function slow(LF)/fast(HF) on one coil: same quantity, time integral, or derivative |
| `probe_timebase.py` | OMAHA slow time-step distribution, where non-uniformity sits, and what `load_omaha`'s slice keeps |
| `probe_omaha.py` | Prints OMAHA channel units/description and any calibration column in `omaha_coils` |
| `check_equivalence.py` | Asserts on real data that the retrofit still matches upstream |
| `check_flattop.py` | `find_flattop` alone over a shot list; every constant is a CLI flag |
| `coherence_survey.py` | Per-bin coherence and power-kept per gate, against a simulated noise floor |
| `compare_pre_plasma.py` | Vacuum vs flat-top: is the coherent power plasma or instrumental? |
| `ladder_base_rate.py` | Ladder CSV -> events, dwell and base hazard per level |
| `plot_ladder.py` | Figures: events vs threshold, and the local slope that locates the knee |
| `plot_mode_track.py` | One shot: gated spectrogram per \|n\| overlaid, with Fa tracks |
| `make_annotation.py` | Ladder CSV + a threshold -> the 7-column survival table |
| `merge_pair.py` | EPQ onto EPM's time base, then flatten to the survival table |
| `compare_trees.py` | Does EPQ use more constraints than EPM? |
| `extract_features.py` | EPM/EPQ descriptor extraction; `--probe` checks node availability |
| `extract_parallel.py` | Threaded, resumable driver for the above |
| `checkpoint.py` | Chunked append-only store; atomic writes, resume on restart |
| `epm_nodes.py` | MAST-U node map, DIII-D feature -> EPM path |
| `run-study.sh` | All of the above, end to end, with parameters pinned per folder |
| `giopath.py` | Locates Giovannozzi's modules and puts them on `sys.path` |
| `egio/` | Vendored dependency — see `egio/README.md` |
| `tests/` | Offline test suite, 7 modules; runs without Freya |
| `docs/` | Design document and audit |

`saddle_extras` imports its shared constants from `saddle_analysis`, so the two
cannot drift apart. Nothing imports in the other direction: `saddle_analysis.py`
stays standalone and testable against the original routine.

## Running

Offline, anywhere — no database needed:

```bash
python tests/run_offline_tests.py
```

Everything at once:

```bash
./run-study.sh GATE-060 0.60
./run-study.sh GATE-060 0.60 --amp-form perbin    # DIII-D's estimator; default is centroid
```

Each stage is skipped if its output exists, so a rerun resumes. Parameters are
pinned to the folder on first run and checked on every later one — a mismatched
gate is a hard error, so a folder cannot mix settings. `PARALLEL_TAGS=true`
harvests EPM and EPQ concurrently; `SHOT_MIN`, `SHOT_MAX`, `WORKERS` and
`N_DETECT` are environment overrides.

Three folders give the gate sensitivity scan, the analogue of DIII-D's
`POINT-A` … `POINT-F`:

```bash
./run-study.sh GATE-055 0.55
./run-study.sh GATE-060 0.60
./run-study.sh GATE-070 0.70
```

Or the same steps by hand:

```bash
python run_master.py --shots $(seq 47000 47099) --build-detector \
    --coherence-min 0.60 --out-csv rm_ladder.csv      # detector -> ladder
python ladder_base_rate.py rm_ladder.csv              # find the knee
python plot_ladder.py rm_ladder.csv --out scan        # figures
python make_annotation.py rm_ladder.csv --threshold <knee> --out joint.csv
                                       # writes joint_threshold<value>.csv
python extract_parallel.py --annotation joint_threshold<value>.csv --tag EPM \
    --store harvest_EPM --workers 8                   # features
python extract_parallel.py --store harvest_EPM --consolidate feat_EPM.pickle
python extract_parallel.py --annotation joint.csv --tag EPQ \
    --store harvest_EPQ --workers 8
python extract_parallel.py --store harvest_EPQ --consolidate feat_EPQ.pickle
python merge_pair.py --primary feat_EPM.pickle --secondary feat_EPQ.pickle \
    --out merged.pickle                               # the survival table
```

The threshold enters only at `make_annotation`, so changing your mind means
redoing from there, not the detector. `run-study.sh` covers everything up to
the consolidated pickles; the merge is a separate call because it joins two
study outputs.

Checks worth running first on a new shot range or after a node-map edit:

```bash
python check_equivalence.py --shot 47000      # retrofit still matches upstream
python check_flattop.py --shots 47000 47001   # the windowing constants
python extract_features.py --probe 47002      # every EPM node resolves
python compare_trees.py --shot 47002          # EPQ really does add constraints
```

### Finding Giovannozzi's modules

`saddle_data`, `mode_functions`, `omaha_coils`, `saddle_geometry`,
`pickup_coil_data` and `sxr_geometry` are not in this repository — they live on
Freya. Every entry point imports `giopath` first, which searches, in order:

1. `$GIOMAST_PATH` — colon-separated like `$PATH`, overrides everything
2. `/home/cm0459/Python/gioMAST` — the usual location on Freya
3. `<repo>/egio` — the vendored copy, `saddle_data.py` only

Directories are *appended* to `sys.path`, so `tests/stubs` can still shadow them
when running offline. If anything is missing you get a diagnostic naming the
modules and the directories searched, ahead of the traceback:

```
Giovannozzi module search:
  [no such ] /home/cm0459/Python/gioMAST
  [found   ] /home/akcay/Python/MASTU_hazard/egio

  MISSING: mode_functions, omaha_coils, saddle_geometry, ...
```

If Gio's directory moves, either set the environment variable

```bash
export GIOMAST_PATH=/path/to/gioMAST
```

or edit `FREYA_DEFAULT` in `giopath.py` — one place, not four.

The ladder CSV has columns `shot, isok_rm, whichn, ta_rm, tb_rm, ip_mean,
ft_nrmse, floor, rm2.0000 … rm30.0000`, one row per shot, each `rm*` column
holding the first crossing time at that level (NaN if never crossed). It is the
MAST-U analogue of `fdp-regen-brm-plain-*.csv`.

## Call graph, one shot

```
run_master.main()
  run_master(shots, build_detector, ...)
    build_rm_detector(shots, client, ladder, ...)
      ├── collect_signals(shot, client, ...)
      │     ├── client.get("/AMC/PLASMA_CURRENT")      measured Ip, kA -> MA
      │     ├── flattop.find_flattop(t, ip, cfg)       -> FlatTop(ok, t_a, t_b, ...)
      │     │     window intersected with the OMAHA record
      │     ├── saddle_extras.analyze_shot(shot, NFFT, ntor, coh_min)
      │     │     ├── saddle_data.load_omaha_slow      -> SaddleFull
      │     │     ├── SaddleFull.spectrum(NFFT)        -> Spectra
      │     │     ├── Spectra.n_detection([0, ±1..±4]) -> RecognizedModes
      │     │     │     n=0 searched as the common-mode sink, never reported
      │     │     ├── amplitude_with_coherence(rm, sel, coh_min)   centroid
      │     │     ├── amplitude_diiid(rm, sel, coh_min)            per-bin
      │     │     │     the gate acts per FREQUENCY BIN, inside the sum
      │     │     └── noise_floor(...)
      │     ├── amp in gauss; /floor only with --normalize-floor
      │     └── aux: XIM/DA/HM10/T, ANE/DENSITY, AYC/T_E
      └── detect_ladder
            schmitt_first_crossing(t, amp, level, debounce,
                                   freq, freq_step_max)   continuity: off by default
            -> rm<level>, fa<level>, co<level>
  -> DataFrame.to_csv                                     the ladder

make_annotation  ladder + threshold -> joint.csv (7 cols + fa_event, coh_event)
extract_parallel analyze -> ChunkStore -> feat_EPM.pickle, feat_EPQ.pickle
merge_pair       EPQ onto EPM at dt/4; label the last pre-event row within 2*dt
                 -> X1, X2, T, ID, Y, matched
```

## What is still not right

**The amplitude scale is relative, not absolute.** Giovannozzi labels
`damplitude_dt` as T/s and `amplitude` as T, but qualifies it: *"I'm saying T
and T/s but, really, I'm not sure that the data I'm reading are in absolute
units, so use it for comparing shots."* So the scale is consistent and
shot-to-shot comparison is valid — which is all a threshold ladder needs — but
a threshold cannot be quoted as a field or compared against DIII-D.

The FFT bookkeeping, at least, is now pinned down. `specgram(mode='complex')`
returns `FFT(w x) / sum(w)` — an amplitude normalisation, not none — so for a
narrowband line Giovannozzi's `Ba` is `sqrt(ENBW/2) = 0.867` of the true RMS
field, *if* the input is calibrated dB/dt in T/s. `amplitude_diiid` removes
that factor and the centroid-frequency approximation, and reproduces a known
synthetic amplitude to 0.1% (`tests/test_coherence.py`). What stays open is the
input's units and calibration, which no formula can fix.

The ladder is therefore **absolute, in gauss** (`GAUSS_PER_TESLA = 1.0e4`,
the same convention as the DIII-D detector). An earlier
version divided by each shot's pre-plasma floor, justified by a ~10x spread in
per-shot median amplitude — but that was measured over the whole record, plasma
included. Measured where there is no plasma (t < 20 ms, instrumental by
construction) the floor varies by only **1.88x** across shots 47000-47099,
12% rel. std dev, with no shot above 3x the median. The factor of ten was
physics, and dividing it out removed the cross-shot variation the hazard model
learns from.

A fixed absolute level is a consistent criterion to ~12%; a fixed
signal-to-floor level would vary by 1.88x in absolute terms. `--normalize-floor`
restores the old behaviour for comparison, and the `floor` column is written
either way.

The tesla reading is settled by plausibility: as stored, the noise floor is
1.8e-07 and the largest excursion 3.0e-05. Taken as gauss, that floor would be
0.018 nT — far below any real magnetic noise. Taken as tesla it is 1.8 mG with
a 0.30 G peak, which is what saddle coils should see. The default ladder spans
2 mG to 0.3 G in 61 log-spaced steps.

These gauss are comparable between your shots, but not yet against DIII-D's
1–30 G until the OMAHA channel units are confirmed. `--amp-form diiid` selects
the DIII-D per-bin 1/ω² estimator (`--amp-form gio`, the default, is
Giovannozzi's). The two differ by a constant 0.867 on a pure line and more on a
broadband spectrum; compare them on the same shots before choosing. Either way
the ladder levels are relabelled by one constant, with no re-detection.

**Toroidal handedness — resolved, by not asking.** `Spectra.n_detection`
projects onto `exp(+i n phi)`, so it detects modes whose spectral phase runs as
`exp(-i n phi)`. A positive-only search list is blind to the opposite sense and,
worse, inflates the *ungated* amplitude of every n at once.

Rather than determine the sense per shot, the detector pools it away:
`analyze_shot(absolute_n=True)`, the default, always searches both signs, keys
its output by |n|, and pools the bins the argmax assigned to `+|n|` or `-|n|`.
`--n-detect 1` therefore means |n| = 1 whatever the rotation sense, and the
ladder's `whichn` column is |n|. This is DIII-D's `get_amplitude(nsigned=False)`
convention, and it suits the hardware: the 5-coil OMAHA subset spans ~60 deg,
on which the +1 and -1 projections are 74% parallel, so reporting them
separately would claim a distinction the array cannot make.

`--signed-n` restores the old signed selection, and
`amplitude_with_coherence(absolute_n=False, coh_min=0)` stays bit-identical to
upstream. The residual geometric risk is |n| = 1 against |n| = 2 aliasing, not
the sign.

## The coherence gate goes inside the frequency sum

`amplitude_with_coherence(rm, sel, coh_min)` zeroes each (frequency, time) bin
whose toroidal-fit coherence is below `coh_min` **before** the amplitude sum,
matching DIII-D's `get_amplitude`, which builds a per-bin weight `W`, sets
`W[SC12 < coh_min] = 0`, and only then sums over frequency.

This matters more than it sounds. An earlier version gated whole *time slices*
after the sum, which left every noise bin inside it. The band is 100 Hz–50 kHz
at 391 Hz resolution, ~127 bins, and with 8 candidate n the argmax hands each
one roughly 16 bins whether or not a mode is present — so the amplitude carried
an irreducible ~1.8 mG floor that no gate setting could lower. Symptoms: a 0.2
gate fires on every low rung of the ladder, a 0.8 gate NaNs every slice and
fires on nothing. Both were observed over 100 shots.

With the gate in the right place, on synthetic data through Giovannozzi's real
classes:

| `coh_min` | noise floor | weak mode | SNR |
|---|---|---|---|
| 0.0 | 0.176 | 0.641 | 3.6 |
| 0.4 | 0.130 | 0.629 | 4.8 |
| 0.8 | 0.055 | 0.613 | 11.2 |

The floor falls 3.2x while the signal moves 0.4%. `coh_min = 0` remains
bit-identical to `RecognizedModes.amplitude()`, which `check_equivalence.py`
asserts.

If every bin in a slice is gated out the amplitude is NaN, not zero — no bin
survived, so there is no coherent power to report. `schmitt_first_crossing`
treats NaN as not-above-threshold, so a gated-out stretch cannot trigger and
resets the debounce, which is the intended behaviour.

## How our selector differs from Giovannozzi's

`amplitude_with_coherence` is `RecognizedModes.amplitude` with six changes.
Everything else — band selection, `zeros_spectrum`, the masked sum, `Bp`, `Fa`
— is byte-identical, and `amplitude_with_coherence(absolute_n=False,
coh_min=0)` is still bit-for-bit equal to his method on real data, which
`check_equivalence.py` asserts.

**1. A coherence gate inside the frequency sum.** His `amplitude()` applies
none. `RecognizedModes.threshold()` exists but is never called, and the driver
passes `v_min=0.0`, so the one remaining rejection (`Fa[Bp < v_min] = nan`) is
inert — nothing is ever discarded. Ours zeroes bins below `coh_min` before
summing, as DIII-D's `W[SC12 < coh_min] = 0` does.

**2. `n = 0` is searched, as a sink.** `saddle_extras.py`, in `analyze_shot`:

```python
search = [0] + [s * m for m in mags for s in (-1, 1)]
```

The argmax must pick a winner from whatever list it is given, so without 0 an
axisymmetric signal is not excluded, it is *mislabelled* — and on these 5 coils
spanning 60 deg it scores 0.866 on n=1, clearing any gate. `search` is what
reaches `n_detection`; the reported magnitudes never include 0.

**3. Pooling on |n|.** His mask is `Ntor != sel.ntor`, signed, so +1 and -1 are
distinct modes — correct given a both-signs search list, which his instructions
specify. (The original script passed `[1,2,3,4]`, positive only, which is what
hid the other sense.) Ours masks on `|Ntor| != |n|`, as DIII-D does with
`nsigned=False`, because a 60 deg array cannot separate +1 from -1 — their
projections are 74% parallel.

**4. A per-bin 1/omega^2 estimator, as an option.** His `Ba = Bp/(2 pi Fa)`
divides the band total by one power-weighted centroid frequency, exact only
for a narrowband line. `--amp-form perbin` instead weights each bin by its own
frequency. **`centroid` remains the default**; `perbin` is there for comparison.

Worth knowing when you gate: the centroid form is unstable under it. Removing
the high-frequency bins collapses `Fa`, so `Ba = Bp/(2 pi Fa)` *rises*.
Measured on 47000-47019, gating 0 -> 0.8 cut `Bp` by 1.8x while `1/Fa` grew
23x, inflating the pre-plasma amplitude 13-fold. Compare the two forms before
trusting a gated amplitude.

**5. ENBW normalisation** (in `perbin` only). `specgram(mode='complex')`
returns `FFT(w x)/sum(w)`, so his `Ba` is `sqrt(ENBW/2) = 0.867` of the true
RMS even for a clean line. `amplitude_diiid` applies `2/ENBW` and reproduces a
known synthetic amplitude to 0.1%.

**6. The coherence is returned.** His method returns `(time, Fa, Bp, Ba)` and
discards the coherence `n_detection` already computed. Ours returns the
power-weighted coherence per slice alongside, so it can be gated on and
inspected.

Separately, `FMIN_DEFAULT` moved from 100 Hz to 1 kHz — Giovannozzi's own
example value. That is a default in `saddle_analysis.py`, not a change to the
selector.

## n=0 absorbs the common mode

A signal every coil sees identically — baseline wander, slow drift — has no
toroidal structure, but on a clustered array it does not cancel against
`exp(-i n phi)`. On the 5 OMAHA coils spanning 60 deg a pure common-mode
signal scores

    |sum exp(-i phi_c)|^2 / N_c^2  =  0.866   on n=1

which clears any gate worth setting. Measured consequence, shots 47000-47019:
at `coh_min = 0.8` the surviving pre-plasma bins were bin 1 alone — Fa = 390 Hz,
exactly the FFT bin width — i.e. drift read as a near-perfect n=1 mode in a
vessel with no plasma.

`n = 0` is now always searched and never reported. The argmax sends those bins
to n=0 (coherence 1.000 beats 0.866) and the |n| >= 1 masks exclude them. On
synthetic pure common mode this cuts the spurious n=1 amplitude by **230x**
(`tests/test_coherence.py`). DIII-D is protected instead by its integerness
gate, which common mode fails at nhat ~ 0.

Two companion changes:

**`--amp-form perbin`** (DIII-D's per-bin 1/omega^2) is available alongside
the default `centroid` (Giovannozzi's `Ba`). The centroid form is unstable
under gating: strip the high-frequency bins and `Fa` collapses, so
`Ba = Bp/(2 pi Fa)` *rises*. Measured: gating 0 -> 0.8 cut `Bp` by 1.8x while
`1/Fa` grew 23x, inflating the pre-plasma amplitude 13-fold. An amplitude that
grows when you remove bins from the sum is not an amplitude.

**`FMIN_DEFAULT` is 1 kHz**, up from 100 Hz, which had put the lowest FFT bin
(390.6 Hz) inside the band. It sat at 2 kHz briefly; with n=0 absorbing the
drift the floor no longer has to do that job, so it is back at Giovannozzi's
own example value. Bins 1 and 2 are excluded, bin 3 up is kept.

### What this invalidates

The 0.60 gate below, the `coherence_survey` power-kept plateau, and every
ladder result predate these fixes. The "coherent minority holding half the
power" may have been the common mode. Rerun `coherence_survey` first — it is
cheap, and it says whether a real plateau survives.

## Where the gate went: 0.60, now superseded

`coherence_survey` on 93 shots once showed power-kept flat across gates
0.40-0.70 while bins-kept fell ninefold, which read as a coherent minority
holding half the in-band power. That measurement predates the n=0 fix, and the
plateau was plausibly the common mode. **Rerun `coherence_survey` before
trusting any gate value**; the method is sound, the number is not.

What replaced it as evidence is `compare_pre_plasma`, which asks whether the
surviving power exists with no plasma in the vessel — a question no ladder
shape can answer.

## Feature extraction

Every DIII-D descriptor has an EPM counterpart, including `S1`/`S2`/`S3` (so
`glitch` is computable) and `FINALCHISQUARED` / `FINALPOLOIDALFLUXERROR`. Run
`--probe` on one shot after any node-map edit; it reports shape, units and time
base per node, and has already caught three things a sweep would only have
surfaced hours in.

**Cadence.** EPM writes 200 slices over [0.020, 1.020] s, so dt = 5.03 ms —
four times finer than the DIII-D grid the tolerances were tuned on. Both are
derived from the measured cadence rather than inherited: `event_delta` = 2
slices = 10.05 ms, merge delta = dt/4 = 1.26 ms. dt is simultaneously the slice
spacing, the prediction horizon, and the Delta in `h*Delta = softplus(f)`.

**R0 = 0.85 m, fixed by convention.** EPM has no fixed-R0 node;
`BVACRADIUSPRODUCT/BVACRGEOM` gives the geometric centre (0.858 m on 47002) and
`/BVACRMAG` the magnetic axis (0.965 m), but both move with the plasma, and
normalising by a moving radius would fold plasma motion into every length
feature. DIII-D's `RZERO` is likewise a convention, not a measurement.

**The chop is at `tevent`, and must be.** `table_stack` labels the last row of
each shot precisely because that row is the last pre-event slice. `--ladder`
extracts the unchopped flat-top window instead, which lets one harvest serve
several thresholds — but then the stacking step must do the chop, and that is
the one place a bug mislabels everything without erroring. At 33 ms/read with
8 workers a re-harvest is ~40 min per tag, so it rarely pays.

**Timing.** Measured on Freya: 33 ms per node read, 29 reads per shot, 0.96
s/shot. 10,000 shots across EPM and EPQ is 5.3 h serial. The time is network
latency, not computation, so threads help nearly linearly; `extract_parallel`
gives each worker its own pyuda client and checkpoints through `ChunkStore`,
so a run that dies resumes where it stopped.

## Fa at the crossing

The ladder now carries three columns per level, not one:

    rm0.01365   first crossing time [s]
    fa0.01365   Fa on that slice -- the mode frequency at onset [Hz]
    co0.01365   coherence on that slice

Both extras are already computed at the crossing slice, so they cost nothing.
`make_annotation` carries them through as `fa_event` and `coh_event`, appended
after the seven-column format, and prints the onset-frequency distribution:

    Fa at onset   4.25 kHz  IQR [4.13, 4.45]  over 12 events

They are set only for `ievent == 1`: for a censored shot `tevent` is just the
end of the window, where no crossing happened.

What they buy: the rotating-versus-locked distinction, an input to a
frequency-drop detector, and a screening column for the survival table.
Everything that reads a ladder filters on `rm*`, so the new columns disturb
nothing, and a ladder written before them still works — `make_annotation` says
so and leaves the two fields empty.

## Optional: Fa continuity in the trigger

`--freq-step-max <Hz>` rejects a slice whose Fa jumped more than that from the
previous one, so a crossing needs the amplitude above the level AND the
frequency behaving. **Off by default.**

Continuity, not stability. A mode that slides is still continuous slice to
slice; noise Fa hops across the band. Per-slice steps at the 1.28 ms stride:

| | per-slice dFa |
|---|---|
| slow chirp, 4->3 kHz over 200 ms | 6 Hz |
| locking transition, 4->1 kHz over 20 ms | 192 Hz |
| fast lock, 4->1 kHz over 5 ms | 768 Hz |
| **noise** | **~16 kHz** |

So 1000-2000 Hz passes every physical case and rejects noise by 10-20x. A
*stability* test over the debounce window would instead reject the locking
transition — the event we most want to keep.

The mask does not depend on the level, so the ladder rows stay non-decreasing
left to right and that consistency check still works
(`tests/test_trigger.py`).

Before adopting it, run your known-TM shots with and without and compare event
counts. The events it removes should be the ones `plot_mode_track` shows as
scatter rather than a track:

```bash
python run_master.py --shots <shots> --build-detector --coherence-min 0.6 \
    --out-csv a.csv
python run_master.py --shots <shots> --build-detector --coherence-min 0.6 \
    --freq-step-max 1500 --out-csv b.csv
python ladder_base_rate.py a.csv b.csv --labels off on
```

## Chirping modes and the pooled Fa

`compare_pre_plasma`'s third criterion, "Fa concentrates", fails on a real
mode whose frequency slides. Fa is a weighted mean over FREQUENCY within one
slice, never across time, so it follows a chirp exactly — verified to 0.000 kHz
median error on an injected 12 -> 3 kHz sweep (`tests/test_coherence.py`). What
loses the chirp is POOLING Fa over slices: the same test gives a pooled IQR of
4.46 kHz while the track is perfect.

So if a shot passes "vacuum collapses" and "strong slices hold" but fails
"Fa concentrates", suspect a chirp before suspecting noise, and look at:

```bash
python plot_mode_track.py --shot 49149 --n-list 1 2 3 --coherence-min 0.6
```

Panel 1 overlays one gated spectrogram per |n|, each in its own hue: colour
says which |n| claimed the bin, intensity is power, and bins failing the mask
or the gate are blank. Fa is drawn on top. A real mode is a continuous track;
noise is scatter. Panels 2 and 3 are the amplitude the ladder triggers on and
the per-slice coherence, same time axis, one line per |n|.

Fa is a centroid, so on noise it wanders across the whole band and says
nothing. The track is therefore drawn only where the amplitude exceeds
`--track-snr` (default 3) times that |n|'s own pre-plasma floor — without
that, the scatter from the absent modes swamps the tracks of the present
ones. Up to four |n| values; a fifth would need a generated hue.

## The merge: full EPM and the matched pair, from one file

```bash
python merge_pair.py --primary feat_EPM.pickle --secondary feat_EPQ.pickle \
    --out merged.pickle
```

EPM is the primary, as EFIT01 is for DIII-D. **Every EPM row is kept**; EPQ
contributes NaN wherever it has no slice in the matching window, and a shot
absent from EPQ entirely keeps its EPM rows with an all-NaN secondary.
`--secondary` may be omitted to work on EPM alone. The choice of analysis is
made downstream, by mask, so both run off one harvest:

    has_secondary   an EPQ slice exists in the window
    usable1         every selected feature is finite in EPM -- the full-EPM rows
    matched         has_secondary AND every feature finite in both -- the pair

A missing slice and a slice with one NaN feature used to look the same. They
no longer do, and the merge prints the NaN fraction per feature plus an event
ledger: how many labelled events are usable in EPM alone, how many are lost
because the label row itself has a NaN, and how many are lost only to the
pairing. That last count matters because a NaN on the label row removes the
event silently as soon as a model filters NaN rows.

Tolerances come from the measured 5.03 ms cadence, not from DIII-D's 20 ms
grid: merge delta `dt/4` = 1.26 ms, `event_delta` `2*dt` = 10.05 ms. Because
the window is ±dt/4, at most one secondary slice can fall inside it, so
"earliest candidate" and "nearest candidate" coincide — the ambiguity in the
DIII-D original does not arise here, and `tests/test_merge.py` asserts it.

The label goes on the last row of an event shot, which *is* the last pre-event
slice because extraction chopped at `tevent`. It is applied only if that row
sits within `event_delta` of the event; otherwise the event is dropped and
counted, since a growing island degrades the fit and an equilibrium from long
before the onset would be a false label.

`error` and `chisq` are dropped from the model features by default — they are
reconstruction quality, for screening only.

## Not yet written

Locked-mode detector (no counterpart to the DIII-D M-matrix path — the spectral
detector cannot see a DC island, and the 1 kHz band floor excludes one by
construction); screening limits re-derived for an ST; the hazard model itself.

EPQ really is the better-constrained tree — `compare_trees.py` confirms it uses
constraints EPM does not — so the EFIT01/EFIT02 analogy holds and the
matched-pair machinery ports. That part is done (`merge_pair.py`).

Nothing in the equilibrium half has yet been run end to end on real data:
`extract_features` has only been exercised through `--probe`. Five shots
through the full chain will surface the unit assumptions, the `nhat` chord
length and whether `lost_events` fires, in seconds rather than hours.
