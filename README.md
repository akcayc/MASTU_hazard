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
| `saddle_extras.py` | Additions. `amplitude_with_coherence`, `noise_floor`, `analyze_shot`, `NTOR_BOTH_SIGNS` |
| `flattop.py` | Flat-top windowing from Ip. `find_flattop`, `FlatTopConfig` |
| `check_equivalence.py` | Asserts on real data that the retrofit still matches upstream |
| `giopath.py` | Locates Giovannozzi's modules and puts them on `sys.path` |
| `egio/` | Vendored dependency — see `egio/README.md` |
| `tests/` | Offline test suite; runs without Freya |
| `docs/` | Design document and audit |

`saddle_extras` imports its shared constants from `saddle_analysis`, so the two
cannot drift apart. Nothing imports in the other direction: `saddle_analysis.py`
stays standalone and testable against the original routine.

## Running

Offline, anywhere — no database needed:

```bash
python tests/run_offline_tests.py
```

On Freya, in order:

```bash
# 1. does the retrofit still match upstream on real data?
python check_equivalence.py --shot 47000

# 2. original batch sweep (unchanged output format + .dropped.json sidecar)
python saddle_analysis.py --shot-min 47000 --shot-max 47100

# 3. the detector: threshold ladder -> CSV
python run_master.py --shots 47000 47001 47002 --build-detector \
    --coherence-min 0.6 --out-csv rm_ladder_47000.csv
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
      │     ├── saddle_extras.analyze_shot(shot, NFFT, ntor)
      │     │     ├── saddle_data.load_omaha_slow      -> SaddleFull
      │     │     ├── SaddleFull.spectrum(NFFT)        -> Spectra
      │     │     ├── Spectra.n_detection(ntor)        -> RecognizedModes
      │     │     └── amplitude_with_coherence(rm, sel), noise_floor(...)
      │     ├── snr = amp / noise_floor;  snr[coherence < gate] = NaN
      │     └── aux: XIM/DA/HM10/T, ANE/DENSITY, AYC/T_E
      └── detect_ladder -> schmitt_first_crossing(time, snr, level, debounce)
  -> DataFrame.to_csv
```

## Two things that are not yet right

**The amplitude scale is relative, not absolute.** Giovannozzi labels
`damplitude_dt` as T/s and `amplitude` as T, but qualifies it: *"I'm saying T
and T/s but, really, I'm not sure that the data I'm reading are in absolute
units, so use it for comparing shots."* So the scale is consistent and
shot-to-shot comparison is valid — which is all a threshold ladder needs — but
a threshold cannot be quoted as a field or compared against DIII-D. `specgram`
also applies no window-power or sample-rate normalisation, so an O(1) factor is
unaccounted for on top.

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

These gauss still carry the unverified O(1) factor from the missing `specgram`
window-power normalisation, so they are comparable between your shots but not
yet against DIII-D's 1–30 G. Settling that factor relabels every column by one
constant, with no re-detection.

**Toroidal handedness.** `Spectra.n_detection` projects onto `exp(+i n phi)`,
so it detects modes whose spectral phase runs as `exp(-i n phi)`. A mode of the
opposite sense is invisible to a positive-only search list — and, worse,
inflates the *ungated* amplitude of every n at once. `NTOR_DEFAULT` therefore
carries both signs, matching Giovannozzi's instruction that `n_detection` takes
*"a list of toroidal mode number (positive and negative)"*. Which half carries
the coherence tells you the sense of rotation. See `docs/` §9.6.

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

## Where the gate goes: 0.60

Measured on 93 shots of 47000-47099 with `coherence_survey.py` — 10.1M in-band
bins, 5 OMAHA coils at phi = [36.8, 26.5, 4.0, -5.0, -23.0]:

| gate | bins kept | power kept | noise through |
|---|---|---|---|
| 0.20 | 80.5% | 92.4% | 100% |
| 0.30 | 42.7% | 64.5% | 99.9% |
| 0.40 | 15.9% | 53.5% | 52.0% |
| 0.50 | 4.9% | 50.4% | 2.1% |
| **0.60** | **2.8%** | **48.8%** | **0.005%** |
| 0.70 | 1.7% | 45.3% | 0% |
| 0.80 | 0.76% | 19.3% | 0% |

Power kept is flat across 0.40–0.70 while bins kept falls ninefold and noise
leakage collapses to zero. That shelf is the detection: about half the in-band
power sits in a small, highly coherent population and the rest is noise spread
over millions of bins. At 0.80 power kept falls to 19.3% — the gate has started
cutting real signal — so the shelf ends near 0.70.

**0.60** leaks roughly 500 noise bins in 10 million while keeping 48.8% of the
power. Run 0.55 and 0.70 as the sensitivity pair, the way DIII-D repeats its
whole study over `POINT-A` … `POINT-F`.

The simulated noise floor (median 0.402 on this geometry) sits *above* the
observed median of 0.279, so the simulation is a conservative white-noise
idealisation — real leakage is lower than the table says, not higher.

## Not yet written

Locked-mode detector (no counterpart to the DIII-D M-matrix path — the spectral
detector cannot see a DC island); joint annotation and pathway model; EPM/EPQ
feature extraction; the matched-time-base merge; the hazard model itself.

EPQ and EPM appear to be the MAST-U counterparts of EFIT02 and EFIT01, so the
matched-pair machinery should port with little change.
