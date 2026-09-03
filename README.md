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

`GIOMAST_PATH` overrides the hard-coded Freya path to Giovannozzi's modules.

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

**Absolute calibration is missing.** `RecognizedModes.amplitude` carries a
`# TODO check`, and it is load-bearing. `matplotlib.mlab.specgram(mode='complex')`
applies no window-power or sample-rate normalisation, and no coil area or
amplifier gain is divided out. The DIII-D counterpart carries an explicit
normalisation, a `Ts` factor, and a `×1e4` that lands the result in Gauss. Until
this is settled the ladder can only be expressed in units of each shot's own
pre-plasma noise floor, and thresholds cannot be compared with any published
number.

**Toroidal handedness is undetermined.** `Spectra.n_detection` projects onto
`exp(+i n phi)`, so it detects modes whose spectral phase runs as `exp(-i n phi)`.
A mode of the opposite sense is invisible to a positive-only search list — and,
worse, inflates the *ungated* amplitude of every n at once. Search both signs
(`saddle_extras.NTOR_BOTH_SIGNS`) until the convention is confirmed against a
shot with a known mode. DIII-D handles this explicitly by branching on
`sign(<Bt><Ip>)`; there is no such branch here yet. See `docs/` §9.6.

## Not yet written

Locked-mode detector (no counterpart to the DIII-D M-matrix path — the spectral
detector cannot see a DC island); joint annotation and pathway model; EPM/EPQ
feature extraction; the matched-time-base merge; the hazard model itself.

EPQ and EPM appear to be the MAST-U counterparts of EFIT02 and EFIT01, so the
matched-pair machinery should port with little change.
