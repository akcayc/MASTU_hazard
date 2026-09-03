# Offline tests

    python tests/run_offline_tests.py

Everything here runs without Freya. `stubs/` shadows the six modules that exist
only on the cluster (`pyuda`, `omaha_coils`, `mode_functions`,
`saddle_geometry`, `pickup_coil_data`, `sxr_geometry`), so the tests exercise
Giovannozzi's **real** `SaddleFull` / `Spectra` / `RecognizedModes` classes on
synthetic data.

| Test | Covers |
|---|---|
| `test_flattop.py` | Window construction, both Ip polarities, rejection paths, asymmetric ramps |
| `test_trigger.py` | Threshold crossing, debounce, back-dating, NaN handling |
| `test_coherence.py` | Coherence array shape; amplitude bit-identical to upstream; the gate separating an injected mode; the handedness trap |

`test_coherence.py` is the one that matters. It injects a known n=2 mode at
10 kHz into white noise across 8 coils at 200 kHz, and asserts that
`amplitude_with_coherence` reproduces `RecognizedModes.amplitude()` exactly
while additionally recovering coherence ≈ 0.999 for the injected n against a
noise floor near 2/N_c.

It then repeats with the toroidal phase reversed and asserts the mode is *not*
detected — documenting that a mode of the wrong handedness collapses the
coherence while still inflating the ungated amplitude by 10–17× for every n.

Not covered here: anything needing the live database. Use `check_equivalence.py`
on Freya for that.
