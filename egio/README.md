# Vendored dependency

`saddle_data.py` is **not ours**. It is from E. Giovannozzi (ENEA), 2024 —
"Read and elaborate data from the magnetics of MASTU". It is included here
because `saddle_extras.py` and `saddle_analysis.py` are useless without it and
because the offline tests exercise its real classes.

Check with the author before making this repository public or redistributing.

## Missing modules

`saddle_data.py` imports five further modules that live only on Freya and are
**not** in this repository:

    omaha_coils        the coil table: name, name_slow, name_fast, phi, pol, orient
    saddle_geometry    saddle-loop geometry
    mode_functions     zeros_spectrum
    pickup_coil_data   load_signal
    sxr_geometry       soft X-ray geometry

`omaha_coils` is the important one: it is the array geometry, and therefore the
MAST-U counterpart of DIII-D's hard-coded probe angle list. Nothing can be
validated against real data without it.

`tests/stubs/` contains minimal stand-ins for all five (plus `pyuda`), enough to
run the offline test suite. They are **not** substitutes for the real modules —
`mode_functions.zeros_spectrum` in particular is reconstructed from its call
site and its exact upstream behaviour may differ.

## Known bugs in `saddle_data.py`

Fix at the source rather than working around them:

1. `load_omaha` / `load_base`: the `except pyuda.cpyuda.ServerException` clause
   resets `time` and `data` to empty arrays, so a single failing channel
   destroys everything already loaded.
2. Both then use `idt = np.flatnonzero(np.diff(time) < 1e-4)` only as
   `time[idt[0]:idt[-1]]`, which discards the mask and takes a contiguous slice.
3. `SaddleFull.spectrum` has its uniform-dt assertion commented out. Combined
   with (2) this suggests irregular time bases were being worked around; worth
   understanding before trusting the FFT stage.
4. `RecognizedModes.amplitude` never applies `threshold()`, and the driver
   passes `v_min=0.0`, so nothing is ever rejected. This is why the shipped
   amplitude follows the noise floor. `saddle_extras.amplitude_with_coherence`
   is the response.
