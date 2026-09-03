"""Locate E. Giovannozzi's MAST-U magnetics modules and put them on sys.path.

Every entry point in this package needs `saddle_data`, and `saddle_extras`
additionally needs `mode_functions`; `saddle_data` itself then pulls in four
more (`omaha_coils`, `saddle_geometry`, `pickup_coil_data`, `sxr_geometry`).
None of those live in this repository -- they are on Freya.  Rather than
repeat a hard-coded path in four files, import this module first:

    import giopath  # noqa: F401
    from saddle_data import load_omaha_slow

Search order, highest priority first:

    $GIOMAST_PATH      colon-separated, like $PATH; overrides everything
    /home/cm0459/...   the usual location on Freya
    <repo>/egio        the vendored copy -- saddle_data.py only

Directories are APPENDED to sys.path, not inserted.  The offline test suite
puts `tests/stubs` at the front so it can shadow the cluster-only modules, and
appending here keeps that working.

If something is missing, call `report()` for a diagnostic that names the
modules and the directories searched -- considerably more useful than a bare
ModuleNotFoundError several imports deep.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: the usual location of Giovannozzi's analysis directory on Freya
FREYA_DEFAULT = "/home/cm0459/Python/gioMAST"

#: modules that must be importable before anything in this package will run
REQUIRED = [
    "saddle_data",       # the library itself
    "mode_functions",    # zeros_spectrum
    "omaha_coils",       # coil table: phi, pol, orient, name_slow/name_fast
    "saddle_geometry",
    "pickup_coil_data",
    "sxr_geometry",
]


def candidates():
    """Directories to search, highest priority first."""
    out = []
    env = os.environ.get("GIOMAST_PATH", "")
    out.extend(p for p in env.split(os.pathsep) if p)
    out.append(FREYA_DEFAULT)
    out.append(os.path.join(HERE, "egio"))
    return out


def install():
    """Append every existing candidate directory to sys.path.  Idempotent."""
    added = []
    for d in candidates():
        d = os.path.expanduser(d)
        if os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)
            added.append(d)
    return added


def missing():
    """Which of REQUIRED cannot be imported, without importing them."""
    import importlib.util
    out = []
    for name in REQUIRED:
        try:
            if importlib.util.find_spec(name) is None:
                out.append(name)
        except (ImportError, ValueError):
            out.append(name)
    return out


def report(stream=sys.stderr):
    """Print what was found and what was not.  Returns True if all present."""
    gone = missing()
    print("Giovannozzi module search:", file=stream)
    for d in candidates():
        d = os.path.expanduser(d)
        mark = "found   " if os.path.isdir(d) else "no such "
        print(f"  [{mark}] {d}", file=stream)
    if not gone:
        print("  all required modules importable", file=stream)
        return True
    print(f"\n  MISSING: {', '.join(gone)}", file=stream)
    print("\n  These live on Freya and are not in this repository.  Either run"
          "\n  there, or point GIOMAST_PATH at a directory holding them:"
          f"\n      export GIOMAST_PATH={FREYA_DEFAULT}"
          "\n  For offline work without the database, use the stubs instead:"
          "\n      python tests/run_offline_tests.py", file=stream)
    return False


install()
