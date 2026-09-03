"""Run every test that works without Freya.

    python tests/run_offline_tests.py

These cover the flat-top windowing, the Schmitt trigger, and the coherence
retrofit.  Anything that needs the live database (check_equivalence.py,
run_master.py, saddle_analysis.py's batch sweep) is not exercised here.
"""
import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = ["test_flattop", "test_trigger", "test_coherence"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    failed = []
    for name in MODULES:
        print(f"\n=== {name} " + "=" * (56 - len(name)))
        rc = importlib.import_module(name).main()
        if rc:
            failed.append(name)

    # Also run each module as a script.  Importing loads every definition
    # before main() is called, which hides import-order bugs -- a helper
    # defined below the `if __name__` guard passes here and fails there.
    print("\n=== standalone execution " + "=" * 39)
    for name in MODULES:
        rc = subprocess.run([sys.executable, os.path.join(here, name + ".py")],
                            capture_output=True)
        mark = "ok" if rc.returncode == 0 else "FAILED"
        print(f"  {mark:>6}  python tests/{name}.py")
        if rc.returncode:
            failed.append(name + " (standalone)")
            print("      " + rc.stderr.decode().strip().splitlines()[-1])

    print("\n" + "=" * 64)
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"all {len(MODULES)} offline test modules passed, imported and standalone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
