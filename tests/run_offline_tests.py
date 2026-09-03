"""Run every test that works without Freya.

    python tests/run_offline_tests.py

These cover the flat-top windowing, the Schmitt trigger, and the coherence
retrofit.  Anything that needs the live database (check_equivalence.py,
run_master.py, saddle_analysis.py's batch sweep) is not exercised here.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = ["test_flattop", "test_trigger", "test_coherence"]


def main():
    failed = []
    for name in MODULES:
        print(f"\n=== {name} " + "=" * (56 - len(name)))
        rc = importlib.import_module(name).main()
        if rc:
            failed.append(name)

    print("\n" + "=" * 64)
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"all {len(MODULES)} offline test modules passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
