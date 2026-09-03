"""Put the package root, the vendored egio module, and the offline stubs on
sys.path, so the tests run without Freya (no pyuda, no MAST-U database)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

for p in (os.path.join(HERE, "stubs"),      # stubs first: they shadow Freya-only modules
          os.path.join(ROOT, "egio"),
          ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
