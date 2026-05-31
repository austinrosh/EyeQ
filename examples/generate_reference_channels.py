#!/usr/bin/env python
"""Generate the reference .s4p channel set into examples/data/.

One synthetic differential channel per reach class (112G). XSR/XSR+/VSR are
smooth (loss budget only); MR/LR carry reflection notches. Drop your own
measured .s4p next to these and point a config's ``channel_s4p`` at it — the
importer reads any 4-port with ports (1,2)=pair A, (3,4)=pair B.

Run:  python examples/generate_reference_channels.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eyeq.io.synth_channel import generate_all
from eyeq.io.touchstone import load_sdd21

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    paths = generate_all(DATA, generation="112G")
    print(f"wrote {len(paths)} reference channels to {DATA}/")
    for p in paths:
        f, h = load_sdd21(str(p))
        i28 = int(np.argmin(np.abs(f - 28e9)))
        i45 = int(np.argmin(np.abs(f - 45e9)))
        loss28 = -20 * np.log10(abs(h[i28]))
        loss45 = -20 * np.log10(abs(h[i45]))
        print(f"  {p.name:20s}  IL@28GHz={loss28:5.1f} dB  IL@45GHz={loss45:5.1f} dB")


if __name__ == "__main__":
    main()
