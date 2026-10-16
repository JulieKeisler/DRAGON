"""Import convention of the experiment code: the checked-out ``lib/dragon`` is
the ``dragon`` package, never an installed ``dragon-autodl``; ``leaderboard/``
(the frozen experiment harness) is importable for the equivalence tests."""
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "leaderboard")):
    if p not in sys.path:
        sys.path.insert(0, p)

import lib.dragon  # noqa: E402
sys.modules["dragon"] = sys.modules["lib.dragon"]

warnings.filterwarnings("ignore")
