"""
DatasetLoader
"""

# data.py
import numpy as np
import pandas as pd

from Config import Experiment, Paths


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SYNTH_CATEGORIES = {
    # physics
    "hubble", "newton", "rydberg", "idealgas", "kepler",
    "bode", "schechter", "leavitt", "planck",
    # nguyen
    "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12",
    # other
    "expreal",
}

_REMOTE_DROP_COLS = ["system:index", "QA60", ".geo", "date"]


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICS GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsGenerator:
    """Generates synthetic (X, y) pairs for physics law benchmarks."""

    def __init__(self, n: int, seed: int):
        self.n   = n
        self.rng = np.random.default_rng(seed)

    def generate(self, formula_id: str) -> tuple[pd.DataFrame, pd.Series]:
        method = getattr(self, f"_{formula_id}", None)
        if method is None:
            raise ValueError(f"Unknown physics formula: {formula_id!r}")
        return method()

    def _hubble(self):
        d = self.rng.uniform(1, 1000, self.n)
        return pd.DataFrame({"d": d}), pd.Series(70.0 * d, name="v")

    def _newton(self):
        G  = 6.674e-11
        m1 = self.rng.uniform(1e24, 1e30, self.n)
        m2 = self.rng.uniform(1e24, 1e30, self.n)
        r  = self.rng.uniform(1e8,  1e12, self.n)
        return pd.DataFrame({"m1": m1, "m2": m2, "r": r}), pd.Series(G * m1 * m2 / r**2, name="F")

    def _rydberg(self):
        R  = 1.097e7
        n1 = self.rng.integers(1, 5, self.n).astype(float)
        n2 = n1 + self.rng.integers(1, 5, self.n).astype(float)
        return pd.DataFrame({"n1": n1, "n2": n2}), pd.Series(R * (1/n1**2 - 1/n2**2), name="inv_lambda")

    def _idealgas(self):
        R  = 8.314
        P  = self.rng.uniform(1e4, 1e6, self.n)
        nc = self.rng.uniform(0.1, 10,  self.n)
        T  = self.rng.uniform(200, 1000, self.n)
        return pd.DataFrame({"P": P, "n": nc, "T": T}), pd.Series(nc * R * T / P, name="V")

    def _kepler(self):
        G, M_sun = 6.674e-11, 1.989e30
        a = self.rng.uniform(0.1, 50, self.n) * 1.496e11
        return pd.DataFrame({"a": a}), pd.Series(2 * np.pi * np.sqrt(a**3 / (G * M_sun)), name="T")

    def _bode(self):
        nv = np.arange(0, 9, dtype=float)
        return pd.DataFrame({"n": nv}), pd.Series(0.4 + 0.3 * 2**nv, name="a")

    def _schechter(self):
        phi_star, L_star, alpha = 1.5e-2, 1e10, -1.1
        L    = np.clip(self.rng.exponential(L_star, self.n), 1e6, 1e13)
        phi  = phi_star * (L / L_star)**alpha * np.exp(-L / L_star)
        mask = np.isfinite(phi) & (phi > 0)
        return pd.DataFrame({"L": L[mask]}), pd.Series(phi[mask], name="phi")

    def _leavitt(self):
        P = self.rng.uniform(1, 100, self.n)
        return pd.DataFrame({"P": P}), pd.Series(-2.81 * np.log10(P) - 1.43, name="M")

    def _planck(self):
        h, c, k = 6.626e-34, 3e8, 1.381e-23
        nu      = self.rng.uniform(1e11, 3e14, self.n)
        T       = self.rng.uniform(1000, 30000, self.n)
        exp_arg = np.clip(h * nu / (k * T), 0, 700)
        B       = 2 * h * nu**3 / c**2 / (np.expm1(exp_arg) + 1e-300)
        mask    = np.isfinite(B) & (B > 0)
        return pd.DataFrame({"nu": nu[mask], "T": T[mask]}), pd.Series(B[mask], name="B")

    def _expreal(self):
        x = self.rng.uniform(0, 2, self.n)
        return pd.DataFrame({"x": x}), pd.Series(np.exp(x) + np.exp(-x) + x**2, name="y")


# ══════════════════════════════════════════════════════════════════════════════
#  NGUYEN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class NguyenGenerator:
    """Generates synthetic (X, y) pairs for the Nguyen SR benchmarks."""

    def __init__(self, n: int, seed: int):
        self.n   = n
        self.rng = np.random.default_rng(seed)

    def generate(self, formula_id: str) -> tuple[pd.DataFrame, pd.Series]:
        method = getattr(self, f"_{formula_id}", None)
        if method is None:
            raise ValueError(f"Unknown Nguyen formula: {formula_id!r}")
        return method()

    def _n1(self):
        x = self.rng.uniform(-1, 1, self.n)
        return pd.DataFrame({"x": x}), pd.Series(x**3 + x**2 + x, name="y")

    def _n2(self):
        x = self.rng.uniform(-1, 1, self.n)
        return pd.DataFrame({"x": x}), pd.Series(x**4 + x**3 + x**2 + x, name="y")

    def _n3(self):
        x = self.rng.uniform(-1, 1, self.n)
        return pd.DataFrame({"x": x}), pd.Series(x**5 + x**4 + x**3 + x**2 + x, name="y")

    def _n4(self):
        x = self.rng.uniform(-1, 1, self.n)
        return pd.DataFrame({"x": x}), pd.Series(1.2329*x**6 + 547.139*x**5 + 1892*x**4 + 2.1*x**3 + 9182*x**2 + 1298*x +12.38, name="y")

    def _n5(self):
        x = self.rng.uniform(-1, 1, self.n)
        return pd.DataFrame({"x": x}), pd.Series(np.sin(x**2) * np.cos(x) - 1, name="y")

    def _n6(self):
        x = self.rng.uniform(-1, 1, self.n)
        return pd.DataFrame({"x": x}), pd.Series(np.sin(x) + np.sin(x + x**2), name="y")

    def _n7(self):
        x = self.rng.uniform(0, 2, self.n)
        return pd.DataFrame({"x": x}), pd.Series(np.log(x + 1) + np.log(x**2 + 1), name="y")

    def _n8(self):
        x = self.rng.uniform(0, 4, self.n)
        return pd.DataFrame({"x": x}), pd.Series(np.sqrt(x), name="y")

    def _n9(self):
        x  = self.rng.uniform(-1, 1, self.n)
        yv = self.rng.uniform(-1, 1, self.n)
        return pd.DataFrame({"x": x, "y": yv}), pd.Series(np.sin(x) + np.sin(yv**2), name="z")

    def _n10(self):
        x  = self.rng.uniform(-1, 1, self.n)
        yv = self.rng.uniform(-1, 1, self.n)
        return pd.DataFrame({"x": x, "y": yv}), pd.Series(2 * np.sin(x) * np.cos(yv), name="z")

    def _n11(self):
        x  = self.rng.uniform(1, 2, self.n)
        yv = self.rng.uniform(1, 2, self.n)
        return pd.DataFrame({"x": x, "y": yv}), pd.Series(x**yv, name="z")

    def _n12(self):
        x  = self.rng.uniform(-1, 1, self.n)
        yv = self.rng.uniform(-1, 1, self.n)
        return pd.DataFrame({"x": x, "y": yv}), pd.Series(x**4 - x**3 + yv**2 / 2 - yv, name="z")


# ══════════════════════════════════════════════════════════════════════════════
#  REMOTE SENSING LOADER
# ══════════════════════════════════════════════════════════════════════════════

class RemoteSensingLoader:
    """Loads the Sentinel-2 CSV and computes spectral index targets (all lowercase)."""

    _FORMULAS = {
        "ndvi":    lambda df: (df["B8"] - df["B4"]) / (df["B8"] + df["B4"]),
        "savi":    lambda df: (df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + 0.5) * 1.5,
        "bsi":     lambda df: (df["B11"] + df["B4"] - df["B8"] - df["B2"]) / (df["B11"] + df["B4"] + df["B8"] + df["B2"]),
        "mndwi":   lambda df: (df["B3"] - df["B11"]) / (df["B3"] + df["B11"]),
        "ndmi":    lambda df: (df["B8"] - df["B11"]) / (df["B8"] + df["B11"]),
        "msi":     lambda df: df["B11"] / df["B8"],
        "ndwi":    lambda df: (df["B3"] - df["B8"]) / (df["B3"] + df["B8"]),
        "bai":     lambda df: 1 / ((0.1 - df["B4"])**2 + (0.06 - df["B8"])**2),
        "awei_sh": lambda df: df["B2"] + 2.5*df["B3"] - 1.5*(df["B11"] + df["B12"]) - 0.25*df["B8"],
        "awei_nsh":lambda df: 4*(df["B3"] - df["B11"]) - (0.25*df["B8"] + 2.75*df["B12"]),
        "wi2015":  lambda df: 1.7204 + 171*(df["B2"]+df["B3"]+df["B4"]) - 3*df["B2"]*df["B3"] - 1.8*df["B2"]*df["B4"] - 48*df["B3"]*df["B4"] - 0.8*df["B8"]*df["B11"],
        "ndre_b5": lambda df: (df["B8A"] - df["B5"]) / (df["B8A"] + df["B5"]),
        "ndre_b6": lambda df: (df["B8A"] - df["B6"]) / (df["B8A"] + df["B6"]),
        "evi2":    lambda df: 2.5 * (df["B8"] - df["B4"]) / (df["B8"] + 2.4*df["B4"] + 1),
        "vari":    lambda df: (df["B3"] - df["B4"]) / (df["B3"] + df["B4"] - df["B2"]).replace(0, np.nan),
        "nirv":    lambda df: df["B8"] * ((df["B8"] - df["B4"]) / (df["B8"] + df["B4"])),
    }

    def __init__(self, data_path: str = Paths.DATA_CSV):
        self.data_path = data_path

    def load(self, target: str) -> tuple[pd.DataFrame, pd.Series]:
        if target not in self._FORMULAS:
            raise ValueError(f"Unknown remote sensing target: {target!r}")

        df = pd.read_csv(self.data_path)
        df = df.drop(columns=[c for c in _REMOTE_DROP_COLS if c in df.columns])

        df[target] = self._FORMULAS[target](df)
        df = df.dropna(subset=[target])
        df = df[np.isfinite(df[target])]

        y = df[target].copy()
        X = df.drop(columns=[target]).select_dtypes(include=[np.number])
        return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

class DatasetLoader:
    """Single entry point — routes any target to the right generator/loader."""

    def __init__(self, data_path: str = Paths.DATA_CSV):
        self._remote = RemoteSensingLoader(data_path)

    def load(
        self,
        target:   str,
        run_id:   int = 0,
        strategy: str = "xgboost",
    ) -> tuple[pd.DataFrame, pd.Series]:
        seed = Experiment.RANDOM_SEED + run_id
        n    = Experiment.N_SYNTH_SAMPLES

        if target in SYNTH_CATEGORIES:
            if target.startswith("n") and target[1:].isdigit():
                return NguyenGenerator(n, seed).generate(target)
            return PhysicsGenerator(n, seed).generate(target)

        return self._remote.load(target)