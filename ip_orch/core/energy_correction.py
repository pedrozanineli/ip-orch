from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, Optional, TypeVar


LinearCorrectionMode = Literal["total_energy", "per_atom"]

TCalc = TypeVar("TCalc")


def apply_linear_correction(
    energy: float,
    natoms: int,
    a: Optional[float],
    b: Optional[float],
    mode: LinearCorrectionMode = "total_energy",
) -> float:
    """Apply a linear correction to an MLIP-predicted energy.

    The correction is optional: if either ``a`` or ``b`` is ``None``, the input
    energy is returned unchanged.

    Modes:
        - ``total_energy``: E_corr = a * E_mlip + b
        - ``per_atom``:
            eps_mlip = E_mlip / N
            eps_corr = a * eps_mlip + b
            E_corr = N * eps_corr
    """

    if a is None or b is None:
        return energy

    if mode == "total_energy":
        return a * energy + b

    if mode == "per_atom":
        if natoms <= 0:
            raise ValueError("natoms must be > 0 for per_atom correction")
        per_atom = energy / natoms
        per_atom_corr = a * per_atom + b
        return natoms * per_atom_corr

    raise ValueError(f"Unknown correction mode: {mode!r}")


def compute_element_reference_energy_shift(
    symbols: Iterable[str],
    *,
    element_energies: dict[str, float],
) -> float:
    """Compute a baseline energy shift from per-element reference energies.

    The returned value is ``sum_i n_i * E_ref[element_i]``.
    """

    counts = Counter(symbols)
    shift = 0.0
    for sym, n in counts.items():
        if sym not in element_energies:
            raise KeyError(f"Missing reference energy for element {sym!r}.")
        shift += float(n) * float(element_energies[sym])
    return shift


def wrap_reference_energy_correction(
    calc: TCalc,
    *,
    element_energies: Optional[dict[str, float]],
) -> TCalc:
    """Optionally correct energies by subtracting element reference energies.

    The correction is applied as:

        E_corr = E_mlip - sum_i n_i * E_ref[element_i]

    Only energy-like outputs are modified ("energy" and, if present, "free_energy").
    Forces and any other properties are untouched.
    """

    if not element_energies:
        return calc

    try:
        from ase.calculators.calculator import Calculator, all_changes
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "ASE is required to wrap calculators for energy correction"
        ) from exc

    base_calc = calc
    element_energies_local = dict(element_energies)

    class _ElementReferenceEnergyCorrectionCalculator(Calculator):
        implemented_properties = getattr(base_calc, "implemented_properties", ["energy"])

        def __init__(self):
            super().__init__()

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):  # type: ignore[override]
            base_calc.calculate(atoms, properties, system_changes)
            self.results = dict(getattr(base_calc, "results", {}))

            if atoms is None:
                return

            shift = compute_element_reference_energy_shift(
                atoms.get_chemical_symbols(), element_energies=element_energies_local
            )

            if "energy" in self.results:
                raw = self.results["energy"]
                self.results["energy_mlip"] = raw
                self.results["energy"] = raw - shift

            if "free_energy" in self.results:
                raw = self.results["free_energy"]
                self.results["free_energy_mlip"] = raw
                self.results["free_energy"] = raw - shift

        def __getattr__(self, name):
            return getattr(base_calc, name)

    return _ElementReferenceEnergyCorrectionCalculator()  # type: ignore[return-value]


def wrap_linear_energy_correction(
    calc: TCalc,
    *,
    a: Optional[float],
    b: Optional[float],
    mode: LinearCorrectionMode = "total_energy",
) -> TCalc:
    """Optionally wrap an ASE calculator to correct only the energy result.

    If either ``a`` or ``b`` is ``None``, the calculator is returned unchanged.
    """

    if a is None or b is None:
        return calc

    try:
        from ase.calculators.calculator import Calculator, all_changes
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "ASE is required to wrap calculators for energy correction"
        ) from exc

    base_calc = calc

    class _LinearEnergyCorrectionCalculator(Calculator):
        implemented_properties = getattr(base_calc, "implemented_properties", ["energy"])

        def __init__(self):
            super().__init__()

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):  # type: ignore[override]
            base_calc.calculate(atoms, properties, system_changes)
            self.results = dict(getattr(base_calc, "results", {}))

            natoms = len(atoms) if atoms is not None else 0
            # Keep compatibility: expose corrected energy as the main output key
            # while preserving original MLIP energy for downstream scripts.
            if "energy" in self.results:
                raw = self.results["energy"]
                self.results["energy_mlip"] = raw
                self.results["energy"] = apply_linear_correction(raw, natoms, a, b, mode)
            elif "free_energy" in self.results and "energy" in properties:
                # Some calculators provide only free_energy; keep it untouched and
                # still provide a corrected "energy" output.
                raw = self.results["free_energy"]
                self.results["energy_mlip"] = raw
                self.results["energy"] = apply_linear_correction(raw, natoms, a, b, mode)

        def __getattr__(self, name):
            return getattr(base_calc, name)

    return _LinearEnergyCorrectionCalculator()  # type: ignore[return-value]
