import pytest


from ip_orch.core.energy_correction import apply_linear_correction
from ip_orch.core.energy_correction import compute_element_reference_energy_shift


def test_apply_linear_correction_optional_noop():
    assert apply_linear_correction(energy=10.0, natoms=5, a=None, b=1.0) == 10.0
    assert apply_linear_correction(energy=10.0, natoms=5, a=2.0, b=None) == 10.0


def test_apply_linear_correction_total_energy():
    assert apply_linear_correction(energy=10.0, natoms=5, a=2.0, b=1.0, mode="total_energy") == 21.0


def test_apply_linear_correction_per_atom():
    # E=10, N=5 => eps=2; eps_corr=2*2+1=5 => E_corr=25
    assert apply_linear_correction(energy=10.0, natoms=5, a=2.0, b=1.0, mode="per_atom") == 25.0


def test_apply_linear_correction_per_atom_requires_positive_natoms():
    with pytest.raises(ValueError):
        apply_linear_correction(energy=10.0, natoms=0, a=1.0, b=0.0, mode="per_atom")


def test_apply_linear_correction_rejects_unknown_mode():
    with pytest.raises(ValueError):
        apply_linear_correction(energy=10.0, natoms=5, a=1.0, b=0.0, mode="bad")  # type: ignore[arg-type]


def test_compute_element_reference_energy_shift():
    shift = compute_element_reference_energy_shift(
        ["Cu", "Cu", "C"], element_energies={"Cu": -3.0, "C": -1.0}
    )
    assert shift == pytest.approx(-7.0)


def test_compute_element_reference_energy_shift_requires_all_elements():
    with pytest.raises(KeyError):
        compute_element_reference_energy_shift(["Cu", "Fe"], element_energies={"Cu": -3.0})
