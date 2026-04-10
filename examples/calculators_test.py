import sys
from ase.build import bulk

def main(calculator_name, ase_calculator):
    
    test_atoms = bulk('Cu', 'fcc', a=3.6)

    if ase_calculator is None:
        print('[error] calculator object is none')
        return
    else: test_atoms.calc = ase_calculator
    
    try:
        e = test_atoms.get_potential_energy()
        print(f'[done] successfully evaluated energy: {e} eV')
    except Exception as exc:
        print("[error] calculator imported, but failed to compute energy on H atom.")
        return

if __name__ == "__main__":
    print("This script is intended to be run via: ip-orch --run examples/calculators_test.py")
