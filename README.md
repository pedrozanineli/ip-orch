
<h1 align="left">IP-Orch</h1>

<p align="left">
  <!-- <strong>Python package for orchestrating machine learning interatomic potentials.</strong> -->
  Python package for orchestrating machine learning interatomic potentials.
</p>

<p align="left">
  <a href="https://github.com/pedrozanineli/ip-orch/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <!-- <a href="https://github.com/yourusername/yourproject/releases">
    <img src="https://img.shields.io/badge/version-1.0.0-green.svg" alt="Version">
  </a> -->
  <!-- <a href="https://github.com/yourusername/yourproject/actions">
    <img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Status">
  </a> -->
</p>

## Table of Contents

- [Project](#project)
- [Installation](#installation)
- [Usage](#usage)
    - [Configure and register models](#configure-and-register-models)
    - [Run a script across models](#run-a-script-across-models)
    - [Reference energy correction](#reference-energy-correction)
- [Contributions and suggestions](#contributions-and-suggestions)
- [License](#license)

<!-- - [Contributing](#contributing) -->

## Project

With the rapid growth of Machine Learning Interatomic Potentials (MLIPs) with different architectures and software ecosystems, a fragmented ladscape has been created, where models are often tied to incompatible dependencies and heterogeneous interfaces. In this scenario, IP-Orch is a lightweight Python package for orchestrating multiple MLIPs, allowing reproducible benchmarking and systematic comparison across models difficult in practice using unified ASE-based workflow.

### Features

- Orchestrates a single ASE script across multiple MLIP environments/models.
- Keeps model aliases + configuration in a local config store.
- Provides an interactive setup flow to discover Conda environments.
- Optional post-processing energy correction (linear and element-reference shift).
- Rich terminal output + per-model success/broken status.

## Installation

`IP-Orch` can be installed directly from the source repository using `pip`:

```bash
# Clone the repository
git clone https://github.com/pedrozanineli/ip-orch

# Navigate to the project directory
cd ip-orch

# Install IP-Orch and dependencies
pip install .
```

## Usage

### Configure and register models

IP-Orch configuration can be both be done manually or automatically.

```bash
# Interactive discovery/setup
ip-orch --configure

# Manually add (env, model) pairs
ip-orch --add mace mace_mp
ip-orch --add orb orb_v3
```

Available and configured models can be checked using the following:

```bash
# List known model aliases
ip-orch --available-models
```

### Run a script across models

<!-- Example: `examples/calculators_test.py`. -->
<!-- (or `mlip_entry(...)`). -->

The script must be defined as a `main()` function and receive `calculator_name, calc` as parameters, as shown in the following example:

```python
import sys
from ase import Atoms
from ase import build 

def main(calculator_name, ase_calculator):
    molecule = build.molecule('H2O')
    molecule.calc = ase_calculator
    e_molecule = molecule.get_potential_energy()
    print(f'Molecule energy {e_molecule:.4f} eV')
```

With the configured script and environments, the script can be executed with multiple models. The current available models are 1) using all MLIPs available in a given environment (e.g. all models available from MACE) or 2) specified models configured in the package.

```bash
# 1) Run all configured models for selected environments
ip-orch --run examples/calculators_test.py --envs mace,orb

# 2) Select specific model aliases
ip-orch --run examples/calculators_test.py --models mace-mp,orb-v3
```

### Reference energy correction

Taking into consideration that Machine Learning Interatomic Potentials can be trained with different datasets, their predicted absolute energies may not be directly comparable due to shifts in reference energy. To address this, `IP-Orch` provides an optional reference energy correction scheme.

This correction computes element-wise reference energies using the same MLIP (e.g., isolated atoms in a large non-periodic box) and subtracts their contribution from the total energy of a structure, based on its composition. The set of elements can be specified via the `--correction_elements` flag, in which case the reference energies are automatically evaluated from the MLIP itself. This approach aligns the energy zero across different models, enabling consistent comparison of quantities such as formation, surface, and interaction energies. This correction can also be combined with an optional linear adjustment of the energy (via `--energy-linear-a`, `--energy-linear-b`, and `--energy-linear-mode`) to account for systematic scaling differences between models.

```bash
# Linear correction
ip-orch --run examples/calculators_test.py \
        --envs mace \
        --energy-linear-a 1.02 \
        --energy-linear-b -0.10 \
        --energy-linear-mode total_energy

# Element-reference shift (auto computed from the MLIP itself)
ip-orch --run examples/calculators_test.py \
        --envs mace \
        --correction_elements Cu

# Disable any correction
ip-orch --run examples/calculators_test.py \
        --envs mace \
        --no-energy-correction
```

## Reference

A paper is under development.

## Contributions and suggestions

For bugs or feature requests, please use [GitHub Issues](https://github.com/pedrozanineli/ip-orch/issues).

## License

IP-Orch is published and distributed under the MIT License.