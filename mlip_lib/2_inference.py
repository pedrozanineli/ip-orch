import numpy as np
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
import pickle
import os
import re
import warnings
import sys
from calculator import Calculator
import config

warnings.filterwarnings("ignore")

console = Console()
filename = os.path.basename(__file__)
console.print(Rule(f"[bold cyan]{filename} - Start"))

# category_name = sys.argv[1]
# calculator_name = sys.argv[2]

calculator_name = sys.argv[1]
model_name = sys.argv[2]
 
energy_per_atom_mp_plot = []
energy_per_atom_bulk_plot = []
mp_surface_energy_EV_PER_ANG2 = []
mp_surfaces_surf_energy_pred_EV_PER_ANG2 = []
dft_per_atom = []
pred_per_atom = []


calc = Calculator.get_calculator(calculator_name, model_name, models_path=config.MODELS_PATH)

# Estrutura de diretório simplificada
input_folder = os.path.join("..", "materials_project", "surfaces")
output_folder = os.path.join("data", calculator_name, "surfaces_energy_pred")
dft_folder = os.path.join("..", "materials_project", "dft")
save_path = os.path.join("data", calculator_name)

os.makedirs(output_folder, exist_ok=True)
os.makedirs(dft_folder, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

if any(os.path.isfile(os.path.join(output_folder, arq)) for arq in os.listdir(output_folder)):
    print(f"[INFO] {calculator_name} already calculated, skipping...")
    recalcular_mlip = 0
else:
    print(f"[INFO] starting inference")
    recalcular_mlip = 1

for pkl_file in os.listdir(input_folder):
    
    if not pkl_file.endswith(".pkl"):
        print('[WARNING] not a pkl file, skipping...')
        continue

    match = re.search(r"(mp-\d+)", pkl_file)
    if not match:
        print('[WARNING] could not extract task name, skipping...')
        continue
    task_name = match.group(1)

    # --- Carrega dados de superfície ---
    pkl_path = os.path.join(input_folder, pkl_file)
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)

    mp_surfaces_structure = data.get("mp_surfaces_structure", [])
    if not mp_surfaces_structure:
        print(f"[WARNING] no surface for {task_name}")
        continue

    mp_surfaces_nsites = np.array(data.get("mp_surfaces_nsites", []))
    mp_surfaces_areas = np.array(data.get("mp_surfaces_areas", []))
    mp_surface_energy_EV_PER_ANG2.append(np.array(data.get("mp_surface_energy_EV_PER_ANG2", [])))

    # --- Energia DFT por átomo ---
    dft_file = os.path.join(dft_folder, f"mp_surfaces_energy_dft_{task_name}.npy")
    if not os.path.exists(dft_file):
        print(f"[WARNING] could not find file for {task_name}")
        continue
    mp_energy, mp_energy_per_atom = np.load(dft_file)
    energy_per_atom_mp_plot.append(mp_energy_per_atom)

    # --- Energia bulk com MLIP (usa mp_bulk_structure se existir) ---
    mp_bulk_structure = data.get("mp_bulk_structure", None)
    if mp_bulk_structure is not None:
        ase_bulk = mp_bulk_structure.copy()
    else:
        ase_bulk = mp_surfaces_structure[0].copy()  # fallback
        print(f"[WARNING] no bulk found for {task_name}, using first surface as fallback")

    ase_bulk.calc = calc
    bulk_energy = ase_bulk.get_potential_energy()
    bulk_energy_per_atom = bulk_energy / len(ase_bulk)
    energy_per_atom_bulk_plot.append(bulk_energy_per_atom)

    # --- MLIP: energia predita para superfícies ---
    pred_file = os.path.join(output_folder, f"mp_surfaces_energy_pred_{task_name}-{calculator_name}.npy")
    if os.path.exists(pred_file):
        mp_surfaces_energy_pred = np.load(pred_file)
    else:
        mp_surfaces_energy_pred = np.array(
            [calc.get_potential_energy(struct) for struct in mp_surfaces_structure]
        )
        np.save(pred_file, mp_surfaces_energy_pred)

    surf_energy_pred_EV_PER_ANG2_calc = (
        mp_surfaces_energy_pred - bulk_energy_per_atom * mp_surfaces_nsites
    ) / (2 * mp_surfaces_areas)
    mp_surfaces_surf_energy_pred_EV_PER_ANG2.append(np.array(surf_energy_pred_EV_PER_ANG2_calc))

    mp_surfaces_tot_energy = (
        surf_energy_pred_EV_PER_ANG2_calc * (2 * mp_surfaces_areas)
        + mp_energy_per_atom * mp_surfaces_nsites
    )
    dft_per_atom.append(mp_surfaces_tot_energy / mp_surfaces_nsites)
    pred_per_atom.append(mp_surfaces_energy_pred / mp_surfaces_nsites)

# === Salva resultados no npz ===
np.savez(os.path.join(save_path, "results_all.npz"),
         energy_per_atom_mp_plot=np.array(energy_per_atom_mp_plot),
         energy_per_atom_bulk_plot=np.array(energy_per_atom_bulk_plot),
         mp_surface_energy_EV_PER_ANG2=np.array(mp_surface_energy_EV_PER_ANG2, dtype=object),
         mp_surfaces_surf_energy_pred_EV_PER_ANG2=np.array(mp_surfaces_surf_energy_pred_EV_PER_ANG2, dtype=object),
         dft_per_atom=np.array(dft_per_atom, dtype=object),
         pred_per_atom=np.array(pred_per_atom, dtype=object))

print(f"[INFO] calculations done and saved at {save_path}/results_all.npz")

