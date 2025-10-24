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

benchmark_name = sys.argv[1]
calculator_name = sys.argv[2]
model_name = sys.argv[3]

base_save_path = os.path.join("data", benchmark_name, calculator_name)
os.makedirs(base_save_path, exist_ok=True)
 
energy_per_atom_mp_plot = []
energy_per_atom_bulk_plot = []
mp_surface_energy_EV_PER_ANG2 = []
mp_surfaces_surf_energy_pred_EV_PER_ANG2 = []
dft_per_atom = []
pred_per_atom = []


calc = Calculator.get_calculator(calculator_name, model_name, models_path=config.MODELS_PATH)

if benchmark_name == 'surfaces':
    # Estrutura de diretório simplificada
    input_folder = os.path.join("..", "materials_project", "surfaces")
    output_folder = os.path.join(base_save_path, "surfaces_energy_pred")
    dft_folder = os.path.join("..", "materials_project", "dft")
    save_path = base_save_path

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

# Em 2_inference.py

elif benchmark_name == 'eos':
    import pickle
    import numpy as np
    from ase.eos import EquationOfState
    from ase.units import GPa
    from pymatgen.io.ase import AseAtomsAdaptor
    from calculator import Calculator
    import config

    print("[INFO] Iniciando inferência para EOS...")
    
    # calc = ... (calc já foi definido no topo do script)

    dft_data_folder = os.path.join("..", "materials_project", "eos_dft")
    dft_file = os.path.join(dft_data_folder, "eos_dft_data.pkl")
    with open(dft_file, "rb") as f:
        dft_references = pickle.load(f)

    mlip_results = []
    dft_results = []
    strains = np.linspace(0.95, 1.05, 11) 

    for ref in dft_references:
        try:
            atoms_base = AseAtomsAdaptor.get_atoms(ref["dft_structure"])
            volumes = []
            energies = []

            for strain_factor in strains:
                atoms_strained = atoms_base.copy()
                atoms_strained.set_cell(atoms_base.get_cell() * (strain_factor ** (1./3.)), scale_atoms=True)
                atoms_strained.calc = calc
                volumes.append(atoms_strained.get_volume() / len(atoms_strained))
                energies.append(atoms_strained.get_potential_energy() / len(atoms_strained))

            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0_mlip, e0_mlip, B_mlip_ev_ang3 = eos.fit()
            
            B_mlip_gpa = B_mlip_ev_ang3 * 160.21766208

            # === CORREÇÃO ===
            # Removemos 'e0_per_atom' dos resultados salvos
            mlip_results.append({
                # "e0_per_atom": e0_mlip, # Removido
                "v0_per_atom": v0_mlip,
                "b0_gpa": B_mlip_gpa
            })
            
            # === CORREÇÃO ===
            # Removemos 'e0_per_atom' dos dados DFT
            dft_results.append({
                # "e0_per_atom": ref["dft_energy_per_atom"], # Removido
                "v0_per_atom": ref["dft_volume_per_atom"],
                "b0_gpa": ref["dft_bulk_modulus_gpa"]
            })
        
        except Exception as e:
            print(f"[WARNING] Falha ao calcular EOS para {ref['material_id']}: {e}")

    results_file = os.path.join(base_save_path, "eos_results.npz")
    np.savez(results_file, 
             mlip_results=mlip_results, 
             dft_results=dft_results)

    print(f"[INFO] Resultados de EOS salvos em {results_file}")
