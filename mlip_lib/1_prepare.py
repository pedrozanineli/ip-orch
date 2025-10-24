import numpy as np
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
import csv
from pymatgen.core import Structure
from pymatgen.core import Lattice
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from monty.serialization import loadfn
from mp_api.client import MPRester
from ase import Atoms
from ase.build import bulk
from ase.units import m, J
ev_to_J = 1/J
ang_to_m = 1/m
import pickle
import os
import subprocess
import re
import warnings
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
import sys
import glob
from tqdm import tqdm

print(r"""
    __    _____    ____    ___            __  ___    __     ____    ____        
   / /   / ___/   /  _/   /   |          /  |/  /   / /    /  _/   / __ \   _____
  / /    \__ \    / /    / /| | ______  / /|_/ /   / /     / /    / /_/ /  / ___/
 / /___ ___/ /  _/ /    / ___ |/_____/ / /  / /   / /___ _/ /    / ____/  (__  ) 
/_____//____/  /___/   /_/  |_|       /_/  /_/   /_____//___/   /_/      /____/                                                                                   
""")

print()

warnings.filterwarnings("ignore")
api_mp = os.getenv("API_MP")


benchmark_name = sys.argv[1]
calculator_name = sys.argv[2]
model_name = sys.argv[3]

base_save_path = os.path.join("data", benchmark_name, calculator_name)
os.makedirs(base_save_path, exist_ok=True)

print(f'[INFO] Benchmark: {benchmark_name} | Calculator: {calculator_name} | Model {model_name}')

if benchmark_name == "surfaces":

    print(f'[INFO] surface evaluation using {calculator_name} with model {model_name}')

    recalcular_mlip = 1
    recalcular_superficie = 1
    recalcular_dft = 1
    recalcular_plots = 1
    dft_per_atom = []
    pred_per_atom = []
    mp_surface_energy_EV_PER_ANG2 = []
    mp_surfaces_surf_energy_pred_EV_PER_ANG2 = []
    task_name_ = []
    ref = []
    energy_per_atom_bulk_plot = []
    energy_per_atom_mp_plot = []

    # Define o caminho onde a figura será salva
    # Verifica se o diretório existe; se não, cria (incluindo subpastas)

    # Input and Output Dir
    input_folder = os.path.join("..", "materials_project", "surfaces")
    output_folder = os.path.join(base_save_path, "surfaces_energy_pred")
    dft_folder = os.path.join("..", "materials_project", "dft")

    os.makedirs('../materials_project', exist_ok=True)
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(dft_folder, exist_ok=True)

    if any(os.path.isfile(os.path.join(dft_folder, arq)) for arq in os.listdir(dft_folder)) and any(os.path.isfile(os.path.join(input_folder, arq)) for arq in os.listdir(input_folder)):
        print("[INFO] MP already retrieved, skipping...")
        recalcular_dft = 0
    else: recalcular_dft = 1

    if recalcular_dft == 1:
        print("[INFO] retrieving MP data...")
        tasks_doc = []
        with MPRester(api_mp) as mpr:
            surface_properties_doc = mpr.materials.surface_properties.search()
            for i in range(len(surface_properties_doc)):
                name = surface_properties_doc[i].material_id
                # === CORREÇÃO ===
                # Trocando o endpoint problemático 'mpr.tasks' pelo 'mpr.summary'
                summary_doc = mpr.summary.get_data_by_id(name)
                
                # Verificamos se o doc tem os dados que precisamos
                if not (summary_doc.energy_per_atom and summary_doc.nsites):
                    print(f"[WARNING] Não foi possível obter dados de energia para {name}. Pulando.")
                    continue # Pula para o próximo material no loop
                
                # Recriamos os mesmos dados que o código antigo precisava
                mp_energy_per_atom = summary_doc.energy_per_atom
                mp_energy = summary_doc.energy_per_atom * summary_doc.nsites # Calculamos a energia total
                
                np.save(f'../materials_project/dft/mp_surfaces_energy_dft_{name}.npy', [mp_energy, mp_energy_per_atom])
        print("[INFO] MP data retrieved successfully")
        
    # Verifica se os arquivos de superfície já foram criado
    if any(os.path.isfile(os.path.join(input_folder, arq)) for arq in os.listdir(input_folder)):
        print("[INFO] surfaces already created, skipping...")
        recalcular_superficie = 0
    else: recalcular_superficie = 1

    if recalcular_superficie == 1:
        print("[INFO] calculating surfaces...")
        with MPRester(api_mp) as mpr:
            surface_properties_doc = mpr.materials.surface_properties.search()
            
        for i in range(len(surface_properties_doc)):
            name = surface_properties_doc[i].material_id
            mp_surfaces = surface_properties_doc[i].surfaces
            
            mp_surfaces_miller_index = [m.miller_index for m in mp_surfaces]
            mp_surface_energy_J_PER_M2 = np.array([m.surface_energy for m in mp_surfaces])
            mp_surface_energy_EV_PER_ANG2 = [m.surface_energy_EV_PER_ANG2 for m in mp_surfaces]
            mp_surfaces_is_reconstructed = [m.is_reconstructed for m in mp_surfaces]
            mp_surfaces_structure = [
                AseAtomsAdaptor.get_atoms(CifParser.from_str(m.structure).parse_structures()[0])
                for m in mp_surfaces
            ]
            mp_surfaces_nsites = np.array([len(struct) for struct in mp_surfaces_structure])
            mp_surfaces_areas = np.array([
                np.linalg.norm(np.cross(struct.cell[0], struct.cell[1]))
                for struct in mp_surfaces_structure
            ])

            # === Novo: obtendo bulk real ===
            try:
                bulk_structure_pm = mpr.get_structure_by_material_id(name)
                mp_bulk_structure = AseAtomsAdaptor.get_atoms(bulk_structure_pm)
            except Exception as e:
                print(f"[WARNING] it was not possible to retrieve the bulk for {name}: {e}")
                mp_bulk_structure = None

            # Criando um dicionário com todas as variáveis
            data_dict = {
                "mp_surfaces": mp_surfaces,
                "mp_surfaces_miller_index": mp_surfaces_miller_index,
                "mp_surface_energy_J_PER_M2": mp_surface_energy_J_PER_M2,
                "mp_surface_energy_EV_PER_ANG2": mp_surface_energy_EV_PER_ANG2,
                "mp_surfaces_is_reconstructed": mp_surfaces_is_reconstructed,
                "mp_surfaces_structure": mp_surfaces_structure,
                "mp_surfaces_nsites": mp_surfaces_nsites,
                "mp_surfaces_areas": mp_surfaces_areas,
                "mp_bulk_structure": mp_bulk_structure
            }
            
            # Salvando o dicionário em um arquivo pickle
            with open(f"../materials_project/surfaces/mp_surfaces_data_{name}.pkl", "wb") as f:
                pickle.dump(data_dict, f)
            
        print("[INFO] surfaces calculated successfully")

elif benchmark_name == "eos":
    print("[INFO] Preparando dados de referência DFT para EOS...")
    dft_data_folder = os.path.join("..", "materials_project", "eos_dft")
    os.makedirs(dft_data_folder, exist_ok=True)
    dft_file = os.path.join(dft_data_folder, "eos_dft_data.pkl")

    if os.path.exists(dft_file):
        print("[INFO] Dados DFT para EOS já baixados.")
    else:
        print("[INFO] Baixando dados de elasticidade e estruturas do Materials Project...")
        from mp_api.client import MPRester
        import pickle

        api_mp = os.getenv("API_MP")
        dft_references = []

        with MPRester(api_mp) as mpr:
            # === CORREÇÃO ===
            # Trocamos 'k_vrh' e 'energy_per_atom' por 'bulk_modulus'
            elasticity_docs = mpr.materials.elasticity.search()
            for doc in elasticity_docs:
                # === CORREÇÃO ===
                # Verificamos se 'bulk_modulus' e o sub-campo 'k_vrh' existem
                if doc.structure and doc.bulk_modulus and doc.bulk_modulus.k_vrh: 
                    dft_references.append({
                        "material_id": doc.material_id.string,
                        "dft_structure": doc.structure,
                        # "dft_energy_per_atom" foi removido
                        "dft_volume_per_atom": doc.structure.volume / doc.structure.num_sites,
                        # Acessamos o 'k_vrh' de dentro do 'bulk_modulus'
                        "dft_bulk_modulus_gpa": doc.bulk_modulus.k_vrh
                    })

        with open(dft_file, "wb") as f:
            pickle.dump(dft_references, f)

        print(f"[INFO] {len(dft_references)} estruturas de referência salvas em {dft_file}")

else:
    print(f"[ERROR] Benchmark '{benchmark_name}' not supported")
    sys.exit(1)
    
