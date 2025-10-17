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

# category_name = sys.argv[1]
# calculator_name = sys.argv[2]

calculator_name = sys.argv[1]
model_name = sys.argv[2]

#console = Console()

#table = Table()
#table.add_column("Path", justify="center", style="cyan", no_wrap=True)
#table.add_row(cfg["conda_activate"])
#console.print(table)


#table = Table()
#table.add_column("Env Base", justify="center", style="cyan", no_wrap=True)
#table.add_row(cfg['base_env'])
#console.print(table)


#table = Table()
#table.add_column("Conda Env", justify="center", style="cyan", no_wrap=True)
#table.add_column("Modelo", justify="center", style="magenta")
#table.add_column("Status", justify="center", style="white")
#found = False
#for env, model in cfg["full_models"].items():
#    if not found:
#        if env == category_name and model == calculator_name:
            # Item atual → carregando
#            table.add_row(env, model, "⏳ Loading...")
#            found = True
#        else:
            # Já passou → check
#            table.add_row(env, model, "✅ Done")
#    else:
#        # Ainda vai rodar → pendente
#        table.add_row(env, model, "⌛ Pending")
#console.print(table)

#filename = os.path.basename(__file__)
#console.print(Rule(f"[bold cyan]{filename} - Start"))

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
output_folder = os.path.join("data", calculator_name, "surfaces_energy_pred")
dft_folder = os.path.join("..", "materials_project", "dft")
save_path = f'data/{calculator_name}/plots/'

os.makedirs('../materials_project', exist_ok=True)
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(dft_folder, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

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
            dft = mpr.tasks.get_data_by_id(
            name,
            fields=["task_id", "orig_inputs", "calcs_reversed", "output", "last_updated"]
            )
            task_name = dft.task_id
            mp_energy, mp_energy_per_atom = dft.entry.energy, dft.entry.energy_per_atom            
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

    
