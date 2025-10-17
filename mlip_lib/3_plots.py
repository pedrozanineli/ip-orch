import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
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
import pandas as pd

# Ignorar todos os warnings do Python
warnings.filterwarnings("ignore")
api_mp = os.getenv("API_MP")

# category_name = sys.argv[1]
# calculator_name = sys.argv[2]

calculator_name = sys.argv[1]
model_name = sys.argv[2]

task_name_ = []
energy_per_atom_mp_plot = []
ref = []

console = Console()
filename = os.path.basename(__file__)
console.print(Rule(f"[bold cyan]{filename} - Start"))

# Input and Output Dir
input_folder = os.path.join("..", "materials_project", "surfaces")
output_folder = os.path.join("data", category_name, calculator_name, "surfaces_energy_pred")
dft_folder = os.path.join("..", "materials_project", "dft")

# Estrutura de diretório simplificada
save_path = os.path.join("data", calculator_name) # Caminho base para os resultados

npz_path = os.path.join(save_path, "results_all.npz")
plots_save_path = os.path.join(save_path, "plots") # Subpasta para os plots
os.makedirs(plots_save_path, exist_ok=True)


if any(os.path.isfile(os.path.join(save_path, arq)) for arq in os.listdir(save_path)):
    print(f"[INFO] plots for {category_name} done, skipping...")
    # recalcular_plots = int(input("Recalcular plots: 0 - Não | 1 - Sim: "))
    recalcular_plots = 0
else: recalcular_plots = 1

if recalcular_plots == 1:

    print(f"[INFO] plotting inference results")

    data = np.load(npz_path, allow_pickle=True)

    print(data)

    energy_per_atom_mp_plot = data["energy_per_atom_mp_plot"]
    energy_per_atom_bulk_plot = data["energy_per_atom_bulk_plot"]
    mp_surface_energy_EV_PER_ANG2 = data["mp_surface_energy_EV_PER_ANG2"]
    mp_surfaces_surf_energy_pred_EV_PER_ANG2 = data["mp_surfaces_surf_energy_pred_EV_PER_ANG2"]

    dft_per_atom = np.concatenate(data["dft_per_atom"])
    pred_per_atom = np.concatenate(data["pred_per_atom"])

    # J/mˆ2
    all_dft_surfaces = np.concatenate(mp_surface_energy_EV_PER_ANG2) * 16.02
    all_pred_surfaces = np.concatenate(mp_surfaces_surf_energy_pred_EV_PER_ANG2) * 16.02

    plt.figure(figsize=(15, 6), dpi=300)

    # 1. Energia bulk
    rmse_bulk = np.sqrt(mean_squared_error(energy_per_atom_mp_plot, energy_per_atom_bulk_plot))
    plt.subplot(131)
    plt.scatter(energy_per_atom_mp_plot, energy_per_atom_bulk_plot, alpha=0.5, s=20, edgecolor="k")
    plt.plot([min(energy_per_atom_mp_plot), max(energy_per_atom_mp_plot)],
            [min(energy_per_atom_mp_plot), max(energy_per_atom_mp_plot)], "r--")
    plt.text(0.05, 0.95, f"RMSE = {rmse_bulk:.3f} eV/atom",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7))
    plt.xlabel("Energia Bulk - DFT (eV/atom)")
    plt.ylabel("Energia Bulk - MLIP (eV/atom)")
    plt.grid(True, linestyle="--", alpha=0.5)

    # 2. Energia por átomo (superficie)
    rmse_per_atom = np.sqrt(mean_squared_error(dft_per_atom, pred_per_atom))
    plt.subplot(132)
    plt.scatter(dft_per_atom, pred_per_atom, alpha=0.5, s=20, edgecolor="k")
    plt.plot([min(dft_per_atom), max(dft_per_atom)],
            [min(dft_per_atom), max(dft_per_atom)], "r--")
    plt.text(0.05, 0.95, f"RMSE = {rmse_per_atom:.3f} eV/atom",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7))
    plt.xlabel("Energia por átomo - DFT (eV)")
    plt.ylabel("Energia por átomo - MLIP (eV)")
    plt.grid(True, linestyle="--", alpha=0.5)

    # 3. Energia de superfície (J/m²)
    rmse_surface = np.sqrt(mean_squared_error(all_dft_surfaces, all_pred_surfaces))
    plt.subplot(133)
    plt.scatter(all_dft_surfaces, all_pred_surfaces, alpha=0.5, s=20, edgecolor="k")
    plt.plot([min(all_dft_surfaces), max(all_dft_surfaces)],
            [min(all_dft_surfaces), max(all_dft_surfaces)], "r--")
    plt.text(0.05, 0.95, f"RMSE = {rmse_surface:.3f} J/m²",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7))
    plt.xlabel("Energia de Superfície - DFT (J/m²)")
    plt.ylabel("Energia de Superfície - MLIP (J/m²)")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"comparacao_DFT_vs_{category_name}_superficie.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] figure saved at {save_path}")

    df_rmse = pd.DataFrame([{
    "category_name": category_name,
    "calculator_name": calculator_name,
    "rmse_bulk_per_atom (eV)": round(rmse_bulk*1000,4),
    "rmse_surface_total_per_atom (eV)": round(rmse_per_atom*1000,4),
    "rmse_surface_energy (J/mˆ2)": round(rmse_surface*1000,4)
    }])

    # === Caminho do CSV ===
    csv_path = os.path.join("data", "rmse_results.csv")

    # === Salva ou acrescenta ===
    if os.path.exists(csv_path):
        df_rmse.to_csv(csv_path, mode="a", header=False, index=False)
        print(f"[INFO] results added to: {csv_path}")
    else:
        df_rmse.to_csv(csv_path, index=False)
        print(f"[INFO] new file with results: {csv_path}")

