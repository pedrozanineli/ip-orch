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

benchmark_name = sys.argv[1]
calculator_name = sys.argv[2]
model_name = sys.argv[3]

base_save_path = os.path.join("data", benchmark_name, calculator_name)
os.makedirs(base_save_path, exist_ok=True)

task_name_ = []
energy_per_atom_mp_plot = []
ref = []

console = Console()
filename = os.path.basename(__file__)
console.print(Rule(f"[bold cyan]{filename} - Start"))

if benchmark_name == 'surfaces':

    print(f"[INFO] plotting inference results for surfaces")

    # --- CORREÇÃO ---
    # Removidos os caminhos antigos (input_folder, output_folder, save_path)
    # Precisamos apenas do caminho para o NPZ e onde salvar os plots.

    # Caminho CORRETO para o arquivo de dados (onde o 2_inference.py salvou)
    npz_path = os.path.join(base_save_path, "results_all.npz")
    
    # Caminho CORRETO para salvar os plots (no mesmo diretório base)
    plots_save_path = base_save_path
    os.makedirs(plots_save_path, exist_ok=True)

    # Nome completo do arquivo de plot que será gerado
    plot_file_path = os.path.join(plots_save_path, f"comparacao_DFT_vs_{calculator_name}_superficie.png")

    # --- Lógica de "pular" CORRIGIDA ---
    # Verifica se o *arquivo de plot final* já existe
    if os.path.exists(plot_file_path):
        # Corrigido: Removida a variável 'category_name'
        print(f"[INFO] plots for surfaces/{calculator_name} done, skipping...")
        recalcular_plots = 0
    else: 
        recalcular_plots = 1

    if recalcular_plots == 1:

        print(f"[INFO] Gerando plots para surfaces...")

        # Adicionada verificação se o NPZ existe
        if not os.path.exists(npz_path):
            print(f"[ERROR] Arquivo de resultados não encontrado: {npz_path}")
            sys.exit(1)
            
        data = np.load(npz_path, allow_pickle=True)

        # print(data) # Descomente se precisar depurar

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
        # ... (seu código de plot 1 está correto) ...
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
        # ... (seu código de plot 2 está correto) ...
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
        # ... (seu código de plot 3 está correto) ...
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
        
        # --- CORREÇÃO no savefig ---
        # Salva no caminho de plot correto ('plot_file_path')
        plt.savefig(plot_file_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # --- CORREÇÃO no print ---
        # Imprime o caminho correto
        print(f"[INFO] figure saved at {plot_file_path}")

        # --- O código de salvar o CSV já estava CORRETO ---
        df_rmse = pd.DataFrame([{
            "benchmark": benchmark_name,
            "calculator_name": calculator_name,
            "model_name": model_name,
            "rmse_bulk_per_atom (meV)": round(rmse_bulk*1000, 4),
            "rmse_surface_total_per_atom (meV)": round(rmse_per_atom*1000, 4),
            "rmse_surface_energy (mJ/m^2)": round(rmse_surface*1000, 4)
            }])
        
        csv_path = os.path.join("data", "rmse_results.csv")

        if os.path.exists(csv_path):
            df_rmse.to_csv(csv_path, mode="a", header=False, index=False)
            print(f"[INFO] results added to: {csv_path}")
        else:
            df_rmse.to_csv(csv_path, index=False)
            print(f"[INFO] new file with results: {csv_path}")

# Em 3_plots.py

# (Use 'elif' aqui se você corrigiu da última vez)
elif benchmark_name == 'eos': 
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    import pandas as pd

    print("[INFO] Gerando plots para EOS...")

    results_file = os.path.join(base_save_path, "eos_results.npz")
    if not os.path.exists(results_file):
        print(f"[ERROR] Arquivo de resultados não encontrado: {results_file}")
        sys.exit(1)

    data = np.load(results_file, allow_pickle=True)
    mlip_results = data["mlip_results"]
    dft_results = data["dft_results"]

    # === CORREÇÃO ===
    # Removemos 'e0' (energia)
    # dft_e0 = [r["e0_per_atom"] for r in dft_results]
    # mlip_e0 = [r["e0_per_atom"] for r in mlip_results]

    dft_v0 = [r["v0_per_atom"] for r in dft_results]
    mlip_v0 = [r["v0_per_atom"] for r in mlip_results]

    dft_b0 = [r["b0_gpa"] for r in dft_results]
    mlip_b0 = [r["b0_gpa"] for r in mlip_results]

    # === CORREÇÃO ===
    # Mudamos o tamanho da figura para 2 painéis (de 15,6 para 10,6)
    plt.figure(figsize=(10, 6), dpi=300)

    # (A função make_parity_plot está perfeita, não mude)
    def make_parity_plot(subplot_idx, dft_data, mlip_data, title, xlabel, ylabel):
        # ... (código da função) ...
        # (copie a função que você já tinha aqui)
        rmse = np.sqrt(mean_squared_error(dft_data, mlip_data))
        # === CORREÇÃO ===
        # Mudamos o layout do subplot de (1, 3, idx) para (1, 2, idx)
        plt.subplot(1, 2, subplot_idx) 
        plt.scatter(dft_data, mlip_data, alpha=0.5, s=20, edgecolor="k")
        min_val = min(min(dft_data), min(mlip_data))
        max_val = max(max(dft_data), max(mlip_data))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.text(0.05, 0.95, f"RMSE = {rmse:.3f}",
                 transform=plt.gca().transAxes,
                 verticalalignment="top",
                 bbox=dict(facecolor="white", alpha=0.7))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.5)


    # === CORREÇÃO ===
    # Plot 1: Volume (índice 1)
    make_parity_plot(1, dft_v0, mlip_v0,
                     "Volume de Equilíbrio ($V_0$)",
                     "DFT $V_0$ (Å³/átomo)",
                     "MLIP $V_0$ (Å³/átomo)")

    # Plot 2: Bulk Modulus (índice 2)
    make_parity_plot(2, dft_b0, mlip_b0,
                     "Módulo de Compressibilidade ($B_0$)",
                     "DFT $B_0$ (GPa)",
                     "MLIP $B_0$ (GPa)")

    plt.tight_layout()
    plot_save_path = os.path.join(base_save_path, f"comparacao_eos_{calculator_name}.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico de EOS salvo em: {plot_save_path}")

    # === CORREÇÃO (Opcional, mas recomendado) ===
    # Salva os RMSEs corretos no CSV
    
    # (Calcule os RMSEs novamente, pois a função `make_parity_plot` não os retorna)
    rmse_v0 = np.sqrt(mean_squared_error(dft_v0, mlip_v0))
    rmse_b0 = np.sqrt(mean_squared_error(dft_b0, mlip_b0))

    df_rmse_eos = pd.DataFrame([{
        "benchmark": benchmark_name,
        "calculator_name": calculator_name,
        "model_name": model_name,
        # "rmse_e0_per_atom (meV)": ... # Removido
        "rmse_v0_per_atom (A^3)": round(rmse_v0, 4),
        "rmse_b0 (GPa)": round(rmse_b0, 4)
    }])
    
    csv_path = os.path.join("data", "rmse_results_eos.csv") 
    if os.path.exists(csv_path):
        df_rmse_eos.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_rmse_eos.to_csv(csv_path, index=False)
    print(f"[INFO] RMSEs do EOS salvos em: {csv_path}")
