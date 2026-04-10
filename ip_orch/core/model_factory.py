"""
Factory to instantiate ASE calculators for common MLIP models.

Usage:
    from ip_orch.core.model_factory import ModelFactory
    calc = ModelFactory.create("mace-mp-0", device="cuda", models_path="~/pretrained")

Notes:
    - This runs inside the target Conda env where the model is installed.
    - Some models expect local weight files; pass models_path accordingly.
"""

import os
import re
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


class ModelFactory:
    """Instantiate calculators by model alias.

    The alias list mirrors common names used in benchmarks and repos.
    """

    _ALIASES = {
        "pet-oam-xl": "pet_oam_xl",
        "nequip-oam-xl": "nequip_oam_xl",
        "matris-10m-oam": "matris_10m_oam",
        "sevennet-omni": "sevennet_omni",
        "nequip-oam-l": "nequip_oam_l",
        "tace-v1-oam-m": "tace_v1_oam_m",
        "grace-2l-oam": "grace_2l_oam",
        "grace-1l-oam": "grace_1l_oam",
        "grace-2l-mp-r6": "grace_2l_mp",
        "orb-v3": "orb_v3",
        "orb-v2": "orb_v2",
        "orb-v2-mptrj": "orb_v2_mptrj",
        "dpa-3.1-3m-ft": "dpa_3_1_3m_ft",
        "dpa-3.1-mptrj": "dpa_3_1_mptrj",
        "mace-mpa-0": "mace_mpa_0",
        "mace-mp-0": "mace_mp_0",
        "mace-mp": "mace_mp",
        "matris-10m-mp": "matris_10m_mp",
        "mattersim-v1": "mattersim_v1",
        "eqnorm-mptrj": "eqnorm_mptrj",
        "nequix-mp-1-pft": "nequix_mp_pft",
        "nequix-mp": "nequix_mp",
        "nequip-mp-l": "nequip_mp_l",
        "allegro-mp-l": "allegro_mp_l",
        "sevennet-l3i5": "sevennet_l3i5",
        "hienet": "hienet",
        "chgnet": "chgnet",
        "m3gnet": "m3gnet",
    }

    @staticmethod
    def _device_to_str(device: Optional[str]) -> str:
        if device:
            return device
        if _HAS_TORCH:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return os.environ.get("MLIP_DEVICE", "cpu")

    @classmethod
    def create(cls, model_name: str, device: Optional[str] = None, models_path: Optional[str] = None):
        key = cls._ALIASES.get(model_name, _norm(model_name))
        device = cls._device_to_str(device)

        # UPET / PET-OAM-XL
        if key == "pet_oam_xl":
            from upet.calculator import UPETCalculator
            return UPETCalculator(model="pet-oam-xl", version="1.0.0", device=device)

        # NequIP OAM-XL
        if key == "nequip_oam_xl":
            from nequip.ase import NequIPCalculator
            path = os.path.expanduser("/home/p.zanineli/pretrained/nequip/nequip-OAM-XL.nequip.pt2")
            return NequIPCalculator.from_compiled_model(compile_path=path, device=device)

        # MatRIS 10M OAM
        if key == "matris_10m_oam":
            from matris.applications.base import MatRISCalculator
            return MatRISCalculator(model='matris_10m_oam', device=device)

        # SevenNet Omni
        if key == "sevennet_omni":
            from sevenn.calculator import SevenNetCalculator
            return SevenNetCalculator(model='7net-omni', device=device, modal='mpa')

        # NequIP OAM-L
        if key == "nequip_oam_l":
            from nequip.ase import NequIPCalculator
            path = os.path.expanduser("/home/p.zanineli/pretrained/nequip/nequip-OAM-L.nequip.pt2")
            return NequIPCalculator.from_compiled_model(compile_path=path, device=device)

        # TACE v1 OAM-M
        if key == "tace_v1_oam_m":
            from tace.foundations import tace_foundations
            from tace.interface.ase import TACEAseCalc, add_dispersion  # noqa: F401
            model = tace_foundations["TACE-v1-OAM-M"]
            return TACEAseCalc(model=model, dtype="float32", device=device, level=0)

        # GRACE 2L OAM
        if key == "grace_2l_oam":
            from tensorpotential.calculator import grace_fm
            return grace_fm('GRACE-2L-OAM', device=device)

        # ORB V3
        if key == "orb_v3":
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.inference.calculator import ORBCalculator
            orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
                device=device,
                precision="float32-high",
            )
            return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)

        # DeepMD DPA3 OpenLAM
        if key == "dpa_3_1_3m_ft":
            from deepmd.calculator import DP
            path = os.path.expanduser("/home/p.zanineli/pretrained/deepmd/dpa3-openlam.pth")
            return DP(model=path)

        # MACE MPA-0
        if key == "mace_mpa_0":
            from mace.calculators import mace_mp
            return mace_mp(model="medium-mpa-0", device=device)

        # MatRIS 10M MP
        if key == "matris_10m_mp":
            from matris.applications.base import MatRISCalculator
            return MatRISCalculator(model='matris_10m_mp', device=device)

        # MatterSim v1
        if key == "mattersim_v1":
            from mattersim.forcefield import MatterSimCalculator
            return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)

        # GRACE 1L OAM
        if key == "grace_1l_oam":
            from tensorpotential.calculator import grace_fm
            return grace_fm('GRACE-1L-OAM', device=device)

        # Eqnorm MPTrj
        if key == "eqnorm_mptrj":
            from eqnorm.calculator import EqnormCalculator
            return EqnormCalculator(model_name='eqnorm', model_variant='eqnorm-mptrj', device=device)

        # Nequix MP-1 PFT (jax)
        if key == "nequix_mp_pft":
            from nequix.calculator import NequixCalculator
            return NequixCalculator(model_name="nequix-mp-1-pft", backend="jax")

        # NequIP MP-L
        if key == "nequip_mp_l":
            from nequip.ase import NequIPCalculator
            path = os.path.expanduser("/home/p.zanineli/pretrained/nequip/nequip-MP-L.nequip.pt2")
            return NequIPCalculator.from_compiled_model(compile_path=path, device=device)

        # Nequix MP-1 torch
        if key == "nequix_mp":
            from nequix.calculator import NequixCalculator
            return NequixCalculator("nequix-mp-1", backend="torch")

        # Allegro MP-L
        if key == "allegro_mp_l":
            from nequip.ase import NequIPCalculator
            path = os.path.expanduser("/home/p.zanineli/pretrained/nequip/allegro-MP-L.nequip.pt2")
            return NequIPCalculator.from_compiled_model(compile_path=path, device=device)

        # DeepMD DPA3 MPTrj
        if key == "dpa_3_1_mptrj":
            from deepmd.calculator import DP
            path = os.path.expanduser("/home/p.zanineli/pretrained/deepmd/dpa3-mptrj.pth")
            return DP(model=path)

        # SevenNet l3i5
        if key == "sevennet_l3i5":
            from sevenn.calculator import SevenNetCalculator
            return SevenNetCalculator(model='SevenNet-l3i5', device=device, modal='mpa')

        # HIENet
        if key == "hienet":
            from hienet.hienet_calculator import HIENetCalculator
            return HIENetCalculator()

        # GRACE 2L MP r6
        if key == "grace_2l_mp":
            from tensorpotential.calculator import grace_fm
            return grace_fm('GRACE-2L-MP-r6', device=device)

        # MACE MP 0 (large)
        if key == "mace_mp_0":
            from mace.calculators import mace_mp
            return mace_mp(model="large", device=device)

        # ORB V2
        if key == "orb_v2":
            from orb_models.forcefield import pretrained
            try:
                from orb_models.forcefield.inference.calculator import ORBCalculator
            except Exception:
                from orb_models.forcefield.calculator import ORBCalculator  # fallback
            _res = pretrained.orb_v2(device=device, precision="float32-high")
            try:
                orbff, atoms_adapter = _res
                return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)
            except Exception:
                orbff = _res
                return ORBCalculator(orbff, device=device)

        # ORB V2 MPTrj
        if key == "orb_v2_mptrj":
            from orb_models.forcefield import pretrained
            try:
                from orb_models.forcefield.inference.calculator import ORBCalculator
            except Exception:
                from orb_models.forcefield.calculator import ORBCalculator  # fallback
            _res = pretrained.orb_mptraj_only_v2(device=device, precision="float32-high")
            try:
                orbff, atoms_adapter = _res
                return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)
            except Exception:
                orbff = _res
                return ORBCalculator(orbff, device=device)

        # CHGNet
        if key == "chgnet":
            from chgnet.model import CHGNetCalculator
            return CHGNetCalculator()

        # M3GNet
        if key == "m3gnet":
            # Some installations require LD_PRELOAD; caller can set it as needed.
            from m3gnet.models import M3GNetCalculator
            return M3GNetCalculator()
