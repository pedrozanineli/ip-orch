import warnings

warnings.filterwarnings("ignore")

class Calculator:

    def get_calculator(calculator_name, model_name=None, models_path=None):
        if models_path is None:
            raise ValueError("O caminho para os modelos ('models_path') não foi fornecido.")

        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if calculator_name == "deepmd":

            """
            available models:
            - DPA3-v2-OpenLAM
            - DPA3-v2-MPtrj

            if not specified, DPA3-v2-OpenLAM
            """

            from deepmd.calculator import DP
            
            if model_name == 'DPA3-v2-OpenLAM' or not model_name:
                return DP(model=f'{models_path}/dpa3-openlam.pth')
            elif model_name == 'DPA3-v2-MPtrj':
                return DP(model=f'{models_path}/dpa3-mptrj.pth')
        
        elif calculator_name == "eqnorm":
            
            """
            available models:
            - Eqnorm-MPtrj
            """

            from eqnorm.calculator import EqnormCalculator
            return EqnormCalculator(model_name='eqnorm', model_variant='eqnorm-mptrj', device=device)
            
        elif calculator_name == "fair_chem":

            """
            available models:
            - eSEN-30M-OAM
            - eSEN-30M-MP
            
            not avaliable:
            - eqV2 M
            - eqV2 S DeNS

            if not specified, eqV2 M
            """

            from fairchem.core import pretrained_mlip, FAIRChemCalculator
            predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
            calc = FAIRChemCalculator(predictor, task_name="omat")
            return calc

        elif calculator_name == "grace":

            """
            available models:
            - GRACE-2L-OAM
            - GRACE-1L-OAM
            - GRACE-2L-MPtrj
            if not specified, GRACE-2L-OAM
            """

            from tensorpotential.calculator.foundation_models import grace_fm, GRACEModels
            if model_name == 'GRACE-1L-OAM':
                return grace_fm(GRACEModels.GRACE_1L_OAM)
            elif model_name == 'GRACE-2L-MPtrj':
                # comp. matbench model
                return grace_fm(GRACEModels.GRACE_2L_MP_r6)
            else:
                return grace_fm(GRACEModels.GRACE_2L_OAM)

        elif calculator_name == "hienet":

            """
            available models:
            - HIENet
            """
            
            from hienet.hienet_calculator import HIENetCalculator
            return HIENetCalculator(model=f'{models_path}/HIENet-V3.pth')

        elif calculator_name == "mace":

            """
            available models:
            - MACE-MPA-0
            - MACE-MP-0
            
            if not specified, MACE-MPA-0
            """
            
            from mace.calculators import mace_mp
            from mace.calculators import MACECalculator
                        
#            if model_name == 'MACE-MP-0':
#                return MACECalculator(model_paths=f'{models_path}/mace-mp.model', device=device
#            else:
            return mace_mp(model='large', device=device.type, default_dtype='float64')

        elif calculator_name == "mattersim":

            """
            available models: MatterSim-v1.0.0-5M
            """

            from mattersim.forcefield import MatterSimCalculator
            
            return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)

        elif calculator_name == "orb-v3":

            """"
            available models:
            - ORB
            - ORB MPTrj
            """

            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
            
            # if model_name == 'orb-mptrj': model = pretrained.orb_mptraj_only_v2
            # else: model = pretrained.orb_v2(device=device)

            # print('ORB V3')

            if model_name == 'ORB-V3':
                model = pretrained.orb_v3_conservative_inf_omat(
                    device=device,
                    precision="float32-high",
                )

            elif model_name == 'ORB-V2-MPTrj':
                model = pretrained.orb_mptraj_only_v2(
                    device=device,
                )
            
            return ORBCalculator(model, device=device)

        elif calculator_name == "sevenn":

            """"
            available models:
            - SevenNet-MF-ompa
            - SevenNet-l3i5
            """

            from sevenn.calculator import SevenNetCalculator

            if not model_name:
                model_name = '7net-mf-ompa'
            elif model_name == '7net-l3i5':
                return SevenNetCalculator(model=model_name)

            return SevenNetCalculator(model=model_name)

        else:
            raise ValueError("calculator not supported...")

    @staticmethod
    def help():
        """
        list of supported calculators and models
        last updated march 26th

        1. "deepmd": DPA3-v2-OpenLAM (default), DPA3-v2-MPtrj
        2. "fair-chem": eSEN-30M-OAM, eSEN-30M-MP, eqV2 M (default), eqV2 S DeNS
        3. "grace": GRACE-2L-OAM (default), GRACE-1L-OAM, GRACE-2L-MPtrj        
        4. "mace": MACE-MPA-0 (default), MACE-MP-0
        5. "mattersim": MatterSim-v1.0.0-5M
        6. "orb": ORB (default), ORB MPTrj
        7. "sevenn": SevenNet-MF-ompa (default), SevenNet-l3i5
        
        usage:
        > calc = Calculator.get_calculator("deepmd", model_name="DPA3-v2-MPtrj")
        > calc = Calculator.get_calculator("mace")        
        """
        print(Calculator.help.__doc__)
