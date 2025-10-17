import sys
import subprocess
from rich.console import Console
from rich.rule import Rule

# Importações para a lógica do "trabalhador"
from ase import Atoms
import config
from calculator import Calculator

# ================================================================= #
# == PASSO 1: DEFINA SUA LÓGICA DE CÁLCULO EM UMA FUNÇÃO ÚNICA   == #
# ================================================================= #
#
# Você só precisa editar esta função.
# Ela recebe o nome do modelo e o objeto da calculadora ASE já pronto.
# Coloque aqui qualquer cálculo que você queira comparar entre os MLIPs.
#
def logica_do_calculo(calculator_name, ase_calculator):
    """
    Esta função contém a ciência que você quer executar.
    Exemplo: Cálculo da energia de atomização do Nitrogênio.
    """
    print(f"--> Executando lógica para: {calculator_name.upper()}")

    # 1. Átomo de Nitrogênio isolado
    atom = Atoms('N')
    atom.calc = ase_calculator
    e_atom = atom.get_potential_energy()

    # 2. Molécula de Nitrogênio (N2)
    d = 1.1  # Distância internuclear em Angstrom
    molecule = Atoms('2N', [(0.0, 0.0, 0.0), (0.0, 0.0, d)])
    molecule.calc = ase_calculator
    e_molecule = molecule.get_potential_energy()

    # 3. Energia de ligação e atomização
    e_binding = e_molecule - (2 * e_atom)

    print(f'Energia do átomo de N (eV): {e_atom:.4f}')
    print(f'Energia da molécula de N2 (eV): {e_molecule:.4f}')
    print(f'Energia de Ligação (eV): {e_binding:.4f}')
    print(f'Energia de Atomização (eV): {-e_binding:.4f}')


# ================================================================= #
# == A PARTIR DAQUI, O SCRIPT SE GERENCIA SOZINHO. NÃO É PRECISO  == #
# == EDITAR MAIS NADA.                                           == #
# ================================================================= #

def _run_worker(calculator_name, model_name):
    """
    Função 'Trabalhador': Inicializa uma calculadora e executa a lógica científica.
    Esta função só é chamada quando o script é executado com o argumento '--worker'.
    """
    print(f"--> [Worker Mode] Inicializando a calculadora '{model_name}'...")
    try:
        ase_calculator = Calculator.get_calculator(
            calculator_name=calculator_name,
            model_name=model_name,
            models_path=config.MODELS_PATH
        )
        print("--> Calculadora inicializada com sucesso!")

        # Chama a função que você definiu acima com a calculadora pronta
        logica_do_calculo(calculator_name, ase_calculator)

    except Exception as e:
        print(f"❌ ERRO ao processar '{calculator_name}': {e}")
        sys.exit(1)


def _run_orchestrator():
    """
    Função 'Gerente': Lê a configuração e chama a si mesmo (como trabalhador)
    para cada MLIP em seu respectivo ambiente Conda.
    """
    console = Console()
    console.print(Rule("[bold green]Orquestrador de Comparação de MLIPs[/bold green]"))

    # Pega o caminho para o script que está rodando atualmente
    script_path = __file__

    for calculator_name, model_name in config.FULL_MODELS.items():
        console.print(Rule(f"Processando: {calculator_name.upper()} ({model_name})"))

        command = [
            "conda", "run", "-n", calculator_name,
            "python", script_path,
            "--worker",  # O argumento mágico que ativa o modo trabalhador
            calculator_name,
            model_name
        ]

        command_str = " ".join(command)
        print(f"🚀 Executando comando: {command_str}")
        try:
            subprocess.run(command, check=True)
            print(f"✅ Sucesso para o ambiente {calculator_name}.\n")
        except Exception as e:
            print(f"❌ ERRO ao executar no ambiente '{calculator_name}': {e}", file=sys.stderr)
            sys.exit(1)
            
    console.print(Rule("[bold blue]Todos os cálculos foram finalizados.[/bold blue]"))


if __name__ == "__main__":
    # O script verifica se foi chamado com o argumento '--worker'
    if "--worker" in sys.argv:
        # Se sim, ele atua como TRABALHADOR
        # Os argumentos esperados são: --worker <calculator_name> <model_name>
        worker_calculator_name = sys.argv[2]
        worker_model_name = sys.argv[3]
        _run_worker(worker_calculator_name, worker_model_name)
    else:
        # Se não, ele atua como GERENTE (comportamento padrão)
        _run_orchestrator()
