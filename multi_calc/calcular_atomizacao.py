# Importações necessárias
from ase import Atoms
from rich.console import Console
from rich.rule import Rule

# Importe seus próprios módulos: o arquivo de configuração e a fábrica de calculadoras
import config
from calculator import Calculator

def calcular_energia_atomizacao(calculator_name, ase_calculator):
    """
    Função que calcula a energia de atomização para a molécula de N2
    usando uma calculadora ASE específica.
    """
    print(f"--> Executando cálculo para: {calculator_name.upper()}")

    # 1. Átomo de Nitrogênio isolado
    atom = Atoms('N')
    atom.calc = ase_calculator
    e_atom = atom.get_potential_energy()

    # 2. Molécula de Nitrogênio (N2)
    d = 1.1  # Distância internuclear em Angstrom
    molecule = Atoms('2N', [(0.0, 0.0, 0.0), (0.0, 0.0, d)])
    molecule.calc = ase_calculator
    e_molecule = molecule.get_potential_energy()

    # 3. Energia de atomização
    # É a energia necessária para quebrar a molécula em átomos separados.
    # e_atomization = (Energia dos produtos) - (Energia dos reagentes)
    # e_atomization = (2 * e_atom) - e_molecule
    # O valor que você calculou (-e_atomization) é a energia de ligação.
    e_binding = e_molecule - (2 * e_atom)

    print(f'Energia do átomo de Nitrogênio: {e_atom:5.2f} eV')
    print(f'Energia da molécula de Nitrogênio: {e_molecule:5.2f} eV')
    print(f'Energia de Ligação: {e_binding:5.2f} eV')
    print(f'Energia de Atomização: {-e_binding:5.2f} eV')


def main():
    """
    Função principal que orquestra a execução para todos os MLIPs definidos no config.
    """
    console = Console()
    console.print(Rule("[bold green]Início do Script de Comparação de MLIPs[/bold green]"))

    # Itera sobre o dicionário de modelos do seu arquivo de configuração
    for calculator_name, model_name in config.FULL_MODELS.items():
        console.print(Rule(f"Processando: {calculator_name.upper()} ({model_name})"))
        
        try:
            # Esta é a etapa chave: usamos sua classe Calculator para obter a calculadora
            # com base no nome lido do arquivo de configuração.
            print(f"--> Inicializando a calculadora '{model_name}'...")
            calc = Calculator.get_calculator(
                calculator_name=calculator_name, 
                model_name=model_name, 
                models_path=config.MODELS_PATH
            )
            print("--> Calculadora inicializada com sucesso!")

            # Chama a função para realizar os cálculos com a calculadora criada
            calcular_energia_atomizacao(calculator_name, calc)

        except Exception as e:
            # Captura e exibe qualquer erro que possa ocorrer ao carregar um modelo
            print(f"❌ ERRO ao processar '{calculator_name}': {e}")
            
    console.print(Rule("[bold blue]Fim do Script[/bold blue]"))


if __name__ == "__main__":
    main()
