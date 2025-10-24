import subprocess
import sys

# Importe seu novo arquivo de configuração
import config

def run_script_in_env(env_name, script_path, *args):
    """
    Executa um script Python em um ambiente Conda específico.
    (Esta função permanece a mesma de antes)
    """
    # ... (código da função run_script_in_env sem alterações) ...
    command = [
        "conda", "run", "-n", env_name,
        "python3", script_path, *args
    ]
    command_str = " ".join(command)
    print(f"🚀 Executando: {command_str}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"\n❌ ERRO: O comando falhou com o código de saída {process.returncode}.", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"✅ Sucesso: {script_path} concluído.\n")
    except FileNotFoundError:
        print(f"❌ ERRO: Comando 'conda' não encontrado. Verifique se o Conda está no seu PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Função principal para orquestrar a execução dos scripts.
    """
    base_env = config.BASE_ENV
    full_models = config.FULL_MODELS
    benchmarks = config.BENCHMARKS

    if not all ([base_env, full_models, benchmarks]):
        print("ERROR: Verify 'BASE_ENV', 'FULL_MODELS' or 'BENCHMARKS' not setted at 'config.py'.", file=sys.stderr)
        sys.exit(1)

    for benchmark_name in benchmarks:
        print(f"\n{'='*60}")
        print(f" Benchmark initialized: {benchmark_name.upper()}")
        print(f"\n{'='*60}")

        for calculator_name, model_name in full_models.items():
            if model_name is None:
                print(f" Pulando calculadora '{calculator_name}' pois não há modelo definido.")
                continue
                
            print(f"\n{'='*50}")
            print(f" Processando calculadora: {calculator_name.upper()} (Modelo: {model_name})")
            print(f"{'='*50}\n")
            
            # Etapa 1: Preparação - Passa o nome da calculadora e do modelo
            run_script_in_env(base_env, '1_prepare.py', benchmark_name, calculator_name, model_name)
            
            # Etapa 2: Inferência - O ambiente é o nome da calculadora. Passa a calculadora e o modelo.
            run_script_in_env(calculator_name, '2_inference.py', benchmark_name, calculator_name, model_name)
            
            # Etapa 3: Plots - Passa o nome da calculadora e do modelo
            run_script_in_env(base_env, '3_plots.py', benchmark_name, calculator_name, model_name)
            
            print(f"--- Pipeline para a calculadora {calculator_name} @ {benchmark_name}  finalizado. ---\n")

if __name__ == "__main__":
    main()

