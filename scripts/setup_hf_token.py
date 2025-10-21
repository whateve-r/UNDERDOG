"""
Script para configurar el token de HuggingFace

Uso:
    Método 1 (Input directo):
        poetry run python scripts/setup_hf_token.py
    
    Método 2 (Variable de entorno):
        $env:HF_TOKEN = 'tu_token_aqui'
        poetry run python scripts/setup_hf_token.py
    
    Método 3 (Argumento):
        poetry run python scripts/setup_hf_token.py --token tu_token_aqui

Obtén tu token en: https://huggingface.co/settings/tokens
"""

from huggingface_hub import login, HfApi
import os
import sys
import argparse

def print_header():
    print("=" * 80)
    print("CONFIGURACIÓN DE HUGGINGFACE - UNDERDOG")
    print("=" * 80)
    print()

def print_instructions():
    print("Para usar datos reales de HuggingFace necesitas un token de acceso.")
    print("Puedes obtenerlo en: https://huggingface.co/settings/tokens")
    print()
    print("Pasos:")
    print("  1. Ve a https://huggingface.co/settings/tokens")
    print("  2. Crea un token de tipo 'Read' (solo lectura)")
    print("  3. Copia el token")
    print()

def validate_token(token: str) -> bool:
    """Validate token by making a test API call"""
    try:
        api = HfApi(token=token)
        # Test call - get user info
        api.whoami()
        return True
    except Exception as e:
        print(f"✗ Token inválido: {e}")
        return False

def authenticate_token(token: str):
    """Authenticate with HuggingFace"""
    print("✓ Token recibido. Validando...")
    
    if not validate_token(token):
        return False
    
    print("✓ Token válido. Autenticando...")
    
    try:
        login(token=token, add_to_git_credential=False)
        print("✓ Autenticación exitosa!")
        print()
        print("🎉 Configuración completada!")
        print()
        print("Ahora puedes:")
        print("  1. Usar datos reales en el dashboard")
        print("  2. Marcar 'Use HuggingFace Data' en el sidebar")
        print("  3. Ejecutar backtests con datos históricos reales")
        print()
        return True
    except Exception as e:
        print(f"✗ Error durante la autenticación: {e}")
        return False

def main():
    print_header()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Configure HuggingFace token")
    parser.add_argument('--token', type=str, help='HuggingFace token')
    parser.add_argument('--test', action='store_true', help='Test current authentication')
    args = parser.parse_args()
    
    # Test mode - check if already authenticated
    if args.test:
        print("Probando autenticación actual...")
        try:
            api = HfApi()
            user_info = api.whoami()
            print(f"✓ Ya estás autenticado como: {user_info.get('name', 'Unknown')}")
            return
        except Exception as e:
            print(f"✗ No autenticado: {e}")
            print("\nEjecuta este script sin --test para configurar.")
            sys.exit(1)
    
    token = None
    
    # Method 1: Command line argument
    if args.token:
        print("📝 Usando token del argumento...")
        token = args.token.strip()
    
    # Method 2: Environment variable
    elif 'HF_TOKEN' in os.environ:
        print("📝 Usando token de variable de entorno HF_TOKEN...")
        token = os.environ['HF_TOKEN'].strip()
    
    # Method 3: Interactive input
    else:
        print_instructions()
        print("Método de entrada:")
        print("  1. Pega el token directamente aquí")
        print("  2. O usa: $env:HF_TOKEN = 'tu_token'")
        print()
        try:
            print("Pega tu token aquí y presiona Enter:")
            token = input().strip()
        except KeyboardInterrupt:
            print("\n\n✗ Cancelado por el usuario")
            sys.exit(1)
        except EOFError:
            print("\n✗ No se pudo leer el input")
            print("\nPrueba uno de estos métodos alternativos:")
            print("  1. Variable de entorno:")
            print("     $env:HF_TOKEN = 'tu_token_aqui'")
            print("     poetry run python scripts/setup_hf_token.py")
            print()
            print("  2. Argumento de línea de comandos:")
            print("     poetry run python scripts/setup_hf_token.py --token tu_token_aqui")
            sys.exit(1)
    
    if not token:
        print("✗ Token vacío. Por favor, intenta de nuevo.")
        sys.exit(1)
    
    # Authenticate
    success = authenticate_token(token)
    
    if not success:
        print()
        print("❌ Autenticación fallida.")
        print()
        print("Métodos alternativos:")
        print("  1. Verifica que el token sea correcto")
        print("  2. Regenera el token en HuggingFace settings")
        print("  3. Intenta con una variable de entorno:")
        print("     $env:HF_TOKEN = 'tu_token'")
        print("     poetry run python scripts/setup_hf_token.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
