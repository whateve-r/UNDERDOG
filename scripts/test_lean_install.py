"""
Test Lean Engine installation and basic functionality.
"""

try:
    # Test 1: Check if Lean is installed
    print("Test 1: Checking Lean installation...")
    import lean
    print(f"✓ Lean installed: version {lean.__version__}")
    
    # Test 2: Try importing QuantConnect modules
    print("\nTest 2: Importing QuantConnect modules...")
    try:
        from QuantConnect import *
        from QuantConnect.Algorithm import *
        print("✓ QuantConnect modules available")
    except ImportError as e:
        print(f"✗ QuantConnect modules not available: {e}")
        print("\nNOTE: Lean requires full installation via 'lean init'")
        print("This creates a project structure with Docker support")
    
    # Test 3: Check Lean CLI
    print("\nTest 3: Checking Lean CLI...")
    import subprocess
    result = subprocess.run(['lean', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Lean CLI available: {result.stdout.strip()}")
    else:
        print("✗ Lean CLI not available")
        print("Install with: dotnet tool install -g QuantConnect.Lean.Cli")
        
except Exception as e:
    print(f"Error during testing: {e}")

print("\n" + "="*80)
print("LEAN ENGINE REQUIREMENTS")
print("="*80)
print("""
Lean Engine requires:
1. .NET SDK 6.0+ (for CLI)
2. Docker (for local backtesting with data)
3. Project initialization: 'lean init'

For simple backtesting without full Lean setup:
→ Backtrader is more straightforward
→ Lean is better for production deployment

RECOMMENDATION: Use Backtrader for development/testing
""")
