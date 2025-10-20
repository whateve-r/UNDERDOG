"""
Automatic EA Instrumentation with Prometheus
==============================================

Automatically adds Prometheus instrumentation to all 6 remaining EAs:
- ParabolicEMA
- KeltnerBreakout
- EmaScalper
- BollingerCCI
- ATRBreakout
- PairArbitrage

USAGE:
    poetry run python scripts/instrument_eas_prometheus.py
"""

import re
from pathlib import Path

# Mapping: filename → EA display name
EAS_TO_INSTRUMENT = {
    "ea_parabolic_ema_v4.py": "ParabolicEMA",
    "ea_keltner_breakout_v4.py": "KeltnerBreakout",
    "ea_ema_scalper_v4.py": "EmaScalper",
    "ea_bollinger_cci_v4.py": "BollingerCCI",
    "ea_atr_breakout_v4.py": "ATRBreakout",
    "ea_pair_arbitrage_v4.py": "PairArbitrage",
}

STRATEGIES_DIR = Path(__file__).parent.parent / "underdog" / "strategies"


def instrument_ea(filepath: Path, ea_name: str):
    """
    Add Prometheus instrumentation to a single EA file.
    
    Steps:
    1. Add imports
    2. Instrument initialize()
    3. Instrument shutdown()
    4. Instrument generate_signal()
    """
    print(f"\n{'='*60}")
    print(f"📊 Instrumenting: {ea_name}")
    print(f"📁 File: {filepath.name}")
    print(f"{'='*60}")
    
    content = filepath.read_text(encoding='utf-8')
    
    # ==================== STEP 1: Add imports ====================
    
    # Check if already instrumented
    if "from underdog.monitoring.prometheus_metrics import" in content:
        print(f"⚠️  {ea_name} already instrumented. Skipping...")
        return False
    
    # Find import section (after existing imports, before class definition)
    import_pattern = r"(import logging\s+)(from underdog\.)"
    
    if not re.search(import_pattern, content):
        print(f"❌ Could not find import section in {ea_name}")
        return False
    
    prometheus_imports = """
from underdog.monitoring.prometheus_metrics import (
    record_signal,
    record_execution_time,
    update_position_count,
    ea_status
)
"""
    
    content = re.sub(
        import_pattern,
        rf"\1{prometheus_imports}\n\2",
        content,
        count=1
    )
    
    print("✅ Added Prometheus imports")
    
    # ==================== STEP 2: Instrument initialize() ====================
    
    # Find initialize() method
    initialize_pattern = r"(async def initialize\(self\) -> None:.*?)(logger\.info\(.*?initialized.*?\))"
    
    if not re.search(initialize_pattern, content, re.DOTALL):
        print(f"❌ Could not find initialize() method in {ea_name}")
        return False
    
    initialize_code = f"""
        # Prometheus: Mark EA as active
        ea_status.labels(ea_name="{ea_name}").set(1)
        logger.info(f"📊 Prometheus: {{self.__class__.__name__}} marked as ACTIVE")
"""
    
    content = re.sub(
        initialize_pattern,
        rf"\1\2{initialize_code}",
        content,
        count=1,
        flags=re.DOTALL
    )
    
    print("✅ Instrumented initialize()")
    
    # ==================== STEP 3: Instrument shutdown() ====================
    
    # Find shutdown() method
    shutdown_pattern = r"(async def shutdown\(self\) -> None:.*?)(logger\.info\(.*?shutdown.*?\))"
    
    if not re.search(shutdown_pattern, content, re.DOTALL):
        print(f"❌ Could not find shutdown() method in {ea_name}")
        return False
    
    shutdown_code = f"""
        # Prometheus: Mark EA as inactive
        ea_status.labels(ea_name="{ea_name}").set(0)
        logger.info(f"📊 Prometheus: {{self.__class__.__name__}} marked as INACTIVE")
"""
    
    content = re.sub(
        shutdown_pattern,
        rf"\1\2{shutdown_code}",
        content,
        count=1,
        flags=re.DOTALL
    )
    
    print("✅ Instrumented shutdown()")
    
    # ==================== STEP 4: Instrument generate_signal() ====================
    
    # Find generate_signal() method start
    signal_start_pattern = r"(async def generate_signal\(self, df: pd\.DataFrame\).*?:)(.*?)(# Validate data)"
    
    if not re.search(signal_start_pattern, content, re.DOTALL):
        print(f"❌ Could not find generate_signal() method in {ea_name}")
        return False
    
    timing_code = """
        # Prometheus: Start timing
        import time
        start_time = time.time()
"""
    
    content = re.sub(
        signal_start_pattern,
        rf"\1\2{timing_code}\n        \3",
        content,
        count=1,
        flags=re.DOTALL
    )
    
    # Find BUY signal return
    buy_pattern = r"(return Signal\(\s+type=SignalType\.BUY,)(.*?)(entry_price=.*?confidence=(\w+).*?\))"
    
    if re.search(buy_pattern, content, re.DOTALL):
        buy_recording = f"""
                # Prometheus: Record BUY signal
                elapsed_ms = (time.time() - start_time) * 1000
                record_signal("{ea_name}", "BUY", self.config.symbol, \\4, True)
                record_execution_time("{ea_name}", elapsed_ms)
                logger.info(f"📊 Prometheus: BUY signal recorded ({{elapsed_ms:.2f}}ms)")
                
                """
        
        content = re.sub(
            buy_pattern,
            rf"{buy_recording}\1\2\3",
            content,
            flags=re.DOTALL
        )
        print("✅ Instrumented BUY signal")
    
    # Find SELL signal return
    sell_pattern = r"(return Signal\(\s+type=SignalType\.SELL,)(.*?)(entry_price=.*?confidence=(\w+).*?\))"
    
    if re.search(sell_pattern, content, re.DOTALL):
        sell_recording = f"""
                # Prometheus: Record SELL signal
                elapsed_ms = (time.time() - start_time) * 1000
                record_signal("{ea_name}", "SELL", self.config.symbol, \\4, True)
                record_execution_time("{ea_name}", elapsed_ms)
                logger.info(f"📊 Prometheus: SELL signal recorded ({{elapsed_ms:.2f}}ms)")
                
                """
        
        content = re.sub(
            sell_pattern,
            rf"{sell_recording}\1\2\3",
            content,
            flags=re.DOTALL
        )
        print("✅ Instrumented SELL signal")
    
    # Find "return None" at end of generate_signal()
    no_signal_pattern = r"(return None\s*$)"
    
    if re.search(no_signal_pattern, content, re.MULTILINE):
        no_signal_recording = f"""
        # Prometheus: Record execution time even if no signal
        elapsed_ms = (time.time() - start_time) * 1000
        record_execution_time("{ea_name}", elapsed_ms)
        
        return None
"""
        content = re.sub(
            no_signal_pattern,
            no_signal_recording,
            content,
            count=1,
            flags=re.MULTILINE
        )
        print("✅ Instrumented no-signal case")
    
    # ==================== WRITE BACK ====================
    
    filepath.write_text(content, encoding='utf-8')
    print(f"💾 Saved {filepath.name}")
    print(f"✅ {ea_name} instrumentation COMPLETE")
    
    return True


def main():
    print("\n" + "="*80)
    print("🔧 AUTOMATIC EA INSTRUMENTATION WITH PROMETHEUS")
    print("="*80)
    
    if not STRATEGIES_DIR.exists():
        print(f"❌ Strategies directory not found: {STRATEGIES_DIR}")
        return
    
    print(f"📁 Strategies directory: {STRATEGIES_DIR}")
    print(f"📊 EAs to instrument: {len(EAS_TO_INSTRUMENT)}")
    
    success_count = 0
    
    for filename, ea_name in EAS_TO_INSTRUMENT.items():
        filepath = STRATEGIES_DIR / filename
        
        if not filepath.exists():
            print(f"\n⚠️  File not found: {filename}")
            continue
        
        if instrument_ea(filepath, ea_name):
            success_count += 1
    
    print("\n" + "="*80)
    print(f"✅ INSTRUMENTATION COMPLETE: {success_count}/{len(EAS_TO_INSTRUMENT)} EAs")
    print("="*80)
    
    if success_count == len(EAS_TO_INSTRUMENT):
        print("\n🎉 All EAs successfully instrumented!")
        print("\nNext steps:")
        print("1. Start Prometheus: docker-compose up -d prometheus")
        print("2. Start Grafana: docker-compose up -d grafana")
        print("3. Run trading system: poetry run python scripts/start_trading_with_monitoring.py")
        print("4. Access metrics: http://localhost:8000/metrics")
        print("5. Access Grafana: http://localhost:3000 (admin/admin)")
    else:
        print(f"\n⚠️  {len(EAS_TO_INSTRUMENT) - success_count} EAs failed instrumentation")
        print("Please check errors above and instrument manually if needed.")


if __name__ == "__main__":
    main()
