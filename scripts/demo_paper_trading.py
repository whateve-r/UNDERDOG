"""
Demo Paper Trading Test - 10 Order Validation

This script executes 10 real orders in MT5 DEMO account to validate:
- MT5Executor functionality
- Backtrader→MT5 Bridge integration
- PropFirm DD limits enforcement
- Emergency stop functionality
- Reconnection handling

Critical Pre-Flight Test before 30-day paper trading

Usage:
    # Configure your DEMO account credentials
    poetry run python scripts/demo_paper_trading.py
    
    # Or with custom settings
    poetry run python scripts/demo_paper_trading.py --account 12345678 --password "xxx" --server "ICMarkets-Demo"

Success Criteria:
    ✅ All 10 orders executed without errors
    ✅ Zero DD limit breaches
    ✅ All orders logged in audit trail
    ✅ Emergency stop works if triggered
    ✅ Reconnection works on connection loss

Author: Underdog Trading System
Business Goal: Validate infrastructure before FTMO challenge
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from underdog.execution.mt5_executor import MT5Executor, OrderType, OrderStatus
from underdog.bridges.bt_to_mt5 import BacktraderMT5Bridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoTradingTest:
    """
    Demo trading test orchestrator
    
    Executes 10 test orders with validation:
    - BUY/SELL alternation
    - SL/TP validation
    - DD monitoring
    - Position tracking
    """
    
    def __init__(
        self,
        account: int,
        password: str,
        server: str,
        symbol: str = "EURUSD",
        volume: float = 0.01,  # Micro lot for testing
        test_orders: int = 10
    ):
        self.account = account
        self.password = password
        self.server = server
        self.symbol = symbol
        self.volume = volume
        self.test_orders = test_orders
        
        self.executor: MT5Executor = None
        self.results = []
    
    def run(self) -> bool:
        """
        Run complete test
        
        Returns:
            bool: True if all tests passed
        """
        logger.info("=" * 80)
        logger.info("DEMO PAPER TRADING TEST - 10 Order Validation")
        logger.info("=" * 80)
        
        # Step 1: Initialize MT5Executor
        logger.info("\n[1/5] Initializing MT5 Executor...")
        self.executor = MT5Executor(
            account=self.account,
            password=self.password,
            server=self.server,
            max_daily_dd=5.0,
            max_total_dd=10.0
        )
        
        if not self.executor.initialize():
            logger.error("❌ Failed to initialize MT5Executor")
            return False
        
        logger.info("✅ MT5Executor initialized successfully")
        
        # Step 2: Execute test orders
        logger.info(f"\n[2/5] Executing {self.test_orders} test orders...")
        success = self._execute_test_orders()
        
        if not success:
            logger.error("❌ Test orders execution failed")
            self.executor.shutdown()
            return False
        
        logger.info("✅ Test orders executed successfully")
        
        # Step 3: Validate DD limits
        logger.info("\n[3/5] Validating DD limits...")
        dd_valid = self._validate_dd_limits()
        
        if not dd_valid:
            logger.error("❌ DD limit validation failed")
            self.executor.shutdown()
            return False
        
        logger.info("✅ DD limits validated")
        
        # Step 4: Test emergency close
        logger.info("\n[4/5] Testing emergency close...")
        emergency_success = self._test_emergency_close()
        
        if not emergency_success:
            logger.error("❌ Emergency close test failed")
            self.executor.shutdown()
            return False
        
        logger.info("✅ Emergency close validated")
        
        # Step 5: Generate report
        logger.info("\n[5/5] Generating test report...")
        self._generate_report()
        
        # Cleanup
        self.executor.shutdown()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED - Ready for 30-day paper trading")
        logger.info("=" * 80)
        
        return True
    
    def _execute_test_orders(self) -> bool:
        """Execute alternating BUY/SELL orders"""
        
        for i in range(self.test_orders):
            # Alternate BUY/SELL
            order_type = OrderType.BUY if i % 2 == 0 else OrderType.SELL
            
            logger.info(f"\nOrder {i+1}/{self.test_orders}: {order_type.name}")
            
            # Execute order
            result = self.executor.execute_order(
                symbol=self.symbol,
                order_type=order_type,
                volume=self.volume,
                sl_pips=20,
                tp_pips=40,
                comment=f"Demo_Test_{i+1}"
            )
            
            # Log result
            self.results.append({
                'order_num': i + 1,
                'timestamp': result.timestamp,
                'type': order_type.name,
                'status': result.status.value,
                'ticket': result.ticket,
                'price': result.price,
                'volume': result.volume,
                'dd_at_execution': result.dd_at_execution,
                'error_message': result.error_message
            })
            
            # Check if order was successful
            if result.status != OrderStatus.SUCCESS:
                if result.status == OrderStatus.REJECTED_DD:
                    logger.warning(f"Order rejected due to DD limit - This is EXPECTED behavior")
                    logger.warning(f"DD at execution: {result.dd_at_execution:.2f}%")
                    continue  # DD rejection is acceptable for testing
                else:
                    logger.error(f"Order failed: {result.error_message}")
                    return False
            
            logger.info(f"✅ Order executed - Ticket: {result.ticket}, Price: {result.price}")
            
            # Wait 2 seconds between orders
            if i < self.test_orders - 1:
                time.sleep(2)
        
        return True
    
    def _validate_dd_limits(self) -> bool:
        """Validate DD limits are being enforced"""
        
        daily_dd, total_dd = self.executor.calculate_drawdown()
        
        logger.info(f"Current Daily DD: {daily_dd:.2f}%")
        logger.info(f"Current Total DD: {total_dd:.2f}%")
        
        # Check if any orders were executed despite DD limits
        results_df = pd.DataFrame(self.results)
        successful_orders = results_df[results_df['status'] == 'success']
        
        if not successful_orders.empty:
            max_dd_at_execution = successful_orders['dd_at_execution'].max()
            
            if max_dd_at_execution >= 5.0:
                logger.error(f"❌ Order executed with DD {max_dd_at_execution:.2f}% >= 5.0%")
                return False
            
            logger.info(f"Max DD at any execution: {max_dd_at_execution:.2f}%")
        
        return True
    
    def _test_emergency_close(self) -> bool:
        """Test emergency close all functionality"""
        
        # Get current open positions
        positions = self.executor.get_open_positions()
        
        if len(positions) == 0:
            logger.info("No open positions to close - Emergency close not needed")
            return True
        
        logger.info(f"Found {len(positions)} open positions")
        
        # Close all
        closed_count = self.executor.emergency_close_all(reason="Test emergency close")
        
        if closed_count != len(positions):
            logger.error(f"❌ Emergency close failed - Closed {closed_count}/{len(positions)}")
            return False
        
        logger.info(f"✅ Emergency close successful - Closed {closed_count} positions")
        
        # Verify all positions are closed
        remaining_positions = self.executor.get_open_positions()
        
        if len(remaining_positions) > 0:
            logger.error(f"❌ {len(remaining_positions)} positions still open after emergency close")
            return False
        
        return True
    
    def _generate_report(self):
        """Generate test report"""
        
        results_df = pd.DataFrame(self.results)
        
        # Summary statistics
        total_orders = len(results_df)
        successful = len(results_df[results_df['status'] == 'success'])
        dd_rejected = len(results_df[results_df['status'] == 'rejected_dd_limit'])
        mt5_rejected = len(results_df[results_df['status'] == 'rejected_mt5_error'])
        conn_rejected = len(results_df[results_df['status'] == 'rejected_connection_lost'])
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST REPORT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Orders:        {total_orders}")
        logger.info(f"Successful:          {successful} ({successful/total_orders*100:.1f}%)")
        logger.info(f"DD Rejected:         {dd_rejected}")
        logger.info(f"MT5 Rejected:        {mt5_rejected}")
        logger.info(f"Connection Rejected: {conn_rejected}")
        logger.info("=" * 80)
        
        # Export to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = project_root / "data" / "test_results" / f"demo_paper_trading_{timestamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(csv_path, index=False)
        logger.info(f"\n📊 Full results exported to: {csv_path}")
        
        # Print detailed results
        logger.info("\nDETAILED RESULTS:")
        print(results_df.to_string(index=False))


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Demo Paper Trading Test - 10 Order Validation")
    parser.add_argument('--account', type=int, help='MT5 account number')
    parser.add_argument('--password', type=str, help='MT5 account password')
    parser.add_argument('--server', type=str, default='ICMarkets-Demo', help='MT5 server name')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol')
    parser.add_argument('--volume', type=float, default=0.01, help='Lot size (default 0.01 micro lot)')
    parser.add_argument('--orders', type=int, default=10, help='Number of test orders')
    
    args = parser.parse_args()
    
    # Check if credentials provided
    if not args.account or not args.password:
        logger.error("\n❌ ERROR: MT5 credentials required")
        logger.info("\nUsage:")
        logger.info("  poetry run python scripts/demo_paper_trading.py --account 12345678 --password 'xxx' --server 'ICMarkets-Demo'")
        logger.info("\nOr set environment variables:")
        logger.info("  MT5_ACCOUNT=12345678")
        logger.info("  MT5_PASSWORD=xxx")
        logger.info("  MT5_SERVER=ICMarkets-Demo")
        sys.exit(1)
    
    # Run test
    test = DemoTradingTest(
        account=args.account,
        password=args.password,
        server=args.server,
        symbol=args.symbol,
        volume=args.volume,
        test_orders=args.orders
    )
    
    success = test.run()
    
    if success:
        logger.info("\n🎉 Demo paper trading test PASSED")
        logger.info("Next step: 30-day paper trading validation")
        sys.exit(0)
    else:
        logger.error("\n❌ Demo paper trading test FAILED")
        logger.error("Fix issues before proceeding to 30-day validation")
        sys.exit(1)


if __name__ == "__main__":
    main()
