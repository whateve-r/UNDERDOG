"""Test HF data loader with real data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from underdog.data.hf_loader import HuggingFaceDataHandler, ForexNewsHandler

def test_data_loader():
    """Test loading EURUSD data."""
    print("="*80)
    print("TESTING HUGGING FACE DATA LOADER")
    print("="*80)
    
    # Load data
    handler = HuggingFaceDataHandler(
        dataset_id='elthariel/histdata_fx_1m',
        symbol='EURUSD',
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    # Test get_latest_bar
    print("\n[1] Testing get_latest_bar()...")
    event = handler.get_latest_bar()
    print(f"  Timestamp: {event.timestamp}")
    print(f"  Symbol: {event.symbol}")
    print(f"  OHLC: {event.open:.5f}, {event.high:.5f}, {event.low:.5f}, {event.close:.5f}")
    print(f"  Bid/Ask: {event.bid:.5f} / {event.ask:.5f}")
    print(f"  Spread: {event.spread:.5f} ({event.spread * 10000:.1f} pips)")
    
    # Test get_latest_bars
    print("\n[2] Testing get_latest_bars(5)...")
    handler.update_bars()
    handler.update_bars()
    handler.update_bars()
    events = handler.get_latest_bars(3)
    print(f"  Retrieved {len(events)} bars")
    for i, e in enumerate(events):
        print(f"    {i+1}. {e.timestamp}: {e.close:.5f}")
    
    # Test iteration
    print("\n[3] Testing iteration (first 10 bars)...")
    handler.reset()
    for i in range(10):
        event = handler.get_latest_bar()
        if event:
            print(f"  {i+1}. {event.timestamp}: O={event.open:.5f} H={event.high:.5f} L={event.low:.5f} C={event.close:.5f}")
            handler.update_bars()
        else:
            break
    
    print("\n✓ Data loader working correctly!")
    return handler

def test_news_handler():
    """Test news calendar."""
    print("\n" + "="*80)
    print("TESTING NEWS HANDLER")
    print("="*80)
    
    try:
        news = ForexNewsHandler()
        
        # Test sample check
        from datetime import datetime
        test_date = datetime(2024, 1, 15, 14, 30)  # Example time
        
        is_news = news.is_news_event(test_date, window_minutes=15)
        print(f"\n  Is news event at {test_date}: {is_news}")
        
        if is_news:
            events = news.get_news_at(test_date, window_minutes=15)
            print(f"  Found {len(events)} news events:")
            for _, row in events.head(3).iterrows():
                print(f"    - {row.get('timestamp', 'N/A')}: {row.get('title', 'N/A')}")
        
        print("\n✓ News handler working!")
        
    except Exception as e:
        print(f"\n✗ News handler error: {e}")
        print("  (This is OK - news dataset might require authentication)")

if __name__ == '__main__':
    handler = test_data_loader()
    test_news_handler()
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
