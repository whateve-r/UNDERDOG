"""
Test Suite for ZeroMQ Message Schemas
Tests message validation, serialization, and dataclass conversions.
"""
import pytest
from datetime import datetime
import json

from underdog.core.schemas.zmq_messages import (
    Tick, OHLCV, OrderRequest, ExecutionACK,
    AccountInfo, PositionUpdate, MessageFactory,
    validate_message
)


class TestTickMessage:
    """Test Tick message dataclass and validation"""
    
    def test_tick_creation(self):
        """Test creating a valid Tick message"""
        tick = Tick(
            symbol='EURUSD',
            bid=1.09850,
            ask=1.09853,
            time=datetime.now().isoformat(),
            volume=100
        )
        
        assert tick.symbol == 'EURUSD'
        assert tick.bid == 1.09850
        assert tick.ask == 1.09853
        assert tick.volume == 100
    
    def test_tick_to_dict(self):
        """Test Tick serialization to dict"""
        tick = Tick(
            symbol='EURUSD',
            bid=1.09850,
            ask=1.09853,
            time=datetime.now().isoformat(),
            volume=100
        )
        
        tick_dict = tick.to_dict()
        
        assert isinstance(tick_dict, dict)
        assert tick_dict['symbol'] == 'EURUSD'
        assert tick_dict['bid'] == 1.09850
    
    def test_tick_spread(self):
        """Test spread calculation"""
        tick = Tick(
            symbol='EURUSD',
            bid=1.09850,
            ask=1.09853,
            time=datetime.now().isoformat(),
            volume=100
        )
        
        spread = tick.ask - tick.bid
        assert spread == pytest.approx(0.00003, abs=1e-8)


class TestOHLCVMessage:
    """Test OHLCV message dataclass"""
    
    def test_ohlcv_creation(self):
        """Test creating a valid OHLCV message"""
        ohlcv = OHLCV(
            symbol='EURUSD',
            timeframe='H1',
            time=datetime.now().isoformat(),
            open=1.09850,
            high=1.09900,
            low=1.09800,
            close=1.09875,
            volume=5000
        )
        
        assert ohlcv.symbol == 'EURUSD'
        assert ohlcv.high >= ohlcv.low
        assert ohlcv.close <= ohlcv.high
        assert ohlcv.close >= ohlcv.low
    
    def test_ohlcv_validation_high_low(self):
        """Test that high >= low validation"""
        # This should be valid
        ohlcv = OHLCV(
            symbol='EURUSD',
            timeframe='H1',
            time=datetime.now().isoformat(),
            open=1.09850,
            high=1.09900,
            low=1.09800,
            close=1.09875,
            volume=5000
        )
        
        assert ohlcv.high >= ohlcv.low
        assert ohlcv.open <= ohlcv.high
        assert ohlcv.close <= ohlcv.high


class TestOrderRequest:
    """Test OrderRequest message"""
    
    def test_order_request_creation(self):
        """Test creating a valid OrderRequest"""
        order = OrderRequest(
            symbol='EURUSD',
            order_type='market',
            side='buy',
            volume=1.0,
            price=None,
            stop_loss=1.09500,
            take_profit=1.10500
        )
        
        assert order.symbol == 'EURUSD'
        assert order.side == 'buy'
        assert order.volume == 1.0
        assert hasattr(order, 'order_id')  # UUID generated
    
    def test_order_request_uuid_unique(self):
        """Test that each order gets unique UUID"""
        order1 = OrderRequest(
            symbol='EURUSD',
            order_type='market',
            side='buy',
            volume=1.0
        )
        
        order2 = OrderRequest(
            symbol='EURUSD',
            order_type='market',
            side='buy',
            volume=1.0
        )
        
        assert order1.order_id != order2.order_id


class TestExecutionACK:
    """Test ExecutionACK message"""
    
    def test_execution_ack_success(self):
        """Test successful execution ACK"""
        ack = ExecutionACK(
            order_id='test-order-123',
            status='filled',
            filled_price=1.09875,
            filled_volume=1.0,
            timestamp=datetime.now().isoformat()
        )
        
        assert ack.status == 'filled'
        assert ack.filled_volume == 1.0
        assert ack.message is None
    
    def test_execution_ack_rejected(self):
        """Test rejected execution ACK"""
        ack = ExecutionACK(
            order_id='test-order-123',
            status='rejected',
            filled_price=None,
            filled_volume=0.0,
            timestamp=datetime.now().isoformat(),
            message='Insufficient margin'
        )
        
        assert ack.status == 'rejected'
        assert ack.filled_volume == 0.0
        assert ack.message == 'Insufficient margin'


class TestMessageFactory:
    """Test MessageFactory parsing"""
    
    def test_parse_tick_message(self):
        """Test parsing Tick from JSON"""
        tick_json = {
            'type': 'tick',
            'data': {
                'symbol': 'EURUSD',
                'bid': 1.09850,
                'ask': 1.09853,
                'time': datetime.now().isoformat(),
                'volume': 100
            }
        }
        
        tick = MessageFactory.parse(tick_json)
        
        assert isinstance(tick, Tick)
        assert tick.symbol == 'EURUSD'
        assert tick.bid == 1.09850
    
    def test_parse_ohlcv_message(self):
        """Test parsing OHLCV from JSON"""
        ohlcv_json = {
            'type': 'ohlcv',
            'data': {
                'symbol': 'EURUSD',
                'timeframe': 'H1',
                'time': datetime.now().isoformat(),
                'open': 1.09850,
                'high': 1.09900,
                'low': 1.09800,
                'close': 1.09875,
                'volume': 5000
            }
        }
        
        ohlcv = MessageFactory.parse(ohlcv_json)
        
        assert isinstance(ohlcv, OHLCV)
        assert ohlcv.symbol == 'EURUSD'
        assert ohlcv.timeframe == 'H1'
    
    def test_parse_invalid_type(self):
        """Test parsing with invalid message type"""
        invalid_json = {
            'type': 'unknown_type',
            'data': {}
        }
        
        result = MessageFactory.parse(invalid_json)
        assert result is None


class TestMessageValidation:
    """Test message validation functions"""
    
    def test_validate_tick_message(self):
        """Test validation of Tick message"""
        tick_data = {
            'symbol': 'EURUSD',
            'bid': 1.09850,
            'ask': 1.09853,
            'time': datetime.now().isoformat(),
            'volume': 100
        }
        
        is_valid = validate_message('tick', tick_data)
        assert is_valid is True
    
    def test_validate_missing_field(self):
        """Test validation fails with missing required field"""
        incomplete_tick = {
            'symbol': 'EURUSD',
            'bid': 1.09850,
            # Missing 'ask'
            'time': datetime.now().isoformat()
        }
        
        is_valid = validate_message('tick', incomplete_tick)
        assert is_valid is False
    
    def test_validate_order_request(self):
        """Test validation of OrderRequest"""
        order_data = {
            'symbol': 'EURUSD',
            'order_type': 'market',
            'side': 'buy',
            'volume': 1.0,
            'price': None,
            'stop_loss': 1.09500,
            'take_profit': 1.10500
        }
        
        is_valid = validate_message('order_request', order_data)
        assert is_valid is True


class TestAccountInfo:
    """Test AccountInfo message"""
    
    def test_account_info_creation(self):
        """Test creating AccountInfo message"""
        account = AccountInfo(
            balance=100000.0,
            equity=101500.0,
            margin=5000.0,
            free_margin=96500.0,
            margin_level=2030.0,
            timestamp=datetime.now().isoformat()
        )
        
        assert account.balance == 100000.0
        assert account.equity > account.balance
        assert account.free_margin == account.equity - account.margin


class TestPositionUpdate:
    """Test PositionUpdate message"""
    
    def test_position_update_creation(self):
        """Test creating PositionUpdate message"""
        position = PositionUpdate(
            symbol='EURUSD',
            volume=1.0,
            side='long',
            entry_price=1.09850,
            current_price=1.09900,
            unrealized_pnl=50.0,
            timestamp=datetime.now().isoformat()
        )
        
        assert position.symbol == 'EURUSD'
        assert position.side == 'long'
        assert position.unrealized_pnl > 0  # Winning position


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
