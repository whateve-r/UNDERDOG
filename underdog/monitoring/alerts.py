"""
Alert Management System
Sends notifications for critical events via Email, Slack, Telegram, etc.
"""
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SLACK = "slack"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert message"""
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


class AlertManager:
    """
    Alert Manager for UNDERDOG Trading System
    
    Sends alerts for:
    - Drawdown breaches (daily/weekly/monthly)
    - Kill switch activation
    - Connection loss (MT5, ZeroMQ, Database)
    - Model staleness
    - Execution errors
    - Anomalous rejections
    """
    
    def __init__(self,
                 email_config: Optional[Dict] = None,
                 slack_webhook: Optional[str] = None,
                 telegram_config: Optional[Dict] = None,
                 custom_webhooks: Optional[List[str]] = None,
                 cooldown_minutes: float = 5.0):
        """
        Initialize alert manager
        
        Args:
            email_config: Email configuration (smtp_server, port, username, password, to_address)
            slack_webhook: Slack webhook URL
            telegram_config: Telegram configuration (bot_token, chat_id)
            custom_webhooks: List of custom webhook URLs
            cooldown_minutes: Minimum time between duplicate alerts
        """
        self.email_config = email_config or {}
        self.slack_webhook = slack_webhook
        self.telegram_config = telegram_config or {}
        self.custom_webhooks = custom_webhooks or []
        self.cooldown_minutes = cooldown_minutes
        
        # Alert history for cooldown
        self.alert_history: Dict[str, datetime] = {}
        
        # Alert counters
        self.alerts_sent: Dict[AlertSeverity, int] = {
            severity: 0 for severity in AlertSeverity
        }
    
    def send_alert(self,
                   severity: AlertSeverity,
                   title: str,
                   message: str,
                   channels: Optional[List[AlertChannel]] = None,
                   metadata: Optional[Dict] = None,
                   bypass_cooldown: bool = False) -> bool:
        """
        Send alert through specified channels
        
        Args:
            severity: Alert severity
            title: Alert title
            message: Alert message
            channels: List of channels (defaults to all configured)
            metadata: Additional metadata
            bypass_cooldown: Bypass cooldown check
        
        Returns:
            True if alert sent successfully
        """
        # Create alert
        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        # Check cooldown
        if not bypass_cooldown:
            alert_key = f"{title}_{severity.value}"
            if self._is_in_cooldown(alert_key):
                logger.debug(f"Alert '{title}' in cooldown, skipping")
                return False
            self.alert_history[alert_key] = datetime.utcnow()
        
        # Default to all configured channels
        if channels is None:
            channels = self._get_configured_channels()
        
        # Send through each channel
        success = False
        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL:
                    self._send_email(alert)
                    success = True
                elif channel == AlertChannel.SLACK:
                    self._send_slack(alert)
                    success = True
                elif channel == AlertChannel.TELEGRAM:
                    self._send_telegram(alert)
                    success = True
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhooks(alert)
                    success = True
                elif channel == AlertChannel.LOG:
                    self._log_alert(alert)
                    success = True
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
        
        # Update counter
        if success:
            self.alerts_sent[severity] += 1
        
        return success
    
    def _is_in_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period"""
        if alert_key not in self.alert_history:
            return False
        
        last_sent = self.alert_history[alert_key]
        elapsed = (datetime.utcnow() - last_sent).total_seconds() / 60.0
        
        return elapsed < self.cooldown_minutes
    
    def _get_configured_channels(self) -> List[AlertChannel]:
        """Get list of configured channels"""
        channels = [AlertChannel.LOG]  # Always log
        
        if self.email_config:
            channels.append(AlertChannel.EMAIL)
        if self.slack_webhook:
            channels.append(AlertChannel.SLACK)
        if self.telegram_config:
            channels.append(AlertChannel.TELEGRAM)
        if self.custom_webhooks:
            channels.append(AlertChannel.WEBHOOK)
        
        return channels
    
    def _send_email(self, alert: Alert):
        """Send alert via email"""
        if not self.email_config:
            return
        
        smtp_server = self.email_config.get('smtp_server')
        port = self.email_config.get('port', 587)
        username = self.email_config.get('username')
        password = self.email_config.get('password')
        to_address = self.email_config.get('to_address')
        
        if not all([smtp_server, username, password, to_address]):
            logger.warning("Incomplete email configuration")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = to_address
        msg['Subject'] = f"[UNDERDOG {alert.severity.value.upper()}] {alert.title}"
        
        # Email body
        body = f"""
UNDERDOG Trading System Alert

Severity: {alert.severity.value.upper()}
Title: {alert.title}
Timestamp: {alert.timestamp.isoformat()}

Message:
{alert.message}

Metadata:
{alert.metadata if alert.metadata else 'None'}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            with smtplib.SMTP(smtp_server, port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            logger.info(f"Email alert sent: {alert.title}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise
    
    def _send_slack(self, alert: Alert):
        """Send alert via Slack webhook"""
        if not self.slack_webhook:
            return
        
        # Slack color mapping
        color_map = {
            AlertSeverity.INFO: '#36a64f',
            AlertSeverity.WARNING: '#ff9900',
            AlertSeverity.ERROR: '#ff0000',
            AlertSeverity.CRITICAL: '#8b0000'
        }
        
        # Create Slack message
        payload = {
            'text': f"*UNDERDOG Trading Alert*",
            'attachments': [
                {
                    'color': color_map.get(alert.severity, '#808080'),
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {
                            'title': 'Severity',
                            'value': alert.severity.value.upper(),
                            'short': True
                        },
                        {
                            'title': 'Timestamp',
                            'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                            'short': True
                        }
                    ],
                    'footer': 'UNDERDOG Monitoring',
                    'ts': int(alert.timestamp.timestamp())
                }
            ]
        }
        
        # Add metadata fields
        if alert.metadata:
            for key, value in alert.metadata.items():
                payload['attachments'][0]['fields'].append({
                    'title': key.replace('_', ' ').title(),
                    'value': str(value),
                    'short': True
                })
        
        # Send to Slack
        try:
            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Slack alert sent: {alert.title}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            raise
    
    def _send_telegram(self, alert: Alert):
        """Send alert via Telegram bot"""
        if not self.telegram_config:
            return
        
        bot_token = self.telegram_config.get('bot_token')
        chat_id = self.telegram_config.get('chat_id')
        
        if not all([bot_token, chat_id]):
            logger.warning("Incomplete Telegram configuration")
            return
        
        # Emoji mapping
        emoji_map = {
            AlertSeverity.INFO: 'ℹ️',
            AlertSeverity.WARNING: '⚠️',
            AlertSeverity.ERROR: '🚨',
            AlertSeverity.CRITICAL: '🔴'
        }
        
        # Create Telegram message
        emoji = emoji_map.get(alert.severity, '📢')
        text = f"{emoji} *UNDERDOG Alert*\n\n"
        text += f"*{alert.title}*\n"
        text += f"Severity: `{alert.severity.value.upper()}`\n"
        text += f"Time: `{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}`\n\n"
        text += f"{alert.message}\n"
        
        if alert.metadata:
            text += "\n*Metadata:*\n"
            for key, value in alert.metadata.items():
                text += f"• {key}: `{value}`\n"
        
        # Send to Telegram
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': text,
            'parse_mode': 'Markdown'
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Telegram alert sent: {alert.title}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            raise
    
    def _send_webhooks(self, alert: Alert):
        """Send alert to custom webhooks"""
        if not self.custom_webhooks:
            return
        
        payload = alert.to_dict()
        
        for webhook_url in self.custom_webhooks:
            try:
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                logger.info(f"Webhook alert sent to {webhook_url}")
            except Exception as e:
                logger.error(f"Failed to send webhook to {webhook_url}: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert"""
        log_level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }
        
        level = log_level_map.get(alert.severity, logging.INFO)
        logger.log(level, f"[ALERT] {alert.title}: {alert.message}")
    
    # === Convenience Methods for Common Alerts ===
    
    def alert_drawdown_breach(self, dd_pct: float, limit_pct: float, timeframe: str):
        """Alert for drawdown limit breach"""
        self.send_alert(
            severity=AlertSeverity.CRITICAL,
            title=f"Drawdown Breach - {timeframe.upper()}",
            message=f"Drawdown {dd_pct:.2f}% exceeded limit {limit_pct:.2f}%",
            metadata={
                'drawdown_pct': dd_pct,
                'limit_pct': limit_pct,
                'timeframe': timeframe
            }
        )
    
    def alert_kill_switch_activated(self, reason: str):
        """Alert for kill switch activation"""
        self.send_alert(
            severity=AlertSeverity.CRITICAL,
            title="Kill Switch Activated",
            message=f"Trading halted: {reason}",
            metadata={'reason': reason},
            bypass_cooldown=True  # Always send kill switch alerts
        )
    
    def alert_connection_loss(self, component: str):
        """Alert for connection loss"""
        self.send_alert(
            severity=AlertSeverity.ERROR,
            title=f"Connection Lost - {component}",
            message=f"Lost connection to {component}. Trading may be affected.",
            metadata={'component': component}
        )
    
    def alert_model_stale(self, hours_old: float):
        """Alert for stale ML model"""
        self.send_alert(
            severity=AlertSeverity.WARNING,
            title="ML Model Stale",
            message=f"Model is {hours_old:.1f} hours old. Consider retraining.",
            metadata={'hours_old': hours_old}
        )
    
    def alert_execution_error(self, symbol: str, error_msg: str):
        """Alert for execution error"""
        self.send_alert(
            severity=AlertSeverity.ERROR,
            title=f"Execution Error - {symbol}",
            message=f"Failed to execute order: {error_msg}",
            metadata={'symbol': symbol, 'error': error_msg}
        )
    
    def alert_high_rejection_rate(self, rate_pct: float, threshold_pct: float):
        """Alert for high signal rejection rate"""
        self.send_alert(
            severity=AlertSeverity.WARNING,
            title="High Signal Rejection Rate",
            message=f"Rejection rate {rate_pct:.1f}% exceeds threshold {threshold_pct:.1f}%",
            metadata={'rate_pct': rate_pct, 'threshold_pct': threshold_pct}
        )
    
    def alert_position_stuck(self, symbol: str, hours_open: float):
        """Alert for position stuck open too long"""
        self.send_alert(
            severity=AlertSeverity.WARNING,
            title=f"Position Stuck - {symbol}",
            message=f"Position open for {hours_open:.1f} hours without exit signal",
            metadata={'symbol': symbol, 'hours_open': hours_open}
        )
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        return {
            'total_alerts': sum(self.alerts_sent.values()),
            'by_severity': {
                severity.value: count 
                for severity, count in self.alerts_sent.items()
            },
            'cooldown_active': len(self.alert_history)
        }
