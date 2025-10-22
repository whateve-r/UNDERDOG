"""
Sentiment Analysis Module

LLM-based sentiment analysis for alternative data integration.

Based on papers:
- arXiv:2510.10526v1: LLM + RL Integration (FinGPT sentiment)

Components:
- llm_connector.py: FinGPT/FinBERT API wrapper
- sentiment_processor.py: Feature engineering (score aggregation)
- news_scraper.py: Web scraping for news sources

Data Sources:
- Reddit (via reddit_collector.py in data/)
- RSS feeds (Investing.com, FXStreet)
- Twitter/X (optional)

Output: Sentiment score [-1, 1] for DRL state vector
"""

__all__ = [
    'FinGPTConnector',
    'SentimentProcessor',
    'NewsScraper',
]
