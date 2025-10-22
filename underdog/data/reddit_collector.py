"""
Reddit Sentiment Collector

Collects retail sentiment from r/forex, r/wallstreetbets for alternative data.
Based on arXiv:2510.10526v1 (LLM + RL Integration).

Features:
- Async polling every 15 minutes
- Symbol mention tracking
- Post score weighting
- 24h rolling window
- TimescaleDB storage

Setup:
1. Create Reddit app: https://www.reddit.com/prefs/apps
2. Get client_id, client_secret
3. Set environment variables:
   export REDDIT_CLIENT_ID="your_client_id"
   export REDDIT_CLIENT_SECRET="your_secret"
   export REDDIT_USER_AGENT="underdog-bot-v1.0"
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass

import praw
from praw.models import Submission

logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Reddit post data structure"""
    timestamp: datetime
    subreddit: str
    title: str
    text: str
    score: int
    num_comments: int
    url: str
    symbol_mentions: List[str]


class RedditSentimentCollector:
    """
    Collect sentiment from financial subreddits
    
    Usage:
        collector = RedditSentimentCollector()
        posts = await collector.collect_symbol_posts('EURUSD', hours=24)
        
        # Start polling loop
        await collector.start_polling(['EURUSD', 'GBPUSD'])
    """
    
    # Symbol variations to search
    SYMBOL_VARIATIONS = {
        'EURUSD': ['EURUSD', 'EUR/USD', 'EUR USD', '$EUR'],
        'GBPUSD': ['GBPUSD', 'GBP/USD', 'GBP USD', '$GBP'],
        'USDJPY': ['USDJPY', 'USD/JPY', 'USD JPY', '$JPY'],
        'XAUUSD': ['XAUUSD', 'GOLD', 'XAU', '$GOLD'],
    }
    
    # Subreddits to monitor
    SUBREDDITS = [
        'forex',
        'Forex',
        'wallstreetbets',
        'stocks',
        'investing',
        'algotrading'
    ]
    
    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        user_agent: str = None
    ):
        """
        Initialize Reddit API client
        
        Args:
            client_id: Reddit app client ID (or env REDDIT_CLIENT_ID)
            client_secret: Reddit app secret (or env REDDIT_CLIENT_SECRET)
            user_agent: User agent string (or env REDDIT_USER_AGENT)
        """
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent or os.getenv('REDDIT_USER_AGENT', 'underdog-bot')
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Reddit credentials required. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET "
                "or pass as arguments. Get credentials: https://www.reddit.com/prefs/apps"
            )
        
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        
        logger.info(f"Reddit API initialized - User: {self.reddit.user.me() if self.reddit.read_only else 'Read-only'}")
    
    def collect_symbol_posts(
        self,
        symbol: str,
        hours: int = 24,
        limit: int = 100
    ) -> List[RedditPost]:
        """
        Collect posts mentioning symbol from last N hours
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            hours: Lookback window in hours
            limit: Max posts per subreddit
        
        Returns:
            List of RedditPost objects
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        posts = []
        
        # Get symbol search terms
        search_terms = self.SYMBOL_VARIATIONS.get(symbol, [symbol])
        query = ' OR '.join(search_terms)
        
        for subreddit_name in self.SUBREDDITS:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search recent posts
                for submission in subreddit.search(query, time_filter='day', limit=limit):
                    post_time = datetime.utcfromtimestamp(submission.created_utc)
                    
                    if post_time < cutoff_time:
                        continue
                    
                    # Check if symbol actually mentioned (avoid false positives)
                    full_text = (submission.title + ' ' + submission.selftext).upper()
                    mentions = [term for term in search_terms if term.upper() in full_text]
                    
                    if not mentions:
                        continue
                    
                    post = RedditPost(
                        timestamp=post_time,
                        subreddit=subreddit_name,
                        title=submission.title,
                        text=submission.selftext[:500],  # Truncate
                        score=submission.score,
                        num_comments=submission.num_comments,
                        url=submission.url,
                        symbol_mentions=mentions
                    )
                    
                    posts.append(post)
                
                logger.debug(f"Found {len([p for p in posts if p.subreddit == subreddit_name])} posts in r/{subreddit_name}")
                
            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit_name}: {e}")
        
        # Sort by timestamp descending
        posts.sort(key=lambda p: p.timestamp, reverse=True)
        
        logger.info(f"Collected {len(posts)} posts for {symbol} (last {hours}h)")
        return posts
    
    def get_hot_posts(self, subreddit_name: str, limit: int = 50) -> List[RedditPost]:
        """Get current hot posts from subreddit"""
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for submission in subreddit.hot(limit=limit):
                post_time = datetime.utcfromtimestamp(submission.created_utc)
                
                post = RedditPost(
                    timestamp=post_time,
                    subreddit=subreddit_name,
                    title=submission.title,
                    text=submission.selftext[:500],
                    score=submission.score,
                    num_comments=submission.num_comments,
                    url=submission.url,
                    symbol_mentions=[]
                )
                
                posts.append(post)
        
        except Exception as e:
            logger.error(f"Error fetching hot posts from r/{subreddit_name}: {e}")
        
        return posts
    
    async def start_polling(
        self,
        symbols: List[str],
        interval_minutes: int = 15,
        callback = None
    ):
        """
        Start async polling loop
        
        Args:
            symbols: List of symbols to monitor
            interval_minutes: Poll frequency
            callback: Optional callback(symbol, posts) when new data collected
        """
        logger.info(f"Starting Reddit polling for {symbols} every {interval_minutes} min")
        
        while True:
            try:
                for symbol in symbols:
                    posts = self.collect_symbol_posts(symbol, hours=24)
                    
                    if callback:
                        await callback(symbol, posts)
                    
                    # Brief delay between symbols
                    await asyncio.sleep(2)
                
                logger.info(f"Reddit poll complete. Next poll in {interval_minutes} min.")
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(60)  # Retry after 1 min on error


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Set environment variables first:
    # export REDDIT_CLIENT_ID="your_id"
    # export REDDIT_CLIENT_SECRET="your_secret"
    
    collector = RedditSentimentCollector()
    
    # Collect last 24h posts for EURUSD
    posts = collector.collect_symbol_posts('EURUSD', hours=24)
    
    print(f"\n{'='*60}")
    print(f"Reddit Posts for EURUSD (last 24h)")
    print(f"{'='*60}\n")
    
    for post in posts[:10]:  # Show top 10
        print(f"[r/{post.subreddit}] {post.title}")
        print(f"  Score: {post.score} | Comments: {post.num_comments}")
        print(f"  Time: {post.timestamp}")
        print(f"  Mentions: {post.symbol_mentions}")
        print()
    
    print(f"\nTotal posts collected: {len(posts)}")
    
    # Calculate basic sentiment (simple heuristic: score > 100 = bullish)
    high_score_posts = [p for p in posts if p.score > 100]
    print(f"High engagement posts (>100 score): {len(high_score_posts)}")
