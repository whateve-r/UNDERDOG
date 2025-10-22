"""
FinGPT/FinBERT Sentiment Analysis Connector

LLM-based sentiment analysis for financial text:
- Reddit posts
- News headlines
- Social media

Models:
- ProsusAI/finbert (FinBERT) - Financial sentiment classifier
- FinGPT variants (Hugging Face)

Output: Sentiment score in [-1, 1] range
- -1: Very Negative
-  0: Neutral
- +1: Very Positive

Paper: arXiv:2510.10526v1 (LLM + RL Integration)
"""

from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    score: float  # [-1, 1]
    label: str  # 'positive', 'neutral', 'negative'
    confidence: float  # [0, 1]


class FinGPTConnector:
    """
    LLM Sentiment Analysis using FinBERT/FinGPT
    
    Usage:
        connector = FinGPTConnector(model_name='ProsusAI/finbert')
        score = connector.analyze("EUR/USD looks bullish today!")
        # Returns: 0.75 (positive sentiment)
    """
    
    def __init__(self, model_name: str = 'ProsusAI/finbert', device: Optional[str] = None):
        """
        Initialize FinGPT connector
        
        Args:
            model_name: Hugging Face model name
                - 'ProsusAI/finbert' (default, 135M params)
                - 'yiyanghkust/finbert-tone' (alternative)
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        self.model_name = model_name
        self.device = device
        self.sentiment_pipeline = None
        self._is_initialized = False
        
        logger.info(f"FinGPTConnector initialized with model: {model_name}")
    
    def _lazy_load(self):
        """Lazy load model (only when first needed)"""
        if self._is_initialized:
            return
        
        try:
            from transformers import pipeline
            
            logger.info(f"Loading model: {self.model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device
            )
            self._is_initialized = True
            logger.info("Model loaded successfully")
            
        except ImportError:
            raise ImportError(
                "transformers library not installed. Install with: "
                "pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of single text
        
        Args:
            text: Input text (post, headline, tweet)
        
        Returns:
            Sentiment score in [-1, 1]
        """
        self._lazy_load()
        
        if not text or not text.strip():
            return 0.0
        
        try:
            result = self.sentiment_pipeline(text[:512])[0]  # Max 512 tokens
            
            # Convert to [-1, 1] scale
            label = result['label'].lower()
            confidence = result['score']
            
            if 'positive' in label:
                return confidence
            elif 'negative' in label:
                return -confidence
            else:  # neutral
                return 0.0
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0
    
    def analyze_batch(self, texts: List[str], aggregate: bool = True) -> float | List[float]:
        """
        Analyze sentiment of multiple texts (efficient batch processing)
        
        Args:
            texts: List of input texts
            aggregate: If True, return mean sentiment. If False, return list of scores.
        
        Returns:
            Mean sentiment score (if aggregate=True) or list of scores
        """
        self._lazy_load()
        
        if not texts:
            return 0.0 if aggregate else []
        
        try:
            # Filter empty texts
            valid_texts = [t[:512] for t in texts if t and t.strip()]
            
            if not valid_texts:
                return 0.0 if aggregate else []
            
            # Batch inference
            results = self.sentiment_pipeline(valid_texts)
            
            # Convert to [-1, 1] scale
            scores = []
            for result in results:
                label = result['label'].lower()
                confidence = result['score']
                
                if 'positive' in label:
                    scores.append(confidence)
                elif 'negative' in label:
                    scores.append(-confidence)
                else:
                    scores.append(0.0)
            
            if aggregate:
                return sum(scores) / len(scores)
            else:
                return scores
                
        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {e}")
            return 0.0 if aggregate else []
    
    def analyze_detailed(self, text: str) -> SentimentResult:
        """
        Detailed sentiment analysis with label and confidence
        
        Args:
            text: Input text
        
        Returns:
            SentimentResult with score, label, confidence
        """
        self._lazy_load()
        
        if not text or not text.strip():
            return SentimentResult(text=text, score=0.0, label='neutral', confidence=0.0)
        
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            
            label = result['label'].lower()
            confidence = result['score']
            
            # Convert to [-1, 1] scale
            if 'positive' in label:
                score = confidence
            elif 'negative' in label:
                score = -confidence
            else:
                score = 0.0
            
            return SentimentResult(
                text=text[:100],  # Truncate for display
                score=score,
                label=label,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Detailed sentiment analysis failed: {e}")
            return SentimentResult(text=text[:100], score=0.0, label='error', confidence=0.0)


# Convenience function
def get_default_connector() -> FinGPTConnector:
    """Get default FinBERT connector (lazy-loaded singleton)"""
    if not hasattr(get_default_connector, '_instance'):
        get_default_connector._instance = FinGPTConnector()
    return get_default_connector._instance


if __name__ == "__main__":
    # Quick test
    print("Testing FinGPT Connector...")
    
    connector = FinGPTConnector()
    
    test_texts = [
        "EUR/USD looking very bullish! Strong momentum!",
        "Market crash imminent, sell everything",
        "Neutral consolidation continues",
    ]
    
    print("\nSingle analysis:")
    for text in test_texts:
        score = connector.analyze(text)
        print(f"  '{text[:50]}...' → {score:.3f}")
    
    print("\nBatch analysis:")
    mean_score = connector.analyze_batch(test_texts)
    print(f"  Mean sentiment: {mean_score:.3f}")
    
    print("\nDetailed analysis:")
    result = connector.analyze_detailed(test_texts[0])
    print(f"  Score: {result.score:.3f}, Label: {result.label}, Confidence: {result.confidence:.3f}")
