"""
News Scraping & Sentiment Analysis Module (100% GRATUITO)

Fuentes de noticias financieras gratuitas:
1. RSS Feeds (Investing.com, Yahoo Finance, Reuters, Bloomberg)
2. Twitter/X (API v2 tiene tier gratuito limitado)
3. Reddit (r/forex, r/wallstreetbets, r/stocks)
4. Google News RSS

Proceso:
1. Scraping de noticias desde fuentes públicas
2. Extracción de texto y metadatos (título, fecha, fuente)
3. Sentiment analysis con modelos pre-entrenados (GRATIS)
4. Almacenamiento en TimescaleDB

Modelos de sentiment GRATIS:
- FinBERT (huggingface: ProsusAI/finbert)
- VADER (nltk.sentiment.vader)
- TextBlob
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import feedparser
import time
import warnings
from bs4 import BeautifulSoup
import json


@dataclass
class NewsSource:
    """Configuración de fuente de noticias"""
    name: str
    url: str
    source_type: str  # "rss", "api", "scraping"
    update_frequency: int = 15  # minutos
    enabled: bool = True


@dataclass
class NewsScrapingConfig:
    """Configuración para scraping de noticias"""
    # Fuentes de noticias
    sources: List[NewsSource] = field(default_factory=lambda: [
        # RSS Feeds (100% GRATIS)
        NewsSource(
            name="Investing.com Forex",
            url="https://www.investing.com/rss/news_1.rss",
            source_type="rss"
        ),
        NewsSource(
            name="Yahoo Finance",
            url="https://finance.yahoo.com/news/rssindex",
            source_type="rss"
        ),
        NewsSource(
            name="Reuters Business",
            url="https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
            source_type="rss"
        ),
        NewsSource(
            name="Forex Live",
            url="https://www.forexlive.com/feed/news",
            source_type="rss"
        ),
        NewsSource(
            name="FXStreet",
            url="https://www.fxstreet.com/rss/",
            source_type="rss"
        )
    ])
    
    # Símbolos relevantes (keywords para filtrar)
    symbols: List[str] = field(default_factory=lambda: [
        "EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD",
        "EURUSD", "GBPUSD", "USDJPY"
    ])
    
    # Sentiment analysis
    sentiment_model: str = "vader"  # "vader", "finbert", "textblob"
    
    # Storage
    cache_dir: str = "data/raw/news"
    output_dir: str = "data/processed/news"
    
    # Rate limiting
    delay_between_requests: float = 1.0
    max_retries: int = 3


class NewsIngestion:
    """
    Cliente para scraping de noticias financieras (100% GRATUITO).
    
    Fuentes soportadas:
    - RSS Feeds (Investing.com, Yahoo Finance, Reuters, etc.)
    - Google News RSS (búsqueda por keyword)
    - Reddit API (praw library)
    """
    
    def __init__(self, config: Optional[NewsScrapingConfig] = None):
        """
        Args:
            config: Configuración de scraping
        """
        self.config = config or NewsScrapingConfig()
        
        # Crear directorios
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Session para requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Inicializar sentiment analyzer
        self.sentiment_analyzer = self._init_sentiment_analyzer()
    
    def _init_sentiment_analyzer(self):
        """Inicializa el analizador de sentimiento (GRATIS)"""
        if self.config.sentiment_model == "vader":
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                import nltk
                
                # Descargar lexicon si no existe
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    print("[NewsIngestion] Descargando VADER lexicon...")
                    nltk.download('vader_lexicon', quiet=True)
                
                return SentimentIntensityAnalyzer()
            
            except ImportError:
                warnings.warn("nltk no instalado. Instalar con: pip install nltk")
                return None
        
        elif self.config.sentiment_model == "finbert":
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                print("[NewsIngestion] Cargando FinBERT (puede tardar en primera ejecución)...")
                
                model_name = "ProsusAI/finbert"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                return {"tokenizer": tokenizer, "model": model}
            
            except ImportError:
                warnings.warn("transformers no instalado. Instalar con: pip install transformers torch")
                return None
        
        elif self.config.sentiment_model == "textblob":
            try:
                from textblob import TextBlob
                return TextBlob
            except ImportError:
                warnings.warn("textblob no instalado. Instalar con: pip install textblob")
                return None
        
        return None
    
    def scrape_rss_feed(self, source: NewsSource) -> List[Dict]:
        """
        Scrape noticias desde RSS feed.
        
        Args:
            source: NewsSource con URL RSS
        
        Returns:
            Lista de diccionarios con noticias
        """
        try:
            print(f"\n[RSS] Scraping: {source.name}")
            
            # Parsear RSS con feedparser (librería estándar)
            feed = feedparser.parse(source.url)
            
            if not feed.entries:
                print(f"  ✗ No se encontraron noticias")
                return []
            
            news_items = []
            
            for entry in feed.entries:
                # Extraer información
                title = entry.get('title', '')
                link = entry.get('link', '')
                published = entry.get('published', '')
                summary = entry.get('summary', '')
                
                # Parsear fecha
                try:
                    timestamp = pd.to_datetime(entry.get('published_parsed'))
                except:
                    timestamp = datetime.now()
                
                # Filtrar por keywords relevantes
                text = f"{title} {summary}".lower()
                relevant = any(symbol.lower() in text for symbol in self.config.symbols)
                
                news_item = {
                    'timestamp': timestamp,
                    'source': source.name,
                    'title': title,
                    'summary': summary,
                    'url': link,
                    'relevant': relevant
                }
                
                # Sentiment analysis
                if relevant and self.sentiment_analyzer:
                    sentiment_scores = self.analyze_sentiment(title + " " + summary)
                    news_item.update(sentiment_scores)
                
                news_items.append(news_item)
            
            print(f"  ✓ Scraped {len(news_items)} noticias ({sum(1 for n in news_items if n['relevant'])} relevantes)")
            
            return news_items
        
        except Exception as e:
            print(f"  ✗ Error scraping {source.name}: {e}")
            return []
    
    def scrape_google_news(self, query: str, language: str = "en") -> List[Dict]:
        """
        Scrape Google News RSS por keyword.
        
        Google News RSS (GRATIS): https://news.google.com/rss/search?q=QUERY
        
        Args:
            query: Búsqueda (ej: "EURUSD forex")
            language: Idioma (ej: "en", "es")
        
        Returns:
            Lista de noticias
        """
        try:
            # Construir URL de Google News RSS
            base_url = "https://news.google.com/rss/search"
            params = {
                'q': query,
                'hl': language,
                'gl': 'US',
                'ceid': 'US:en'
            }
            
            # Construir URL completa
            from urllib.parse import urlencode
            url = f"{base_url}?{urlencode(params)}"
            
            print(f"\n[Google News] Búsqueda: {query}")
            
            # Parsear RSS
            feed = feedparser.parse(url)
            
            news_items = []
            
            for entry in feed.entries:
                try:
                    timestamp = pd.to_datetime(entry.get('published_parsed'))
                except:
                    timestamp = datetime.now()
                
                news_item = {
                    'timestamp': timestamp,
                    'source': 'Google News',
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'url': entry.get('link', ''),
                    'relevant': True  # Ya está filtrado por query
                }
                
                # Sentiment
                if self.sentiment_analyzer:
                    sentiment_scores = self.analyze_sentiment(news_item['title'])
                    news_item.update(sentiment_scores)
                
                news_items.append(news_item)
            
            print(f"  ✓ Encontradas {len(news_items)} noticias")
            
            return news_items
        
        except Exception as e:
            print(f"  ✗ Error en Google News: {e}")
            return []
    
    def scrape_reddit(self, subreddit: str = "forex", limit: int = 100) -> List[Dict]:
        """
        Scrape Reddit posts (GRATIS con praw).
        
        NOTA: Requiere crear app en https://www.reddit.com/prefs/apps/
              (100% gratis, sin tarjeta de crédito)
        
        Args:
            subreddit: Subreddit a scrapear (ej: "forex", "wallstreetbets")
            limit: Número de posts
        
        Returns:
            Lista de posts
        """
        try:
            import praw
            
            # Configurar Reddit API (reemplazar con tus credenciales)
            # Crear app GRATIS en: https://www.reddit.com/prefs/apps/
            reddit = praw.Reddit(
                client_id="YOUR_CLIENT_ID",  # Reemplazar
                client_secret="YOUR_CLIENT_SECRET",  # Reemplazar
                user_agent="UNDERDOG Trading Bot v1.0"
            )
            
            print(f"\n[Reddit] Scraping r/{subreddit}")
            
            posts = []
            
            for submission in reddit.subreddit(subreddit).hot(limit=limit):
                # Filtrar por keywords
                text = f"{submission.title} {submission.selftext}".lower()
                relevant = any(symbol.lower() in text for symbol in self.config.symbols)
                
                if not relevant:
                    continue
                
                post = {
                    'timestamp': datetime.fromtimestamp(submission.created_utc),
                    'source': f'Reddit r/{subreddit}',
                    'title': submission.title,
                    'summary': submission.selftext[:500],  # Primeros 500 chars
                    'url': f"https://reddit.com{submission.permalink}",
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'relevant': True
                }
                
                # Sentiment
                if self.sentiment_analyzer:
                    sentiment_scores = self.analyze_sentiment(post['title'] + " " + post['summary'])
                    post.update(sentiment_scores)
                
                posts.append(post)
            
            print(f"  ✓ Scraped {len(posts)} posts relevantes")
            
            return posts
        
        except ImportError:
            warnings.warn("praw no instalado. Instalar con: pip install praw")
            return []
        except Exception as e:
            print(f"  ✗ Error en Reddit: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analiza sentimiento de texto.
        
        Args:
            text: Texto a analizar
        
        Returns:
            Dict con scores: {
                'sentiment_score': float (-1 a 1),
                'sentiment_label': str ('positive', 'neutral', 'negative'),
                'sentiment_confidence': float (0 a 1)
            }
        """
        if not self.sentiment_analyzer:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'sentiment_confidence': 0.0
            }
        
        try:
            if self.config.sentiment_model == "vader":
                # VADER (simple y rápido)
                scores = self.sentiment_analyzer.polarity_scores(text)
                
                compound = scores['compound']  # -1 a 1
                
                # Clasificar
                if compound >= 0.05:
                    label = 'positive'
                elif compound <= -0.05:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                return {
                    'sentiment_score': compound,
                    'sentiment_label': label,
                    'sentiment_confidence': abs(compound),
                    'sentiment_pos': scores['pos'],
                    'sentiment_neg': scores['neg'],
                    'sentiment_neu': scores['neu']
                }
            
            elif self.config.sentiment_model == "finbert":
                # FinBERT (más preciso para finanzas, pero más lento)
                import torch
                
                tokenizer = self.sentiment_analyzer['tokenizer']
                model = self.sentiment_analyzer['model']
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs.detach().numpy()[0]
                
                # FinBERT labels: [positive, negative, neutral]
                labels = ['positive', 'negative', 'neutral']
                label_idx = probs.argmax()
                
                # Score: positive - negative
                score = probs[0] - probs[1]
                
                return {
                    'sentiment_score': float(score),
                    'sentiment_label': labels[label_idx],
                    'sentiment_confidence': float(probs[label_idx]),
                    'sentiment_pos': float(probs[0]),
                    'sentiment_neg': float(probs[1]),
                    'sentiment_neu': float(probs[2])
                }
            
            elif self.config.sentiment_model == "textblob":
                # TextBlob (simple)
                from textblob import TextBlob
                
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 a 1
                
                if polarity > 0:
                    label = 'positive'
                elif polarity < 0:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                return {
                    'sentiment_score': polarity,
                    'sentiment_label': label,
                    'sentiment_confidence': abs(polarity)
                }
        
        except Exception as e:
            warnings.warn(f"Error en sentiment analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'sentiment_confidence': 0.0
            }
    
    def scrape_all_sources(self) -> pd.DataFrame:
        """
        Scrape todas las fuentes configuradas.
        
        Returns:
            DataFrame consolidado con todas las noticias
        """
        all_news = []
        
        print(f"\n{'='*70}")
        print(f"Scraping {len(self.config.sources)} fuentes de noticias")
        print(f"{'='*70}")
        
        for source in self.config.sources:
            if not source.enabled:
                continue
            
            if source.source_type == "rss":
                news = self.scrape_rss_feed(source)
                all_news.extend(news)
            
            # Rate limiting
            time.sleep(self.config.delay_between_requests)
        
        # Convertir a DataFrame
        if len(all_news) == 0:
            print("\n[WARNING] No se scraped ninguna noticia")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_news)
        
        # Filtrar solo relevantes
        df_relevant = df[df['relevant'] == True].copy()
        
        # Sort por timestamp
        df_relevant = df_relevant.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        print(f"\n{'='*70}")
        print(f"Total noticias: {len(df)}")
        print(f"Relevantes: {len(df_relevant)}")
        print(f"Sentiment promedio: {df_relevant['sentiment_score'].mean():.3f}")
        print(f"{'='*70}")
        
        return df_relevant
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str = "news"):
        """Guarda noticias en Parquet"""
        output_path = Path(self.config.output_dir) / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        df.to_parquet(output_path, compression='snappy', index=False)
        
        print(f"\n[NewsIngestion] Guardado en: {output_path}")


# ========================================
# Ejemplo de Uso
# ========================================

def run_news_scraping_example():
    """Ejemplo completo de news scraping"""
    print("="*70)
    print("News Scraping & Sentiment Analysis (100% GRATUITO)")
    print("="*70)
    
    # Configurar con VADER (el más simple)
    config = NewsScrapingConfig(
        sentiment_model="vader",  # Cambiar a "finbert" para mayor precisión
        symbols=["EURUSD", "GBPUSD", "USD", "EUR", "GBP"]
    )
    
    scraper = NewsIngestion(config)
    
    # Scrape todas las fuentes RSS
    news_df = scraper.scrape_all_sources()
    
    if len(news_df) > 0:
        # Mostrar resumen
        print("\nTop 5 Noticias Positivas:")
        print(news_df.nlargest(5, 'sentiment_score')[['timestamp', 'title', 'sentiment_score']])
        
        print("\nTop 5 Noticias Negativas:")
        print(news_df.nsmallest(5, 'sentiment_score')[['timestamp', 'title', 'sentiment_score']])
        
        # Guardar
        scraper.save_to_parquet(news_df)
    
    # Ejemplo: Google News para símbolo específico
    print("\n" + "="*70)
    print("Google News Search: EURUSD")
    print("="*70)
    
    eurusd_news = scraper.scrape_google_news("EURUSD forex")
    
    if eurusd_news:
        print(f"\nEncontradas {len(eurusd_news)} noticias de EURUSD")
        print(pd.DataFrame(eurusd_news)[['timestamp', 'title', 'sentiment_label']])


if __name__ == '__main__':
    run_news_scraping_example()
