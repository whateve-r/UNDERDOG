"""
Explore Hugging Face Hub for Forex datasets.
Find datasets with bid/ask data suitable for backtesting.
"""

from huggingface_hub import HfApi, list_datasets
import datasets

def search_forex_datasets():
    """Search for forex-related datasets on HF Hub."""
    print("Searching Hugging Face Hub for Forex datasets...\n")
    
    api = HfApi()
    
    # Search terms
    search_terms = ['forex', 'fx', 'currency', 'eurusd', 'gbpusd', 'ohlc', 'tick']
    
    for term in search_terms:
        print(f"\n{'='*80}")
        print(f"Search term: '{term}'")
        print('='*80)
        
        try:
            results = list(api.list_datasets(search=term, limit=10))
            
            if results:
                for dataset in results:
                    print(f"\n  Dataset: {dataset.id}")
                    print(f"  Author: {dataset.author}")
                    if hasattr(dataset, 'downloads'):
                        print(f"  Downloads: {dataset.downloads}")
                    if hasattr(dataset, 'tags'):
                        print(f"  Tags: {dataset.tags}")
            else:
                print(f"  No datasets found for '{term}'")
                
        except Exception as e:
            print(f"  Error searching: {e}")

def test_load_dataset(dataset_id: str):
    """Test loading a specific dataset."""
    print(f"\n{'='*80}")
    print(f"Testing dataset: {dataset_id}")
    print('='*80)
    
    try:
        ds = datasets.load_dataset(dataset_id, split='train[:5]')
        print(f"\nDataset loaded successfully!")
        print(f"Features: {ds.features}")
        print(f"\nFirst row:")
        print(ds[0])
        
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == '__main__':
    search_forex_datasets()
    
    # Test common dataset patterns
    test_datasets = [
        'FredZhang7/forex-2023-5m-high-mid-low-close',
        'ceyda/forex_rates',
    ]
    
    for ds_id in test_datasets:
        try:
            test_load_dataset(ds_id)
        except:
            pass
