"""
Telugu Text Data Collection Pipeline
"""

__version__ = "1.0.0"
__author__ = "Data Collection Team"

from .enhanced_scraper import EnhancedTeluguScraper
from .data_cleaner import TeluguDataCleaner

__all__ = [
    'EnhancedTeluguScraper',
    'TeluguDataCleaner',
]
