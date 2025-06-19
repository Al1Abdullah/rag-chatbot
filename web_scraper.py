import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict
from urllib.parse import urljoin, urlparse
import time
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class WebScraper:
    def __init__(self, delay: float = 1.0):
        """
        Initialize web scraper
        Args:
            delay: Delay between requests to be respectful to servers
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_article(self, url: str) -> Dict[str, str]:
        """
        Scrape article content from a URL
        Args:
            url: URL to scrape
        Returns:
            Dictionary with title, content, and metadata
        """
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            content = self._extract_content(soup)
            
            # Clean and process content
            cleaned_content = self._clean_text(content)
            
            time.sleep(self.delay)  # Be respectful to the server
            
            return {
                'url': url,
                'title': title,
                'content': cleaned_content,
                'word_count': len(cleaned_content.split()),
                'char_count': len(cleaned_content)
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'title': '',
                'content': '',
                'error': str(e),
                'word_count': 0,
                'char_count': 0
            }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML"""
        # Try different title selectors
        title_selectors = [
            'h1',
            'title',
            '.title',
            '.article-title',
            '[data-testid="headline"]'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element and element.get_text().strip():
                return element.get_text().strip()
        
        return "No title found"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'ad']):
            element.decompose()
        
        # Try different content selectors in order of preference
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '.main-content',
            '[role="main"]'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text()
        
        # Fallback: get all paragraph text
        paragraphs = soup.find_all('p')
        return '\n'.join([p.get_text() for p in paragraphs])
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-"]', '', text)
        
        # Remove very short lines (likely navigation/ads)
        lines = text.split('\n')
        meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 20]
        
        return ' '.join(meaningful_lines).strip()

class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize text chunker
        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks
        Args:
            text: Text to chunk
            metadata: Additional metadata to include
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        # Use sentence tokenization for better chunk boundaries
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, create a new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk_dict(chunk_text, metadata, len(chunks)))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.overlap//10:] if len(current_chunk) >= self.overlap//10 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk_dict(chunk_text, metadata, len(chunks)))
        
        return chunks
    
    def _create_chunk_dict(self, text: str, metadata: Dict, chunk_id: int) -> Dict:
        """Create a chunk dictionary with metadata"""
        chunk_dict = {
            'chunk_id': chunk_id,
            'text': text,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
        
        if metadata:
            chunk_dict.update(metadata)
        
        return chunk_dict

# Example usage
if __name__ == "__main__":
    # Test the scraper
    scraper = WebScraper()
    chunker = TextChunker()
    
    # Test URL (replace with your target URL)
    test_url = "https://medium.com/@aminajavaid30/building-a-rag-system-the-data-ingestion-pipeline-d04235fd17ea"
    
    # Scrape content
    article_data = scraper.scrape_article(test_url)
    print(f"Title: {article_data['title']}")
    print(f"Content length: {article_data['word_count']} words")
    
    # Create chunks
    if article_data['content']:
        chunks = chunker.chunk_text(
            article_data['content'], 
            metadata={
                'url': article_data['url'],
                'title': article_data['title']
            }
        )
        print(f"Created {len(chunks)} chunks")
        
        # Show first chunk
        if chunks:
            print(f"First chunk: {chunks[0]['text'][:200]}...")