import os
from typing import List, Dict, Optional, Tuple
from groq import Groq
from dotenv import load_dotenv
from web_scraper import WebScraper, TextChunker

from vector_store import VectorStore
import time

load_dotenv()

class RAGChatbot:
    def __init__(self):
        """Initialize RAG Chatbot with all components"""
        print("🤖 Initializing RAG Chatbot...")
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize components
        self.vector_store = VectorStore()
        self.web_scraper = WebScraper(delay=1.0)
        self.text_chunker = TextChunker(
            chunk_size=int(os.getenv("MAX_CHUNK_SIZE", 500)),
            overlap=50
        )
        
        # Configuration
        self.model_name = "llama3-8b-8192"
        self.top_k = int(os.getenv("TOP_K_RESULTS", 5))
        self.max_tokens = 1000
        
        print("✅ RAG Chatbot initialized successfully!")
    
    def ingest_url(self, url: str) -> Dict[str, any]:
        """
        Ingest content from a URL into the knowledge base
        Args:
            url: URL to scrape and ingest
        Returns:
            Dictionary with ingestion results
        """
        try:
            print(f"📥 Ingesting content from: {url}")
            
            # Scrape the article
            article_data = self.web_scraper.scrape_article(url)
            
            if not article_data['content']:
                return {
                    'success': False,
                    'message': f"Could not extract content from {url}",
                    'chunks_added': 0
                }
            
            # Create chunks
            chunks = self.text_chunker.chunk_text(
                article_data['content'],
                metadata={
                    'url': article_data['url'],
                    'title': article_data['title']
                }
            )
            
            if not chunks:
                return {
                    'success': False,
                    'message': "No valid chunks created from content",
                    'chunks_added': 0
                }
            
            # Add to vector store
            success = self.vector_store.add_documents(chunks)
            
            if success:
                return {
                    'success': True,
                    'message': f"Successfully ingested '{article_data['title']}'",
                    'chunks_added': len(chunks),
                    'title': article_data['title'],
                    'word_count': article_data['word_count']
                }
            else:
                return {
                    'success': False,
                    'message': "Failed to add chunks to vector store",
                    'chunks_added': 0
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Error ingesting {url}: {str(e)}",
                'chunks_added': 0
            }
    
    def chat(self, message: str, include_sources: bool = True) -> Dict[str, any]:
        """
        Chat with the RAG system
        Args:
            message: User's question/message
            include_sources: Whether to include source information
        Returns:
            Dictionary with response and metadata
        """
        try:
            print(f"💬 Processing query: {message[:50]}...")
            
            # Step 1: Retrieve relevant context
            start_time = time.time()
            relevant_docs = self.vector_store.search_similar(message, top_k=self.top_k)
            retrieval_time = time.time() - start_time
            
            if not relevant_docs:
                return {
                    'response': "I don't have enough information to answer your question. Please add some relevant content to my knowledge base first.",
                    'sources': [],
                    'retrieval_time': retrieval_time,
                    'generation_time': 0,
                    'total_time': retrieval_time
                }
            
            # Step 2: Create context from retrieved documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"Context {i+1}: {doc['text']}")
                sources.append({
                    'title': doc['title'],
                    'url': doc['url'],
                    'similarity_score': doc['score'],
                    'snippet': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                })
            
            context = "\n\n".join(context_parts)
            
            # Step 3: Generate response using Groq
            generation_start = time.time()
            response = self._generate_response(message, context)
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            return {
                'response': response,
                'sources': sources if include_sources else [],
                'retrieval_time': round(retrieval_time, 3),
                'generation_time': round(generation_time, 3),
                'total_time': round(total_time, 3),
                'context_used': len(relevant_docs)
            }
            
        except Exception as e:
            return {
                'response': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'retrieval_time': 0,
                'generation_time': 0,
                'total_time': 0,
                'error': str(e)
            }
    
    def _generate_response(self, query: str, context: str) -> str:
        """
        Generate response using Groq API
        Args:
            query: User's question
            context: Retrieved context
        Returns:
            Generated response
        """
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 

Guidelines:
- Use ONLY the information provided in the context to answer questions
- If the context doesn't contain enough information, say so clearly
- Be accurate and cite specific details from the context
- Provide comprehensive answers but stay focused on the question
- If asked about sources, refer to the context provided
- Be conversational and helpful in your tone
"""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a detailed answer based on the context above. If the context doesn't contain sufficient information to answer the question, please say so clearly."""

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for more focused responses
                top_p=0.9
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_knowledge_base_stats(self) -> Dict[str, any]:
        """Get statistics about the knowledge base"""
        try:
            stats = self.vector_store.get_index_stats()
            return {
                'total_documents': stats.get('total_vectors', 0),
                'index_dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0),
                'model_used': self.model_name,
                'embedding_model': os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            }
        except Exception as e:
            return {'error': str(e)}
    
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from knowledge base"""
        try:
            return self.vector_store.delete_all()
        except Exception as e:
            print(f"Error clearing knowledge base: {str(e)}")
            return False

# Test the chatbot
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Test ingestion (replace with your URL)
    test_url = "https://medium.com/@aminajavaid30/building-a-rag-system-the-data-ingestion-pipeline-d04235fd17ea"
    
    print("Testing content ingestion...")
    ingestion_result = chatbot.ingest_url(test_url)
    print(f"Ingestion result: {ingestion_result}")
    
    if ingestion_result['success']:
        print("\nTesting chat functionality...")
        
        # Test questions
        test_questions = [
            "What is RAG?",
            "How does the data ingestion pipeline work?",
            "What are the main components of a RAG system?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            response = chatbot.chat(question)
            print(f"🤖 Answer: {response['response']}")
            print(f"⏱️ Time: {response['total_time']}s (Retrieval: {response['retrieval_time']}s, Generation: {response['generation_time']}s)")
            print(f"📚 Sources used: {response['context_used']}")
    
    # Show knowledge base stats
    stats = chatbot.get_knowledge_base_stats()
    print(f"\n📊 Knowledge Base Stats: {stats}")