import gradio as gr
import os
import sys
from typing import List, Tuple
import time
from datetime import datetime

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import RAGChatbot

class ChatbotUI:
    def __init__(self):
        """Initialize the Gradio UI for RAG Chatbot"""
        print("🚀 Initializing Chatbot UI...")
        self.chatbot = RAGChatbot()
        self.chat_history = []
        
    def add_url(self, url: str) -> Tuple[str, str]:
        """
        Add URL to knowledge base
        Args:
            url: URL to ingest
        Returns:
            Tuple of (status_message, updated_stats)
        """
        if not url or not url.strip():
            return "❌ Please enter a valid URL", self.get_stats_display()
        
        url = url.strip()
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'https://' + url
        
        # Show processing message
        status_msg = f"📥 Processing {url}..."
        
        try:
            result = self.chatbot.ingest_url(url)
            
            if result['success']:
                success_msg = f"""✅ Successfully added: {result['title']}
📊 Added {result['chunks_added']} chunks ({result['word_count']} words)
🔗 Source: {url}"""
                return success_msg, self.get_stats_display()
            else:
                error_msg = f"❌ Failed to add URL: {result['message']}"
                return error_msg, self.get_stats_display()
                
        except Exception as e:
            error_msg = f"❌ Error processing URL: {str(e)}"
            return error_msg, self.get_stats_display()
    
    def chat_response(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """
        Generate chat response
        Args:
            message: User message
            history: Chat history
        Returns:
            Tuple of (empty_string, updated_history)
        """
        if not message or not message.strip():
            return "", history
        
        # Get response from chatbot
        response_data = self.chatbot.chat(message.strip(), include_sources=True)
        
        # Format response with sources
        formatted_response = self.format_response(response_data)
        
        # Update history
        history.append([message, formatted_response])
        
        return "", history
    
    def format_response(self, response_data: dict) -> str:
        """Format the chatbot response with sources and timing info"""
        response = response_data['response']
        
        # Add timing information
        timing_info = f"\n\n⏱️ *Response time: {response_data['total_time']}s*"
        
        # Add sources if available
        if response_data.get('sources'):
            sources_text = "\n\n📚 **Sources:**\n"
            for i, source in enumerate(response_data['sources'][:3], 1):  # Limit to top 3 sources
                score = f"({source['similarity_score']:.3f})" if source['similarity_score'] else ""
                sources_text += f"{i}. **{source['title']}** {score}\n"
                sources_text += f"   {source['snippet']}\n"
                sources_text += f"   🔗 {source['url']}\n\n"
            
            response += sources_text
        
        response += timing_info
        return response
    
    def get_stats_display(self) -> str:
        """Get formatted knowledge base statistics"""
        try:
            stats = self.chatbot.get_knowledge_base_stats()
            
            stats_text = f"""📊 **Knowledge Base Statistics**
            
🗄️ **Total Documents:** {stats.get('total_documents', 0)}
🧠 **AI Model:** {stats.get('model_used', 'Unknown')}
🔤 **Embedding Model:** {stats.get('embedding_model', 'Unknown')}
📐 **Vector Dimension:** {stats.get('index_dimension', 0)}
📈 **Index Fullness:** {stats.get('index_fullness', 0):.1%}

*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
            
            return stats_text
            
        except Exception as e:
            return f"❌ Error getting stats: {str(e)}"
    
    def clear_knowledge_base(self) -> Tuple[str, str]:
        """Clear all documents from knowledge base"""
        try:
            success = self.chatbot.clear_knowledge_base()
            if success:
                return "✅ Knowledge base cleared successfully!", self.get_stats_display()
            else:
                return "❌ Failed to clear knowledge base", self.get_stats_display()
        except Exception as e:
            return f"❌ Error clearing knowledge base: {str(e)}", self.get_stats_display()
    
    def create_interface(self):
        """Create and return the Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 500px !important;
        }
        .input-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            padding: 20px !important;
            border-radius: 10px !important;
        }
        """
        
        with gr.Blocks(
            title="🤖 RAG Chatbot",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🤖 RAG-Powered AI Chatbot
            ### Intelligent Q&A with Web Content Integration
            
            **How to use:**
            1. 📥 Add URLs containing articles or content you want the bot to learn from
            2. 💬 Ask questions about the content - the bot will provide accurate answers with sources
            3. 📊 Monitor your knowledge base statistics in the sidebar
            """)
            
            with gr.Row():
                # Main chat area (left side)
                with gr.Column(scale=2):
                    # URL Input Section
                    gr.Markdown("## 📥 Add Content to Knowledge Base")
                    with gr.Row():
                        url_input = gr.Textbox(
                            placeholder="Enter URL (e.g., https://medium.com/article-url)",
                            label="Website URL",
                            scale=3
                        )
                        add_btn = gr.Button("Add URL", variant="primary", scale=1)
                    
                    url_status = gr.Markdown(value="", visible=True)
                    
                    # Chat Interface
                    gr.Markdown("## 💬 Chat with Your Knowledge Base")
                    
                    chatbot_interface = gr.Chatbot(
                        value=[],
                        height=400,
                        label="RAG Chatbot",
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask a question about your added content...",
                            label="Your Message",
                            scale=4,
                            lines=1
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    # Example questions
                    gr.Markdown("### 💡 Example Questions:")
                    example_questions = [
                        "What is the main topic of this article?",
                        "Can you summarize the key points?",
                        "What are the benefits mentioned?",
                        "How does this relate to AI/ML?"
                    ]
                    
                    with gr.Row():
                        for question in example_questions[:2]:
                            gr.Button(question, size="sm").click(
                                lambda q=question: (q, ""),
                                outputs=[msg_input, url_status]
                            )
                    
                    with gr.Row():
                        for question in example_questions[2:]:
                            gr.Button(question, size="sm").click(
                                lambda q=question: (q, ""),
                                outputs=[msg_input, url_status]
                            )
                
                # Sidebar (right side)
                with gr.Column(scale=1):
                    gr.Markdown("## 📊 Knowledge Base")
                    
                    stats_display = gr.Markdown(
                        value=self.get_stats_display(),
                        label="Statistics"
                    )
                    
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                    clear_kb_btn = gr.Button("🗑️ Clear Knowledge Base", variant="stop")
                    
                    gr.Markdown("""
                    ### ℹ️ About
                    This RAG chatbot uses:
                    - **Groq API** with Mixtral-8x7B for fast inference
                    - **Faiss** for vector storage
                    - **Sentence Transformers** for embeddings
                    - **Beautiful Soup** for web scraping
                    
                    The bot retrieves relevant content and generates accurate answers based on your added sources.
                    
                    -Made By Ali Abdullah"""    
                       )
            
            # Event handlers
            add_btn.click(
                fn=self.add_url,
                inputs=[url_input],
                outputs=[url_status, stats_display]
            ).then(
                lambda: "",  # Clear URL input after adding
                outputs=[url_input]
            )
            
            send_btn.click(
                fn=self.chat_response,
                inputs=[msg_input, chatbot_interface],
                outputs=[msg_input, chatbot_interface]
            )
            
            msg_input.submit(
                fn=self.chat_response,
                inputs=[msg_input, chatbot_interface],
                outputs=[msg_input, chatbot_interface]
            )
            
            refresh_stats_btn.click(
                fn=lambda: self.get_stats_display(),
                outputs=[stats_display]
            )
            
            clear_kb_btn.click(
                fn=self.clear_knowledge_base,
                outputs=[url_status, stats_display]
            )
        
        return interface

def main():
    """Main function to run the Gradio app"""
    print("🚀 Starting RAG Chatbot UI...")
    
    try:
        # Initialize the UI
        ui = ChatbotUI()
        
        # Create and launch interface
        interface = ui.create_interface()
        
        # Launch with custom settings
        interface.launch(
            server_name="127.0.0.1",  # Allow external access
            server_port=7860,       # Default Gradio port
            share=True             # Set to True for public link
        )

    except Exception as e:
        print(f"❌ Failed to launch the app: {e}")

if __name__ == "__main__":
    main()

