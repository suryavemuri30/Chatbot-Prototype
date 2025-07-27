import streamlit as st
import os
from streamlit_chat import message
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
import requests
import re
import time

# Configuration
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

def check_environment():
    """Check and display environment variable status"""
    pinecone_key = os.environ.get('PINECONE_API_KEY')
    hf_key = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    
    st.write("🔍 **Environment Check:**")
    
    if pinecone_key:
        masked_key = pinecone_key[:8] + "..." + pinecone_key[-4:] if len(pinecone_key) > 12 else "Set"
        st.success(f"✅ PINECONE_API_KEY: {masked_key}")
    else:
        st.error("❌ PINECONE_API_KEY not found")
        return False
    
    if hf_key:
        masked_hf = hf_key[:8] + "..." + hf_key[-4:] if len(hf_key) > 12 else "Set"
        st.success(f"✅ HUGGINGFACEHUB_API_TOKEN: {masked_hf}")
    else:
        st.error("❌ HUGGINGFACEHUB_API_TOKEN not found")
        return False
    
    return True

def manual_api_key_setup():
    """Allow manual API key input as fallback"""
    st.write("### 🔧 Manual API Key Setup")
    
    pinecone_key = st.text_input(
        "Enter your Pinecone API Key:", 
        type="password",
        help="Get your API key from https://app.pinecone.io"
    )
    
    hf_key = st.text_input(
        "Enter your HuggingFace API Token:", 
        type="password",
        help="Get your token from https://huggingface.co/settings/tokens"
    )
    
    if st.button("Set API Keys"):
        if pinecone_key and hf_key:
            os.environ['PINECONE_API_KEY'] = pinecone_key
            os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_key
            st.success("✅ API keys set successfully!")
            st.rerun()
        else:
            st.error("❌ Please provide both API keys")

def test_huggingface_connection():
    """Test HuggingFace API connection"""
    try:
        api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
        if not api_token:
            st.error("❌ No API token found")
            return False
        
        if not api_token.startswith('hf_'):
            st.error("❌ Invalid token format. Should start with 'hf_'")
            return False
        
        headers = {"Authorization": f"Bearer {api_token}"}
        response = requests.get("https://huggingface.co/api/whoami", headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_info = response.json()
            st.success(f"✅ Token valid for user: {user_info.get('name', 'Unknown')}")
            return True
        else:
            st.error(f"❌ Token validation failed: {response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"❌ Connection test failed: {e}")
        return False

def get_documents_directory():
    """Get the correct documents directory path for macOS"""
    directory = os.path.expanduser("~/Documents/Chatbot-Prototype/content/")
    
    st.write("📁 **Directory Information:**")
    st.write(f"**Raw path:** `~/Documents/Chatbot-Prototype/content/`")
    st.write(f"**Expanded path:** `{directory}`")
    st.write(f"**Directory exists:** {os.path.exists(directory)}")
    
    if os.path.exists(directory):
        try:
            files = os.listdir(directory)
            st.write(f"**Files found:** {len(files)} files")
            if files:
                st.write("**File types:**")
                file_extensions = {}
                for file in files[:10]:
                    ext = os.path.splitext(file)[1].lower() or 'no extension'
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
                for ext, count in file_extensions.items():
                    st.write(f"  - {ext}: {count} files")
            else:
                st.warning("⚠️ Directory exists but is empty")
        except PermissionError:
            st.error("❌ Permission denied accessing directory")
        except Exception as e:
            st.error(f"❌ Error reading directory: {e}")
    else:
        st.error("❌ Directory not found")
        st.info("💡 **Create the directory**: Run `mkdir -p ~/Documents/Chatbot-Prototype/content/` in Terminal")
    
    return directory

@st.cache_data
def load_docs(_directory):
    """Load documents from directory"""
    try:
        expanded_directory = os.path.expanduser(_directory)
        
        if not os.path.exists(expanded_directory):
            st.error(f"❌ Directory not found: {expanded_directory}")
            return []
        
        loader = DirectoryLoader(expanded_directory)
        documents = loader.load()
        
        if documents:
            st.success(f"✅ Successfully loaded {len(documents)} documents")
        else:
            st.warning("⚠️ No documents found in the directory")
        
        return documents
    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")
        return []

@st.cache_data
def split_docs(_documents, chunk_size=500, chunk_overlap=20):
    """Split documents into chunks"""
    if not _documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(_documents)
    st.info(f"📄 Split {len(_documents)} documents into {len(docs)} chunks")
    return docs

@st.cache_resource
def init_pinecone():
    """Initialize Pinecone client"""
    try:
        api_key = os.environ.get('PINECONE_API_KEY')
        if not api_key:
            st.error("❌ PINECONE_API_KEY environment variable not set")
            return None
        
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        st.success(f"✅ Successfully connected to Pinecone. Found {len(indexes)} indexes.")
        return pc
    except Exception as e:
        st.error(f"❌ Error initializing Pinecone: {str(e)}")
        return None

@st.cache_resource
def get_embeddings():
    """Get HuggingFace embeddings - 768D model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def get_vectorstore(_docs, _pc):
    """Get or create vector store"""
    try:
        index_name = "langchain-chatbot2"
        embeddings = get_embeddings()
        
        test_embedding = embeddings.embed_query("test")
        embedding_dim = len(test_embedding)
        st.info(f"📏 Embedding dimension: {embedding_dim}D")
        
        existing_indexes = [index.name for index in _pc.list_indexes()]
        
        if index_name in existing_indexes:
            index_info = _pc.describe_index(index_name)
            index_dim = index_info.dimension
            st.info(f"📊 Existing index dimension: {index_dim}D")
            
            if index_dim != embedding_dim:
                st.error(f"❌ Dimension mismatch: Index={index_dim}D, Embeddings={embedding_dim}D")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Delete & Recreate Index"):
                        _pc.delete_index(index_name)
                        st.success("Index deleted. Please restart the app.")
                        st.stop()
                with col2:
                    st.info("Or restart app to recreate index")
                st.stop()
        else:
            _pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            st.info("⏳ Creating new Pinecone index...")
            time.sleep(10)
        
        vectorstore = PineconeVectorStore.from_documents(
            _docs, embedding=embeddings, index_name=index_name
        )
        st.success("✅ Vector store created successfully")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Failed to create vector store: {e}")
        return None

def create_mock_llm():
    """Create a simple mock LLM for testing purposes"""
    from langchain.llms.base import LLM
    from typing import Any, List, Mapping, Optional
    
    class MockLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "mock"
        
        def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
            if "context" in prompt.lower():
                return "Based on the provided documents, I can process your question. However, I'm currently using a mock LLM due to HuggingFace connection issues. The document retrieval system is working properly."
            return "I'm a mock LLM. Your document processing system is working, but please configure a proper language model for better responses."
        
        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"type": "mock"}
    
    return MockLLM()

@st.cache_resource
def get_llm():
    """Get LLM with comprehensive fallback options"""
    
    api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    if not api_token:
        st.error("❌ Missing HUGGINGFACEHUB_API_TOKEN")
        return create_mock_llm()
    
    models_to_try = [
        {
            "repo_id": "google/flan-t5-base",
            "params": {"temperature": 0.7, "max_new_tokens": 256, "timeout": 30}
        },
        {
            "repo_id": "microsoft/DialoGPT-medium",
            "params": {"temperature": 0.7, "max_new_tokens": 200, "timeout": 30}
        },
        {
            "repo_id": "google/flan-t5-small",
            "params": {"temperature": 0.7, "max_new_tokens": 128, "timeout": 30}
        }
    ]
    
    for model_config in models_to_try:
        try:
            st.info(f"🔄 Trying model: {model_config['repo_id']}")
            
            llm = HuggingFaceEndpoint(
                repo_id=model_config['repo_id'],
                huggingfacehub_api_token=api_token,
                **model_config['params']
            )
            
            test_response = llm.invoke("Hello")
            if test_response and len(test_response.strip()) > 0:
                st.success(f"✅ Successfully initialized: {model_config['repo_id']}")
                return llm
            else:
                st.warning(f"⚠️ Empty response from: {model_config['repo_id']}")
                
        except Exception as e:
            st.warning(f"⚠️ Failed {model_config['repo_id']}: {str(e)}")
            continue
    
    try:
        st.info("🔄 Trying local HuggingFace pipeline...")
        from langchain_huggingface import HuggingFacePipeline
        import transformers.pipelines as pipe1
        
        pipe = pipe1.pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_length=256,
            do_sample=True,
            temperature=0.7
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        test_response = llm.invoke("Hello")
        if test_response:
            st.success("✅ Local HuggingFace pipeline initialized")
            return llm
            
    except Exception as e:
        st.warning(f"⚠️ Local pipeline failed: {e}")
    
    try:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if openai_key:
            st.info("🔄 Trying OpenAI as fallback...")
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
            )
            
            test_response = llm.invoke("Hello")
            if test_response:
                st.success("✅ OpenAI fallback initialized")
                return llm
    except Exception as e:
        st.warning(f"⚠️ OpenAI fallback failed: {e}")
    
    st.error("❌ All LLM options failed. Using mock LLM for testing.")
    return create_mock_llm()

@st.cache_resource
def get_conversation_chain(_vectorstore, _llm):
    """Get conversation chain"""
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=_llm,
            retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            verbose=True
        )
        st.success("✅ Conversation chain created successfully")
        return chain
    except Exception as e:
        st.error(f"❌ Failed to create conversation chain: {e}")
        return None

def main():
    st.title("🤖 Document Chatbot with Pinecone & HuggingFace")
    st.markdown("Ask questions about your documents and get AI-powered answers!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if not check_environment():
            st.warning("⚠️ Environment variables not properly set.")
            manual_api_key_setup()
            st.stop()
        
        st.subheader("🔍 Connection Test")
        if st.button("Test HuggingFace Connection"):
            if not test_huggingface_connection():
                st.error("❌ HuggingFace connection failed")
        
        st.subheader("📄 Document Settings")
        chunk_size = st.slider("Chunk Size", 200, 1000, 500)
        chunk_overlap = st.slider("Chunk Overlap", 0, 100, 20)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        directory = get_documents_directory()
        documents = load_docs(directory)
        if not documents:
            st.stop()
    
    with col2:
        st.subheader("📊 System Status")
    
    docs = split_docs(documents, chunk_size, chunk_overlap)
    if not docs:
        st.error("❌ No document chunks created")
        st.stop()
    
    # Initialize components
    with st.spinner("Initializing system components..."):
        pc = init_pinecone()
        if pc is None:
            st.stop()
        
        vectorstore = get_vectorstore(docs, pc)
        if vectorstore is None:
            st.stop()
        
        llm = get_llm()
        if llm is None:
            st.error("❌ No working LLM found")
            st.stop()
        
        conversation_chain = get_conversation_chain(vectorstore, llm)
        if conversation_chain is None:
            st.stop()
    
    # Chat interface - FIXED VERSION
    st.markdown("---")
    st.subheader("💬 Chat with Your Documents")
    
    # Initialize session state for chat
    if 'chat_responses' not in st.session_state:
        st.session_state['chat_responses'] = ["Hi! I'm ready to help you with your documents. What would you like to know?"]
    
    if 'chat_requests' not in st.session_state:
        st.session_state['chat_requests'] = []
    
    if 'last_query' not in st.session_state:
        st.session_state['last_query'] = ""
    
    # Chat input with form to prevent looping
    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input(
            "Ask your question:", 
            placeholder="What would you like to know about your documents?"
        )
        submitted = st.form_submit_button("Send")
    
    # Process query only when form is submitted and query is new
    if submitted and query and query != st.session_state['last_query']:
        st.session_state['last_query'] = query
        
        with st.spinner("🤔 Analyzing documents and generating response..."):
            try:
                result = conversation_chain.invoke({"question": query})
                response = result.get('answer', 'No response generated')
            except Exception as e:
                response = f"❌ Error generating response: {e}"
                st.error(f"Detailed error: {str(e)}")
        
        # Add to chat history
        st.session_state.chat_requests.append(query)
        st.session_state.chat_responses.append(response)
    
    # Display chat history
    if st.session_state['chat_responses']:
        st.subheader("💭 Conversation History")
        
        for i in range(len(st.session_state['chat_responses'])):
            message(st.session_state['chat_responses'][i], key=str(i))
            if i < len(st.session_state['chat_requests']):
                message(st.session_state["chat_requests"][i], is_user=True, key=str(i) + '_user')
    
    # Chat controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state['chat_responses'] = ["Hi! I'm ready to help you with your documents. What would you like to know?"]
            st.session_state['chat_requests'] = []
            st.session_state['last_query'] = ""
            st.rerun()
    
    with col2:
        if st.button("📊 Show Stats"):
            st.info(f"📄 {len(documents)} documents loaded, {len(docs)} chunks processed")
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by LangChain, Pinecone, and HuggingFace*")

if __name__ == "__main__":
    main()
