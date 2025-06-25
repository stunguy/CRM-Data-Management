import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from openai import OpenAI
import os
import json
import base64
try:
    import fitz
except ImportError:
    fitz = None

st.set_page_config(page_title="Retex Assistant", page_icon="🤖", layout="wide")

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .chat-container {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
    }
    .source-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #d1d5db;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stSelectbox > div > div {
        border-radius: 6px;
    }
    .upload-section {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bfdbfe;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 Retex Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent PDF document assistant</p>', unsafe_allow_html=True)

# Configuration
DB_FAISS_PATH = 'vectorstore/db_faiss'
FILE_LIST_PATH = 'vectorstore/file_list.json'
PDF_STORAGE_PATH = 'vectorstore/pdfs'

# Get API key from Streamlit secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except KeyError:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your Streamlit secrets.")
    st.stop()

def load_existing_db():
    if os.path.exists(DB_FAISS_PATH):
        try:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            return db
        except Exception as e:
            st.sidebar.error(f"Failed to load DB: {str(e)}")
            return None
    return None

def save_file_list(file_list):
    os.makedirs(os.path.dirname(FILE_LIST_PATH), exist_ok=True)
    with open(FILE_LIST_PATH, 'w') as f:
        json.dump(file_list, f)

def load_file_list():
    if os.path.exists(FILE_LIST_PATH):
        try:
            with open(FILE_LIST_PATH, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def filter_content(text):
    """Filter out TOC, blank pages, and low-quality content"""
    text = text.strip()
    
    # Skip if too short (likely blank or image-only)
    if len(text) < 100:
        return False
    
    # Skip table of contents patterns
    toc_patterns = ['table of contents', 'contents', '........', '......', 
                   'chapter', 'section', 'page']
    text_lower = text.lower()
    
    # If mostly dots/numbers (TOC formatting)
    if text.count('.') > len(text) * 0.3:
        return False
    
    # If contains many TOC keywords
    toc_count = sum(1 for pattern in toc_patterns if pattern in text_lower)
    if toc_count >= 3 and len(text) < 500:
        return False
    
    return True

def is_table_content(text):
    """Detect if content contains table data"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    table_indicators = 0
    
    # Look for table patterns
    for line in lines:
        if ('|' in line and line.count('|') >= 2) or \
           ('\t' in line and line.count('\t') >= 2) or \
           (len(line.split()) >= 3 and any(word.replace('.', '').replace(',', '').isdigit() for word in line.split())) or \
           ('troubleshooting' in line.lower() and any(word in line.lower() for word in ['tip', 'solution', 'setting'])):
            table_indicators += 1
    
    return table_indicators >= 1

def create_smart_chunks(documents, filename):
    """Create table-aware chunks preserving tabular data"""
    filtered_docs = []
    for doc in documents:
        if filter_content(doc.page_content):
            filtered_docs.append(doc)
    
    chunks = []
    for doc in filtered_docs:
        content = doc.page_content
        
        # Handle tables differently
        if is_table_content(content):
            # Keep tables as single chunks with larger size
            if len(content.strip()) > 50:
                chunk_doc = Document(
                    page_content=content,
                    metadata={**doc.metadata, 'source_file': filename, 'content_type': 'table'}
                )
                chunks.append(chunk_doc)
        else:
            # Regular text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            split_docs = text_splitter.split_documents([doc])
            for i, chunk in enumerate(split_docs):
                if len(chunk.page_content.strip()) > 50:
                    chunk.metadata.update({
                        'source_file': filename,
                        'chunk_id': i,
                        'content_type': 'text'
                    })
                    chunks.append(chunk)
    
    return chunks

# Initialize session state
if 'db' not in st.session_state:
    st.session_state['db'] = load_existing_db()
if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = load_file_list()
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'last_sources' not in st.session_state:
    st.session_state['last_sources'] = []
if 'view_pdf' not in st.session_state:
    st.session_state['view_pdf'] = None

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Document Management")
    st.markdown("---")
    
    # File upload
    uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("📤 Process Files", type="primary"):
            with st.spinner("Processing PDFs..."):
                try:
                    all_data = []
                    new_files = []
                    
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name not in st.session_state['stored_files']:
                            # Save PDF for viewing
                            os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
                            pdf_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
                            with open(pdf_path, 'wb') as f:
                                f.write(uploaded_file.getvalue())
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name

                            loader = PyMuPDFLoader(file_path=tmp_file_path)
                            data = loader.load()
                            
                            # Apply smart chunking
                            chunked_data = create_smart_chunks(data, uploaded_file.name)
                            all_data.extend(chunked_data)
                            new_files.append(uploaded_file.name)

                    if all_data:
                        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                        
                        if st.session_state['db'] is None:
                            db = FAISS.from_documents(all_data, embeddings)
                        else:
                            new_db = FAISS.from_documents(all_data, embeddings)
                            st.session_state['db'].merge_from(new_db)
                            db = st.session_state['db']
                        
                        db.save_local(DB_FAISS_PATH)
                        st.session_state['db'] = db
                        st.session_state['stored_files'].extend(new_files)
                        save_file_list(st.session_state['stored_files'])
                        
                        st.success(f"✅ Added {len(new_files)} files!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Show stored files
    if st.session_state['stored_files']:
        st.markdown("#### 📚 Stored Documents")
        for file_name in st.session_state['stored_files']:
            st.markdown(f"📄 **{file_name}**")
    
    # Clear button
    if st.session_state['db'] is not None:
        st.divider()
        if st.button("🗑️ Clear All Data", type="secondary"):
            if os.path.exists(DB_FAISS_PATH):
                import shutil
                shutil.rmtree(DB_FAISS_PATH)
            if os.path.exists(FILE_LIST_PATH):
                os.remove(FILE_LIST_PATH)
            if os.path.exists(PDF_STORAGE_PATH):
                import shutil
                shutil.rmtree(PDF_STORAGE_PATH)
            st.session_state['db'] = None
            st.session_state['stored_files'] = []
            st.session_state['messages'] = []
            st.session_state['view_pdf'] = None
            st.rerun()

# Chat interface
def chat_with_pdf(query):
    if st.session_state['db'] is None:
        return "Please upload PDF documents first."
    
    try:
        # Enhanced search for tables
        docs = st.session_state['db'].similarity_search(query, k=6)
        st.session_state['last_sources'] = docs
        
        # Prioritize table content and combine related chunks
        table_docs = [doc for doc in docs if doc.metadata.get('content_type') == 'table']
        text_docs = [doc for doc in docs if doc.metadata.get('content_type') != 'table']
        
        # Combine table content first, then text
        context_parts = [doc.page_content for doc in table_docs] + [doc.page_content for doc in text_docs[:3]]
        context = "\n\n".join(context_parts)
        
        if len(context.strip()) < 50:
            return "Sorry, I can't find this information in the documents."
        
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer based only on the context:"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer based only on the provided context. For tables or lists, include ALL items/rows mentioned. Present information in a clear, structured format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 💬 AI Assistant")
    
    # Chat history container with scrolling
    chat_container = st.container(height=450)
    with chat_container:
        if st.session_state['messages']:
            for message in st.session_state['messages']:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            st.info("👋 **Welcome!** Upload documents and ask questions to get started.")
    
    # Chat input at bottom
    if prompt := st.chat_input("💭 Ask me anything about your documents...", key="main_chat"):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        
        with st.spinner("🔍 Analyzing documents..."):
            response = chat_with_pdf(prompt)
        
        st.session_state['messages'].append({"role": "assistant", "content": response})
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="source-container">', unsafe_allow_html=True)
    # PDF Viewer Controls
    if ('last_sources' in st.session_state and 
        len(st.session_state.get('last_sources', [])) > 0 and
        st.session_state['db'] is not None):
        
        st.markdown("### 📄 Document Sources")
        
        # Create source options
        source_options = ["Select a source..."]
        source_data = {}
        
        for j, doc in enumerate(st.session_state['last_sources'], 1):
            source_file = doc.metadata.get('source_file', 'Unknown')
            page_num = doc.metadata.get('page', 'Unknown')
            option_text = f"{source_file} (Page {page_num})"
            source_options.append(option_text)
            source_data[option_text] = {
                'file': source_file,
                'page': page_num,
                'text': doc.page_content[:500]
            }
        
        selected_source = st.selectbox(
            "Choose source to view:",
            source_options,
            key="pdf_selector"
        )
        
        if selected_source != "Select a source...":
            if st.button("📖 View PDF", key="open_pdf", type="primary"):
                st.session_state['view_pdf'] = source_data[selected_source]
                st.rerun()
    else:
        st.markdown("### 📄 Document Viewer")
        st.info("💡 Ask a question to see relevant source documents here!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# PDF Display
if st.session_state.get('view_pdf'):
    st.divider()
    pdf_info = st.session_state['view_pdf']
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 📄 **{pdf_info['file']}** - Page {pdf_info['page']}")
    with col2:
        if st.button("✖️ Close", key="close_pdf", type="secondary"):
            st.session_state['view_pdf'] = None
            st.rerun()
    
    pdf_path = os.path.join(PDF_STORAGE_PATH, pdf_info['file'])
    
    if os.path.exists(pdf_path):
        try:
            import fitz
            doc = fitz.open(pdf_path)
            
            # Get the specific page
            page_num = pdf_info.get('page', 1)
            if isinstance(page_num, str):
                try:
                    page_num = int(page_num)
                except:
                    page_num = 1
            elif not isinstance(page_num, int):
                page_num = 1
            
            # Correct page index (0-based)
            page_index = max(0, page_num - 1)
            
            # Page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_page", disabled=page_num <= 1):
                    st.session_state['view_pdf']['page'] = page_num - 1
                    st.rerun()
            with col2:
                st.markdown(f"<div style='text-align: center; padding: 8px; font-weight: 500;'>Page **{page_num}** of **{len(doc)}**</div>", unsafe_allow_html=True)
            with col3:
                if st.button("➡️ Next", key="next_page", disabled=page_num >= len(doc)):
                    st.session_state['view_pdf']['page'] = page_num + 1
                    st.rerun()
            
            if page_index < len(doc):
                page = doc[page_index]
                # Render page as image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                st.image(img_data, use_container_width=True)
            else:
                st.error(f"Page {page_num} not found in PDF")
            
            doc.close()
            
        except Exception as e:
            st.error(f"Could not process PDF: {str(e)}")
    else:
        st.error("PDF file not found!")

# Clear chat button
if st.session_state['messages']:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Clear Chat History", key="clear_chat", type="secondary"):
            st.session_state['messages'] = []
            st.rerun()
