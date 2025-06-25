import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
import os
import json
import base64
try:
    import fitz
except ImportError:
    fitz = None

st.set_page_config(page_title="Retex Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Retex Assistant")
st.markdown("*Your intelligent PDF document assistant*")

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
    st.header("📁 Document Management")
    
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
                            
                            for doc in data:
                                doc.metadata['source_file'] = uploaded_file.name
                            
                            all_data.extend(data)
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
        st.subheader("📚 Stored Documents")
        for file_name in st.session_state['stored_files']:
            st.text(f"• {file_name}")
    
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
        docs = st.session_state['db'].similarity_search(query, k=2)
        st.session_state['last_sources'] = docs
        
        context = "\n".join([doc.page_content[:1000] for doc in docs])
        
        if len(context.strip()) < 50:
            return "Sorry, I can't find this information in the documents."
        
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer based only on the context:"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer only based on the provided context. If not found, say 'Sorry, I can't find this information in the documents.'"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat")
    
    # Display messages
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_pdf(prompt)
                st.write(response)
                st.session_state['messages'].append({"role": "assistant", "content": response})

with col2:
    # PDF Viewer Controls
    if ('last_sources' in st.session_state and 
        len(st.session_state.get('last_sources', [])) > 0 and
        st.session_state['db'] is not None):
        
        st.header("📄 View Sources")
        
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
        st.header("📄 PDF Viewer")
        st.info("Ask a question to see source documents here!")

# PDF Display (full width at bottom)
if st.session_state.get('view_pdf'):
    st.divider()
    pdf_info = st.session_state['view_pdf']
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader(f"📄 {pdf_info['file']} - Page {pdf_info['page']}")
    with col2:
        if st.button("✖️ Close", key="close_pdf"):
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
            
            # Add offset to account for page numbering mismatch
            page_offset = 2
            page_index = max(0, page_num - 1 + page_offset)
            
            # Page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", key="prev_page"):
                    new_page = max(1, page_num - 1)
                    st.session_state['view_pdf']['page'] = new_page
                    st.rerun()
            with col2:
                st.write(f"Page {page_num} of {len(doc)}")
            with col3:
                if st.button("➡️ Next", key="next_page"):
                    new_page = min(len(doc), page_num + 1)
                    st.session_state['view_pdf']['page'] = new_page
                    st.rerun()
            
            if page_index < len(doc):
                page = doc[page_index]
                
                # Search for text and auto-correct page if needed
                full_text = pdf_info['text'].replace('\n', ' ').strip()
                words = full_text.split()
                meaningful_words = []
                
                for word in words:
                    if (not word.replace('-', '').replace('.', '').isdigit() and
                        len(word) > 2 and
                        not word.isupper() or word in ['KNIT', 'SCAN']):
                        meaningful_words.append(word)
                    if len(meaningful_words) >= 8:
                        break
                
                if len(meaningful_words) < 3:
                    search_phrase = ' '.join(words[10:18]) if len(words) > 10 else ' '.join(words[:8])
                else:
                    search_phrase = ' '.join(meaningful_words)
                
                # Search all pages for this text
                found_pages = []
                for page_idx in range(len(doc)):
                    test_page = doc[page_idx]
                    page_text = test_page.get_text()
                    page_text_clean = page_text.lower().replace('\n', ' ')
                    if (search_phrase.lower() in page_text_clean or
                        any(word.lower() in page_text_clean for word in meaningful_words[-3:]) or
                        any(word.lower() in page_text_clean for word in meaningful_words[:3])):
                        found_pages.append(page_idx + 1)
                
                if found_pages and page_num not in found_pages:
                    correct_page = found_pages[0]
                    st.session_state['view_pdf']['page'] = correct_page
                    st.rerun()
                
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
    st.divider()
    if st.button("🔄 Clear Chat History", key="clear_chat"):
        st.session_state['messages'] = []
        st.rerun()
