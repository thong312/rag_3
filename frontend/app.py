import streamlit as st
import requests

from typing import Dict, Any

# Constants
API_URL = "http://127.0.0.1:8080"  

def send_chat_request(query: str, search_type: str = "hybrid", **kwargs) -> Dict[str, Any]:
    """Send chat request to backend API"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "query": query,
                "search_type": search_type,
                **kwargs
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def upload_pdf(file):
    """Upload PDF file to backend"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_URL}/pdf", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def upload_idioms(file, source_name="idioms"):
    """Upload idioms file to backend"""
    try:
        files = {"file": file}
        data = {"source_name": source_name}
        response = requests.post(f"{API_URL}/idioms", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def check_api_connection():
    """Check if backend API is running"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def main():
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG Chat Assistant")

    # Check API connection
    if not check_api_connection():
        st.error("""
        ⚠️ Cannot connect to backend API. Please ensure:
        1. Backend Flask server is running on port 5000
        2. You started backend with `python app.py`
        3. There are no firewall issues blocking the connection
        """)
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # File Upload Section with Tabs
        st.subheader("Upload Files")
        upload_tab1, upload_tab2 = st.tabs(["PDF Upload", "Idioms Upload"])
        
        with upload_tab1:
            # PDF Upload
            pdf_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
            if pdf_file:
                if st.button("Upload PDF"):
                    with st.spinner("Uploading and processing PDF..."):
                        result = upload_pdf(pdf_file)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(f"PDF uploaded successfully! {result['chunks']} chunks created")

        with upload_tab2:
            # Idioms Upload
            idioms_file = st.file_uploader("Choose an idioms PDF", type="pdf", key="idioms_uploader")
            source_name = st.text_input("Source Name", value="idioms", 
                                      help="Name to identify this idioms collection")
            if idioms_file:
                if st.button("Upload Idioms"):
                    with st.spinner("Uploading and processing idioms..."):
                        result = upload_idioms(idioms_file, source_name)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(f"""
                                Idioms uploaded successfully!
                                - Documents: {result['docs']}
                                - Chunks: {result['chunks']}
                                - Total in database: {result['final_count']}
                            """)

        # Search Settings
        st.subheader("Search Settings")
        search_type = "hybrid"  # cố định search method
        st.write(f"Search Method: {search_type}")
        
        # Add filter for idioms
        if search_type == "hybrid":
            st.subheader("Filters")
            doc_type = "all"
            metadata_filter = None
            if doc_type != "all":
                metadata_filter = {"type": doc_type}

            alpha = st.slider(
                "Alpha (Vector vs BM25)", 
                0.0, 1.0, 0.5,
                help="1.0 = Pure Vector, 0.0 = Pure BM25"
            )
        # Set cứng Use Rerank = True
        use_rerank = True
        st.write(f"Use Reranking: {use_rerank}")

        # Set cứng k = 3
        k = 3
        st.write(f"Number of results: {k}")

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display sources if available
            # if "sources" in message and message["sources"]:
            #     with st.expander("View Sources"):
            #         for idx, source in enumerate(message["sources"], 1):
            #             st.markdown(f"""
            #             **Source {idx}:**
            #             - File: {source['file_name']}
            #             - Page: {source['page']}
            #             - Chunk: {source['chunk']}
            #             """)

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if search_type == "hybrid":
                    response = send_chat_request(
                        prompt,
                        search_type="hybrid",
                        alpha=alpha,
                        k=k,
                        use_rerank=use_rerank,
                        metadata_filter=metadata_filter  # Add this
                    )
                else:
                    response = send_chat_request(prompt, search_type=search_type)

                if "error" in response:
                    st.error(response["error"])
                else:
                    st.write(response["answer"])
                    if response.get("sources"):
                        with st.expander("View Sources"):
                            for idx, source in enumerate(response["sources"], 1):
                                st.markdown(f"""
                                **Source {idx}:**
                                - File: {source['file_name']}
                                - Page: {source['page']}
                                - Chunk: {source['chunk']}
                                """)

                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.get("answer", "Error occurred"),
                    "sources": response.get("sources", [])
                })

    # Clear chat button


if __name__ == "__main__":
    main()