# app.py
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot with Debug & Better QA")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    
    st.success("✅ PDF loaded successfully!")

    # Split text
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    st.write(f"📑 Total chunks created: {len(chunks)}")

    # Embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L3-v2")

    # Load or create vectorstore (safe load)
    if os.path.exists("vectorstore.faiss"):
        vectorstore = FAISS.load_local("vectorstore.faiss", embeddings, allow_dangerous_deserialization=True)
        st.info("🔄 Loaded embeddings from cache")
    else:
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local("vectorstore.faiss")
        st.info("💾 Embeddings created and saved")

    # Use a better model
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # more reliable than small
        max_length=512
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Conversational retrieval chain with sources
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True,
        verbose=True
    )

    # Sidebar chat
    st.sidebar.header("💬 Ask Questions")
    if "history" not in st.session_state:
        st.session_state.history = []

    user_question = st.sidebar.text_input("Type your question here:")
    if st.sidebar.button("Ask") and user_question:
        result = qa({"question": user_question, "chat_history": st.session_state.history})
        st.session_state.history.append((user_question, result["answer"]))

        # Debug retrieved context
        st.subheader("🔎 Retrieved Context Chunks")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:400]}...")

    # Display chat history
    st.subheader("💬 Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

    # Summary
    if st.button("📝 Generate 2-Page Summary"):
        summary_splitter = CharacterTextSplitter(separator="\n", chunk_size=4000, chunk_overlap=400)
        segments = summary_splitter.split_text(text)
        summary_text = ""
        for seg in segments:
            prompt = f"summarize: {seg}"
            summary = llm(prompt)
            summary_text += summary + "\n\n"

        st.subheader("📄 2-Page Summary")
        st.text(summary_text)
