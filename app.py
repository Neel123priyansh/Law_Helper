import os 
import fitz  # PyMuPDF
import json 
import streamlit as st
from typing import List 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI    
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS  
from langchain_pinecone import PineconeVectorStore  
from pinecone import Pinecone
from langchain.chains import LLMChain 
import pytesseract
from PIL import Image 

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
LLM_MODEL_NAME = "gemini-1.5-flash"
INDEX_NAME = "lawyerlens-kb"
GOOGLE_API_KEY = "AIzaSyDUkq0nrIXwPfVkS3tgagS7zV6EtCMPSss"
os.environ["PINECONE_API_KEY"] = "pcsk_3aziQH_FhYmdBpXgFTL6borDTmnUcMFSnsXd55CXq5YdYmge17qF14iQP7ZScvAXxnT261"

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
    )

# Extract text from PDFs
def get_pdf_text(pdf_docs: List) -> str:
    text = ""
    for pdf_file in pdf_docs:
        try:
            # Read the file into memory first
            file_bytes = pdf_file.read()  

            # Open with fitz
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc, start=1):
                    page_text = page.get_text()

                    # OCR fallback if no text
                    if not page_text.strip():
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        page_text = pytesseract.image_to_string(img)
                        

                    text += f"\n--- Page {page_num} ---\n" + page_text

        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {e}")
    return text

def get_text_chunks(raw_text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def generate_improved_clauses_with_location(chunks, recommendations, llm):
    template = """
    You are a contract drafting assistant.
    Based on the following recommendations, rewrite or add improved clauses.

    For each improvement, specify:
    - Improved Text
    - Page number
    - Paragraph number (where it belongs)

    Recommendations:
    {recs}

    Agreement Chunks:
    {doc}

    Return JSON like:
    [
      {{
        "Page": 2,
        "Paragraph": 3,
        "ImprovedText": "..."
      }},
      ...
    ]
    """

    prompt = PromptTemplate(template=template, input_variables=["recs", "doc"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({
        "recs": "\n".join(recommendations),
        "doc": "\n".join([f"[Page {c['page']}, Para {c['paragraph']}]: {c['content']}" for c in chunks])
    })

    return response["text"]


def generate_improved_clauses(document_text, recommendations, llm):
    template = """
    You are an AI legal drafting assistant.
    The user has uploaded a rental agreement. Based on the following recommendations, 
    generate improved contract text/clauses that can be directly added to the agreement. 

    Recommendations:
    {recs}

    Agreement Text:
    {doc}

    TASK:
    - Rewrite risky or incomplete clauses with safer alternatives.
    - For missing clauses, draft new ones in formal legal language.
    - Keep the text concise, professional, and enforceable under Indian law.

    Return only the improved text that the user can copy-paste into their agreement.
    """

    prompt = PromptTemplate(template=template, input_variables=["recs", "doc"])
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.invoke({
        "recs": "\n".join(recommendations),
        "doc": document_text
    })

    return response["text"]

def analyze_risk(document_text):
    return {
        "Liability": "High Risk",
        "Termination": "Medium Risk",
        "Payment Terms": "Low Risk",
        "Confidentiality": "Medium Risk",
        "IP Rights": "Medium Risk"
    }

def analyze_risk_with_recommendations(document_text, llm):
    template = """
    You are an AI legal contract risk analyst and advisor. 
    Read the following rent/lease agreement carefully.

    1. Classify each of these clauses into a risk level: High Risk, Medium Risk, or Low Risk.
       - Liability
       - Termination
       - Payment Terms
       - Confidentiality
       - IP Rights

    2. For each clause that is Medium or High Risk → suggest **specific improvements** or contract modifications that would reduce risk and make it legally safer.

    3. If any important clauses are missing (e.g., Dispute Resolution, Rent Increase Rules, Notice Period), identify them and suggest what should be added.

    Agreement Text:
    {doc}

    Return your response in JSON format like this:
    {{
      "Risk Summary": {{
        "Liability": "High Risk",
        "Termination": "Medium Risk",
        "Payment Terms": "Low Risk",
        "Confidentiality": "Medium Risk",
        "IP Rights": "Low Risk"
      }},
      "Recommendations": [
        "Liability clause is too one-sided. Suggest adding mutual indemnity.",
        "Add a clear dispute resolution mechanism (e.g., arbitration in Delhi).",
        "Specify notice period for termination (e.g., 30 days)."
      ]
    }}
    """

    prompt = PromptTemplate(template=template, input_variables=["doc"])
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.invoke({"doc": document_text})

    import json
    try:
        result = json.loads(response["text"])
    except:
        result = {
            "Risk Summary": {
                "Liability": "Medium Risk",
                "Termination": "Medium Risk",
                "Payment Terms": "Medium Risk",
                "Confidentiality": "Medium Risk",
                "IP Rights": "Medium Risk"
            },
            "Recommendations": [
                "Ensure rent increase follows state Rent Control Act.",
                "Add arbitration clause for dispute resolution.",
                "Define notice period clearly."
            ]
        }
    return result

def risk_color(risk_level):
    if "High" in risk_level:
        return "background-color:#f8d7da; color:#721c24; padding:4px; border-radius:5px;"
    elif "Medium" in risk_level:
        return "background-color:#fff3cd; color:#856404; padding:4px; border-radius:5px;"
    elif "Low" in risk_level:
        return "background-color:#d4edda; color:#155724; padding:4px; border-radius:5px;"
    return ""

def main():
    st.set_page_config(page_title="LawyerLens", page_icon="⚖️")
    st.header("LegalAI: AI for legal productivity ⚖️")

    embeddings = load_embeddings()
    llm = load_llm()
    pc = Pinecone(api_key="pcsk_3aziQH_FhYmdBpXgFTL6borDTmnUcMFSnsXd55CXq5YdYmge17qF14iQP7ZScvAXxnT261")
    index = pc.Index(INDEX_NAME)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever_uploaded" not in st.session_state:
        st.session_state.retriever_uploaded = None

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar upload
    with st.sidebar:
        st.subheader("Upload Legal Documents")
        pdf_docs = st.file_uploader("Upload agreements (PDF)", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    if not chunks:
                        st.warning("No valid text chunks found in the uploaded PDF.")
                        return
                    st.success("Uploaded documents processed ✅")
                    vector_store_uploaded = FAISS.from_texts(texts=chunks, embedding=embeddings)
                    st.session_state.retriever_uploaded = vector_store_uploaded.as_retriever()
                    st.subheader("📊 Risk Summary & Suggestions")
                    result = analyze_risk_with_recommendations(raw_text, llm)
                    risks = analyze_risk(raw_text)  
                    for clause, risk in risks.items():
                        st.markdown(
                            f"""
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span><b>{clause}</b></span>
                                <span style="{risk_color(risk)}">{risk}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    # Display Recommendations
                    st.markdown("### 📝 Recommended Changes")
                    for rec in result["Recommendations"]:
                        st.markdown(f"- {rec}")    
            else:
                st.warning("Please upload at least one PDF file.") 

    # Main chat
    if user_prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            if "improvement" in user_prompt.lower() and st.session_state.last_risk_result:
                with st.spinner("Drafting improved clauses..."):
                    improved_text = generate_improved_clauses(
                    raw_text, 
                    st.session_state.last_risk_result["Recommendations"], 
                    llm
                )
                st.markdown("### ✨ Improved Clauses")
                st.markdown(improved_text)
                st.session_state.messages.append({"role": "assistant", "content": improved_text})

            if st.session_state.retriever_uploaded is None:
                st.warning("Please upload and process a document first.")
                st.stop()

            with st.spinner("Analyzing..."):
                # 1. Connect to Indian Law Knowledge Base
                vector_store_law = PineconeVectorStore.from_existing_index(
                    index_name=INDEX_NAME,
                    embedding=embeddings
                )
                retriever_indian_law = vector_store_law.as_retriever(search_kwargs={'k': 5})

                # 2. Retrieve from both sources
                docs_from_law = retriever_indian_law.get_relevant_documents(user_prompt)
                docs_from_upload = st.session_state.retriever_uploaded.get_relevant_documents(user_prompt)

                context_law = "\n".join([doc.page_content for doc in docs_from_law])
                context_upload = "\n".join([doc.page_content for doc in docs_from_upload])

                risk_summary = {
                    "Liability": "High Risk",
                    "Termination": "Medium Risk",
                    "Payment Terms": "Low Risk",
                    "Confidentiality": "Medium Risk",
                    "IP Rights": "Medium Risk",
                }

                # Show risk summary
                st.subheader("📊 Risk Summary")
                for clause, risk in risk_summary.items():
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                            <span><b>{clause}</b></span>
                            <span style="{risk_color(risk)}">{risk}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # 4. Prompt template
                template = """
                ROLE: You are an AI legal assistant with the expertise of a seasoned Indian High Court lawyer.

                TASK: Compare the 'Uploaded Document Context' with the 'Indian Law Context' and provide a precise legal analysis.

                - If there is a conflict → explain and state the correct law.
                - If it aligns → confirm compliance.
                - If law is silent → answer from uploaded document only.

                ---
                Indian Law Context:
                {context_indian_law}

                Uploaded Document Context:
                {context_uploaded_doc}

                User's Question:
                {question}
                ---
                Legal Analysis:
                """

                prompt_template = PromptTemplate(
                    template=template,
                    input_variables=["context_uploaded_doc", "context_indian_law", "question"]
                )

                chain = LLMChain(llm=llm, prompt=prompt_template)
                response = chain.invoke({
                    "context_uploaded_doc": context_upload,
                    "context_indian_law": context_law,
                    "question": user_prompt
                })

                response_text = response['text']
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == '__main__':
    main()
