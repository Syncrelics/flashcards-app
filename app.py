import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(page_title="RAG Flashcards", page_icon="📚", layout="wide")
st.title("📚 RAG Flashcard Generator")

# 1. Initialize Vector Model and Database
@st.cache_resource
def load_ai_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_or_create_collection(name="flashcard_notes")
    return embed_model, db_client, collection

model, chroma_client, collection = load_ai_models()

# 2. Session State Initialization
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

# 3. Sidebar Configuration & Database Management
st.sidebar.header("Settings")
st.sidebar.info("Generation is powered by Gemini API (Free Tier).")

st.sidebar.divider()
st.sidebar.subheader("Database Management")

if st.sidebar.button("🗑️ Reset Database & Clear Deck", type="secondary"):
    chroma_client.delete_collection(name="flashcard_notes")
    collection = chroma_client.get_or_create_collection(name="flashcard_notes")
    st.session_state.flashcards = []
    st.sidebar.success("Database and deck cleared!")
    st.rerun()

# 4. PDF Upload & Processing
uploaded_file = st.file_uploader("1. Upload a PDF to study", type="pdf")

if uploaded_file and st.button("Process Document"):
    with st.spinner("Extracting text and generating embeddings..."):
        reader = PdfReader(uploaded_file)
        all_text = " \n".join([page.extract_text() or "" for page in reader.pages])
        
        words = all_text.split()
        chunks = [" ".join(words[i : i + 150]) for i in range(0, len(words), 120)]
        
        current_count = collection.count()
        ids = [f"chunk_{current_count + i}" for i in range(len(chunks))]
        
        embeddings = model.encode(chunks).tolist()
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        
        st.success(f"Successfully processed {len(chunks)} chunks! (Total chunks in DB: {collection.count()})")

# 5. Flashcard Generation
st.divider()
topic = st.text_input("2. What specific concept do you want a flashcard for?")

if topic and st.button("Generate Flashcard"):
    if collection.count() == 0:
        st.warning("Your database is empty. Please upload and process a PDF first.")
    else:
        with st.spinner("Searching vectors and consulting Gemini..."):
            topic_vector = model.encode([topic]).tolist()
            results = collection.query(query_embeddings=topic_vector, n_results=2)
            context = "\n\n--- Next Chunk ---\n\n".join(results["documents"][0])
            
            # Fetch API key securely from Streamlit Secrets
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            
            prompt = f"""You are an expert tutor. Based ONLY on the following context, create a clear, high-yield flashcard.

Context:
{context}

Format your output exactly as follows:
Q: [Question]
A: [Answer]"""
            
            # Use Gemini's fast, free model
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
            )
            
            raw_text = response.text
            
            if "Q:" in raw_text and "A:" in raw_text:
                q_part = raw_text.split("A:")[0].replace("Q:", "").strip()
                a_part = raw_text.split("A:")[1].strip()
                st.session_state.flashcards.append({"q": q_part, "a": a_part})
                st.success("Flashcard generated and saved!")
            else:
                st.error("Failed to parse the flashcard format. Please try again.")

# 6. Display Deck & Export
if st.session_state.flashcards:
    st.subheader(f"Your Deck ({len(st.session_state.flashcards)} cards)")
    
    anki_export_text = ""
    for card in st.session_state.flashcards:
        st.info(f"Q:\n\nA:")
        clean_q = card['q'].replace('\n', ' ')
        clean_a = card['a'].replace('\n', '<br>')
        anki_export_text += f"{clean_q}\t{clean_a}\n"
    
    st.download_button(
        label="📥 Download for Anki (.txt)",
        data=anki_export_text,
        file_name="rag_flashcards.txt",
        mime="text/plain"
    )