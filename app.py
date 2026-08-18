import json
import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

st.set_page_config(page_title="RAG Flashcards", page_icon="📚", layout="wide")
st.title("📚 RAG Flashcard Generator")

# 1. Initialize Vector Model and Database
@st.cache_resource
def load_ai_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    db_client = chromadb.PersistentClient(path="./chroma_db")
    return embed_model, db_client

model, chroma_client = load_ai_models()
collection = chroma_client.get_or_create_collection(name="flashcard_notes")

# 2. Session State Initialization
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

# 3. Sidebar Configuration & Database Management
st.sidebar.header("Settings")
st.sidebar.info("Generation is powered by Gemini API (Free Tier).")

st.sidebar.divider()
st.sidebar.subheader("Database Management")

if st.sidebar.button("🗑️ Reset Database & Clear Deck", type="secondary"):
    try:
        chroma_client.delete_collection(name="flashcard_notes")
    except Exception:
        pass
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
        
        if not all_text.strip():
            st.error("⚠️ No text could be extracted from this PDF. Please check the file.")
            st.stop()
            
        words = all_text.split()
        chunks = [" ".join(words[i : i + 150]) for i in range(0, len(words), 120)]
        
        current_count = collection.count()
        ids = [f"chunk_{current_count + i}" for i in range(len(chunks))]
        
        embeddings = model.encode(chunks).tolist()
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        
        st.success(f"Successfully processed {len(chunks)} chunks! (Total chunks in DB: {collection.count()})")

# 5. Flashcard Generation
st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("2. What specific concept do you want to study?", placeholder="e.g., Lucknow Pact or Home Rule League")
with col2:
    num_cards = st.slider("Number of cards", min_value=1, max_value=10, value=3)

if topic and st.button("Generate Flashcards"):
    if collection.count() == 0:
        st.warning("Your database is empty. Please upload and process a PDF first.")
    else:
        with st.spinner(f"Searching vectors and generating {num_cards} flashcards..."):
            topic_vector = model.encode([topic]).tolist()
            query_k = min(collection.count(), 5)
            results = collection.query(query_embeddings=topic_vector, n_results=query_k)
            context = "\n\n--- Next Chunk ---\n\n".join(results["documents"][0])
            
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            
            prompt = f"""You are an expert tutor. Based ONLY on the following context, create exactly {num_cards} distinct flashcards about '{topic}'.

Context:
{context}

Return a valid JSON array of objects. Each object must have a "question" field and an "answer" field.
Example:
[
  {{"question": "When did Jinnah join the Muslim League?", "answer": "In 1913, on the condition of loyalty to the larger national cause."}}
]"""
            
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            
            raw_text = response.text.strip()
            
            # Optional debug expander
            with st.expander("🔍 Debug: Inspect Retrieved Context & Raw AI Output"):
                st.write("**Retrieved Context from ChromaDB:**")
                st.text(context)
                st.write("**Raw Response from Gemini:**")
                st.code(raw_text, language="json")

            try:
                parsed_data = json.loads(raw_text)
                
                # If wrapped in a dictionary, extract the inner list
                if isinstance(parsed_data, dict):
                    card_list = next((v for v in parsed_data.values() if isinstance(v, list)), [])
                elif isinstance(parsed_data, list):
                    card_list = parsed_data
                else:
                    card_list = []

                added_count = 0
                for item in card_list:
                    if isinstance(item, dict):
                        # Match any casing or alternate key names
                        q = (
                            item.get("question")
                            or item.get("Question")
                            or item.get("q")
                            or item.get("Q")
                            or item.get("front")
                            or ""
                        )
                        a = (
                            item.get("answer")
                            or item.get("Answer")
                            or item.get("a")
                            or item.get("A")
                            or item.get("back")
                            or ""
                        )
                        
                        if str(q).strip() and str(a).strip():
                            st.session_state.flashcards.append({"q": str(q).strip(), "a": str(a).strip()})
                            added_count += 1
                
                if added_count > 0:
                    st.success(f"Successfully generated {added_count} flashcard(s)!")
                else:
                    st.warning("No matching information was found in the text for that topic. Try a broader topic like 'Lucknow' or 'Bombay'.")
                    
            except json.JSONDecodeError:
                st.error("Failed to parse JSON response. Please try again.")

# 6. Display Deck & Export (Only display non-empty cards)
valid_cards = [c for c in st.session_state.flashcards if c.get("q") and c.get("a")]

if valid_cards:
    st.subheader(f"Your Deck ({len(valid_cards)} cards)")
    
    anki_export_text = ""
    for card in valid_cards:
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
