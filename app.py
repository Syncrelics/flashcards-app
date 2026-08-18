import json
import time
import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="RAG Flashcards", page_icon="📚", layout="wide")
st.title("📚 RAG Flashcard Generator")

# 1. 3D CSS Injection
st.markdown("""
<style>
/* 3D Flip Card Styling */
.flip-card {
  background-color: transparent;
  width: 100%;
  height: 250px;
  perspective: 1000px;
  margin-bottom: 20px;
}
.flip-card-inner {
  position: relative;
  width: 100%;
  height: 100%;
  text-align: center;
  transition: transform 0.6s;
  transform-style: preserve-3d;
  cursor: pointer;
}
.flip-card:hover .flip-card-inner {
  transform: rotateY(180deg);
}
.flip-card-front, .flip-card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  -webkit-backface-visibility: hidden;
  backface-visibility: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
  box-sizing: border-box;
  overflow-y: auto;
}
/* Hide scrollbar for a cleaner look */
.flip-card-front::-webkit-scrollbar, .flip-card-back::-webkit-scrollbar {
  display: none;
}
.flip-card-front {
  background-color: #2b313e;
  color: #ffffff;
  border: 2px solid #4da6ff;
}
.flip-card-front h3 {
  font-size: 1.2rem;
}
.flip-card-back {
  background-color: #4da6ff;
  color: #111111;
  transform: rotateY(180deg);
}
.flip-card-back p {
  font-size: 1.1rem;
  font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# 2. Strict Pydantic Data Models (Prevents blank outputs)
class Flashcard(BaseModel):
    question: str = Field(description="A specific question based on the text.")
    answer: str = Field(description="A direct, accurate explanation answering the question.")

class FlashcardDeck(BaseModel):
    flashcards: list[Flashcard]

# 3. Initialize Vector Model and Database
@st.cache_resource
def load_ai_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    db_client = chromadb.PersistentClient(path="./chroma_db")
    return embed_model, db_client

model, chroma_client = load_ai_models()
collection = chroma_client.get_or_create_collection(name="flashcard_notes")

# 4. Session State Initialization
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

# 5. Sidebar Configuration
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

# 6. PDF Upload & Processing
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
        
        st.success(f"Successfully processed {len(chunks)} chunks!")

# 7. Flashcard Generation
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
{context}"""
            
            max_retries = 4
            raw_text = None
            
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=FlashcardDeck, # Strictly forces the AI to use our Pydantic structure
                        ),
                    )
                    raw_text = response.text.strip()
                    break
                except Exception as e:
                    if "503" in str(e) and attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        st.toast(f"Google servers are busy. Retrying in {wait_time} seconds...", icon="⏳")
                        time.sleep(wait_time)
                    else:
                        st.error("Google's API is currently overloaded and cannot complete the request. Please try again later.")
                        st.stop()
            
            if raw_text:
                try:
                    parsed_data = json.loads(raw_text)
                    # We are guaranteed to have a 'flashcards' key now
                    card_list = parsed_data.get("flashcards", [])
                    
                    added_count = 0
                    for item in card_list:
                        q = item.get("question", "")
                        a = item.get("answer", "")
                        
                        if str(q).strip() and str(a).strip():
                            st.session_state.flashcards.append({"q": str(q).strip(), "a": str(a).strip()})
                            added_count += 1
                    
                    if added_count > 0:
                        st.success(f"Successfully generated {added_count} flashcard(s)!")
                    else:
                        st.warning("No matching information was found in the text for that topic. Try a broader topic.")
                        
                except json.JSONDecodeError:
                    st.error("Failed to parse the response. Please try again.")

# 8. Display 3D Deck & Export
valid_cards = [c for c in st.session_state.flashcards if c.get("q") and c.get("a")]

if valid_cards:
    st.subheader(f"Your Deck ({len(valid_cards)} cards)")
    
    # Render cards in a grid of 3 columns
    cols = st.columns(3)
    
    anki_export_text = ""
    for idx, card in enumerate(valid_cards):
        # Text formatting for Anki Export
        clean_q = card['q'].replace('\n', ' ')
        clean_a = card['a'].replace('\n', '<br>')
        anki_export_text += f"{clean_q}\t{clean_a}\n"
        
        # HTML escaping for the 3D web UI
        html_q = card['q'].replace('<', '&lt;').replace('>', '&gt;')
        html_a = card['a'].replace('\n', '<br>').replace('<', '&lt;').replace('>', '&gt;')
        
        html_code = f"""
        <div class="flip-card">
          <div class="flip-card-inner">
            <div class="flip-card-front">
              <h3>{html_q}</h3>
            </div>
            <div class="flip-card-back">
              <p>{html_a}</p>
            </div>
          </div>
        </div>
        """
        
        # Distribute the HTML components evenly across the columns
        with cols[idx % 3]:
            st.markdown(html_code, unsafe_allow_html=True)
    
    st.divider()
    st.download_button(
        label="📥 Download for Anki (.txt)",
        data=anki_export_text,
        file_name="rag_flashcards.txt",
        mime="text/plain"
    )
