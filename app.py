import streamlit as st
import os
import joblib
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import re

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="LAWGPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# DARK ENTERPRISE UI
# ---------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at 20% 20%, #1e293b, #0f172a 70%);
    color: #E2E8F0;
}

header {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    max-width: 1200px;
    padding-top: 20px;
    padding-bottom: 40px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #020617);
    border-right: 1px solid rgba(255,255,255,0.05);
}

input, textarea {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
}

input:focus, textarea:focus {
    border: 1px solid #3B82F6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.4) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    border-radius: 12px;
    height: 45px;
    font-weight: 600;
    border: none;
    color: white;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 20px;
}

.progress-bar {
    height: 6px;
    background: #1e293b;
    border-radius: 6px;
    overflow: hidden;
    margin-top: 10px;
}

.progress-fill {
    height: 6px;
    background: linear-gradient(90deg, #3B82F6, #60A5FA);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# BACKEND LOAD
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    classifier = joblib.load(os.path.join(BASE_DIR, "models", "legal_outcome_classifier.joblib"))
    return embedder, classifier

embedder, classifier = load_models()

@st.cache_resource
def load_vector_store():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(name="legal_cases")

collection = load_vector_store()

def generate_answer(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 250, "temperature": 0.1}
        },
        timeout=120
    )
    return response.json()["response"]

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:
    st.markdown("## ⚖️ LAWGPT")
    st.markdown("AI-powered Legal Intelligence")
    st.markdown("---")

    page = st.radio(
        "",
        ["Ask Legal Question", "Similar Case Finder", "Outcome Predictor"],
        key="main_navigation"
    )

    st.markdown("---")
    st.markdown("Version 1.0 © 2026 LAWGPT")

# ---------------------------------------------------
# ASK LEGAL QUESTION
# ---------------------------------------------------

if page == "Ask Legal Question":

    st.markdown("<div class='section-title'>🔎 Legal Research Workspace</div>", unsafe_allow_html=True)

    query = st.text_input(
        "",
        placeholder="When can a High Court exercise inherent powers under Section 482 CrPC?"
    )

    analyze = st.button("Analyze Case")

    if analyze and query:

        with st.spinner("Analyzing..."):

            query_embedding = embedder.encode([query]).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=5)

            retrieved_docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            context = "\n\n".join(retrieved_docs[:3])[:2000] if retrieved_docs else ""

            prompt = f"""
You are a legal research assistant.
Answer professionally using provided context only.

LEGAL CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
"""

            answer = generate_answer(prompt)

            if context:
                emb = embedder.encode([context])
                outcome = classifier.predict(emb)[0]
                confidence = int(max(classifier.predict_proba(emb)[0]) * 100)
            else:
                outcome = "Unknown"
                confidence = 0

        # SUMMARY
        with st.expander("Summary", expanded=True):
            st.write(answer)

        # KEY CONSIDERATIONS
        with st.expander("Key Considerations"):
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs[:3]):
                    similarity_score = round((1 - distances[i]) * 100, 2)
                    st.markdown(f"**Case {i+1}**")
                    st.write(doc[:800] + "...")
                    st.markdown("---")
            else:
                st.write("No relevant precedents found.")

        # RELEVANT SECTIONS
        with st.expander("Relevant Sections"):

            sections = re.findall(r"Section\s+\d+\s*\w*", context)

            if sections:
                for sec in list(set(sections)):
                    st.write(f"- {sec}")
            else:
                st.write("No statutory sections identified.")


# ---------------------------------------------------
# SIMILAR CASE FINDER
# ---------------------------------------------------

elif page == "Similar Case Finder":

    st.markdown("<div class='section-title'>📚 Similar Case Finder</div>", unsafe_allow_html=True)

    query = st.text_input(
        "",
        placeholder="Enter case facts or keywords to find similar precedents..."
    )

    search = st.button("Find Similar Cases")

    if search and query:

        with st.spinner("Searching..."):

            query_embedding = embedder.encode([query]).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=5)

            retrieved_docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                similarity_score = round((1 - distances[i]) * 100, 2)

                with st.expander(f"Case {i+1}"):
                    st.write(doc)
        else:
            st.write("No similar cases found.")

# ---------------------------------------------------
# OUTCOME PREDICTOR
# ---------------------------------------------------

elif page == "Outcome Predictor":

    st.markdown("<div class='section-title'>⚖️ Outcome Predictor</div>", unsafe_allow_html=True)

    case_text = st.text_area(
        "",
        height=200,
        placeholder="Paste full case facts here to predict probable outcome..."
    )

    predict = st.button("Predict Outcome")

    if predict and case_text:

        with st.spinner("Analyzing case facts..."):

            embedding = embedder.encode([case_text])
            prediction = classifier.predict(embedding)[0]
            proba = classifier.predict_proba(embedding)[0]
            confidence = int(max(proba) * 100)

        # -------------------------
        # CONFIDENCE INTERPRETATION
        # -------------------------
        if confidence >= 80:
            confidence_label = "High Confidence"
            risk_level = "Low Litigation Risk"
        elif confidence >= 60:
            confidence_label = "Moderate Confidence"
            risk_level = "Moderate Litigation Risk"
        else:
            confidence_label = "Low Confidence"
            risk_level = "Uncertain / Fact-Sensitive Outcome"

        # -------------------------
        # OUTCOME INSIGHT TEXT
        # -------------------------
        if prediction == "Allowed":
            insight = "Courts in similar matters have generally granted relief when procedural compliance and substantive merit were established."
            recommendation = "Ensure documentary evidence and procedural requirements are strongly demonstrated."
        elif prediction == "Dismissed":
            insight = "Similar cases were dismissed primarily due to procedural defects, delay, or lack of maintainability."
            recommendation = "Review procedural grounds, limitation period, and jurisdictional validity."
        else:
            insight = "Comparable matters resulted in partial relief depending on factual nuances and judicial discretion."
            recommendation = "Strengthen weak factual elements and clarify relief sought."

        # -------------------------
        # DISPLAY STRUCTURED OUTPUT
        # -------------------------

        st.markdown("### Predicted Outcome")
        st.write(f"**{prediction}**")

        st.markdown("### Risk Evaluation")
        st.write(risk_level)

        st.markdown("### Judicial Pattern Insight")
        st.write(insight)

        st.markdown("### Strategic Consideration")
        st.write(recommendation)

st.markdown("""
<style>
.disclaimer-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: rgba(15, 23, 42, 0.95);
    border-top: 1px solid rgba(255,255,255,0.08);
    padding: 12px 20px;
    font-size: 13px;
    color: #94A3B8;
    text-align: center;
    z-index: 100;
}
</style>

<div class="disclaimer-bar">
⚠️ All the answers provided are for informational purposes only and do not constitute legal advice.
</div>
""", unsafe_allow_html=True)