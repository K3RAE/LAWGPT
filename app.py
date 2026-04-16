import streamlit as st
import os
import json
import joblib
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import re

st.set_page_config(
    page_title="LawGPT — Legal Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #07090f;
    color: #E2E8F0;
}

header {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    max-width: 1100px;
    padding-top: 32px;
    padding-bottom: 60px;
    padding-left: 40px;
    padding-right: 40px;
}

section[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif;
}

input, textarea {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #E2E8F0 !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}

input:focus, textarea:focus {
    border: 1px solid #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    outline: none !important;
}

.stButton > button {
    background: #2563EB;
    border-radius: 10px;
    height: 46px;
    font-weight: 500;
    font-size: 15px;
    border: none;
    color: white;
    padding: 0 28px;
    letter-spacing: 0.01em;
    transition: background 0.2s ease;
}

.stButton > button:hover {
    background: #1d4ed8;
}

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    color: #CBD5E1 !important;
}

.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

.page-header {
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

.page-title {
    font-size: 24px;
    font-weight: 600;
    color: #F1F5F9;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}

.page-subtitle {
    font-size: 14px;
    color: #64748B;
    margin: 0;
    font-weight: 400;
}

.info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
}

.info-card-label {
    font-size: 11px;
    font-weight: 500;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.info-card-value {
    font-size: 22px;
    font-weight: 600;
    margin: 0;
    line-height: 1.2;
}

.info-card-sub {
    font-size: 12px;
    color: #475569;
    margin-top: 4px;
}

.verdict-allowed {
    display: inline-block;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    color: #4ade80;
    border-radius: 8px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 500;
}

.verdict-dismissed {
    display: inline-block;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.25);
    color: #f87171;
    border-radius: 8px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 500;
}

.verdict-partial {
    display: inline-block;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    color: #fbbf24;
    border-radius: 8px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 500;
}

.precedent-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #3B82F6;
    border-radius: 0 10px 10px 0;
    padding: 16px 18px;
    margin-bottom: 14px;
    font-size: 14px;
    line-height: 1.7;
    color: #94A3B8;
}

.precedent-header {
    font-size: 12px;
    font-weight: 500;
    color: #3B82F6;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.section-tag {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    color: #a5b4fc;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px 4px 3px 0;
}

.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 24px 0;
}

.sidebar-brand {
    font-size: 18px;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}

.sidebar-tagline {
    font-size: 12px;
    color: #475569;
    margin-bottom: 0;
}

.sidebar-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #94A3B8;
    padding: 10px 0;
}

.status-dot-green {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    flex-shrink: 0;
}

.status-dot-red {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    flex-shrink: 0;
}

.insight-body {
    font-size: 14px;
    line-height: 1.75;
    color: #94A3B8;
    padding: 4px 0;
}

.disclaimer {
    font-size: 12px;
    color: #334155;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.05);
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# BACKEND LOAD
# ---------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "phi3:latest"

@st.cache_resource
def load_models():
    embedder = SentenceTransformer(
        "nlpaueb/legal-bert-base-uncased",
        local_files_only=True
    )
    classifier = joblib.load(os.path.join(BASE_DIR, "models", "legal_outcome_classifier.joblib"))
    return embedder, classifier

embedder, classifier = load_models()

@st.cache_resource
def load_vector_store():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(name="legal_cases")

collection = load_vector_store()

def get_embedding(text: str):
    return embedder.encode([text]).tolist()

def mmr_deduplicate(docs, metadatas, n=5):
    seen_ids = set()
    filtered = []
    for doc, meta in zip(docs, metadatas):
        cid = meta.get("case_id", doc[:40])
        if cid not in seen_ids:
            seen_ids.add(cid)
            filtered.append(doc)
        if len(filtered) == n:
            break
    return filtered

def generate_answer_stream(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 150,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            },
            stream=True,
            timeout=120
        )
        for line in response.iter_lines():
            if line:
                try:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    except Exception as e:
        yield f"\n\n[Connection error: {str(e)}]"

def run_classifier(text: str):
    try:
        emb  = embedder.encode([text])
        pred = classifier.predict(emb)[0]
        conf = int(max(classifier.predict_proba(emb)[0]) * 100)
        return pred, conf
    except Exception:
        return "Unknown", 0

def build_legal_prompt(query: str, context: str) -> str:
    return f"""You are an expert Indian legal research assistant.
Use ONLY the context below. Do not invent cases, statutes, or citations.
If the context does not answer the question, say so clearly.

CONTEXT:
{context}

QUESTION: {query}

Respond in this exact format:
LEGAL PRINCIPLE: [one-sentence rule of law applicable here]
REASONING: [apply the principle to the facts, 2-3 sentences]
RELEVANT SECTIONS: [list any statutes or articles cited in the context]
ANSWER: [direct, concise answer to the question]
"""

def relevance_label(score: float) -> str:
    if score >= 85:
        return "Highly Relevant"
    elif score >= 70:
        return "Relevant"
    elif score >= 55:
        return "Moderately Relevant"
    else:
        return "Loosely Related"

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:
    st.markdown("<div class='sidebar-brand'>⚖ LawGPT</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-tagline'>Indian Legal Intelligence Platform</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:18px 0;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Legal Research", "Precedent Search", "Case Assessment"],
        key="main_navigation",
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:18px 0;'>", unsafe_allow_html=True)

    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        st.markdown(
            "<div class='sidebar-status'><div class='status-dot-green'></div>AI Engine &nbsp;·&nbsp; Online</div>",
            unsafe_allow_html=True
        )
    except Exception:
        st.markdown(
            "<div class='sidebar-status'><div class='status-dot-red'></div>AI Engine &nbsp;·&nbsp; Offline</div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px;color:#475569;'>LawGPT v2.0 &nbsp;·&nbsp; © 2026<br>For informational use only.</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# LEGAL RESEARCH
# ---------------------------------------------------

if page == "Legal Research":

    st.markdown("""
        <div class='page-header'>
            <div class='page-title'>Legal Research</div>
            <div class='page-subtitle'>Ask any question on Indian law — judgments, statutes, or procedural matters</div>
        </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Your legal question",
        placeholder="e.g. When can a High Court exercise inherent powers under Section 482 CrPC?",
        label_visibility="collapsed"
    )

    analyze = st.button("Get Legal Analysis")

    if analyze and query:

        with st.spinner("Searching case law and statutes..."):
            query_embedding = embedder.encode([query]).tolist()

            results = collection.query(
                query_embeddings=query_embedding,
                n_results=10,
                include=["documents", "metadatas", "distances"]
            )

            raw_docs      = results.get("documents",  [[]])[0]
            raw_metas     = results.get("metadatas",  [[]])[0]
            raw_distances = results.get("distances",  [[]])[0]

            deduped_docs = mmr_deduplicate(raw_docs, raw_metas, n=5)
            context      = "\n\n".join(deduped_docs[:3])[:2500] if deduped_docs else ""

        prompt = build_legal_prompt(query, context)

        st.markdown("#### Legal Analysis")
        answer_placeholder = st.empty()
        full_answer = ""
        for token in generate_answer_stream(prompt):
            full_answer += token
            answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)

        if context:
            outcome, confidence = run_classifier(context)
        else:
            outcome, confidence = "Unknown", 0

        pred_colour = {"Allowed": "#22c55e", "Dismissed": "#ef4444"}.get(outcome, "#f59e0b")

        st.markdown(
            f"<div class='info-card'>"
            f"<div class='info-card-label'>Likely Outcome</div>"
            f"<div class='info-card-value' style='color:{pred_colour}'>{outcome}</div>"
            f"<div class='info-card-sub'>Based on comparable judgments in the database</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        with st.expander("Supporting Judgments"):
            if deduped_docs:
                for i, doc in enumerate(deduped_docs[:3]):
                    try:
                        orig_idx  = raw_docs.index(doc)
                        similarity = round((1 - raw_distances[orig_idx]) * 100, 1)
                        rel_label  = relevance_label(similarity)
                    except ValueError:
                        rel_label  = "Related"
                    st.markdown(
                        f"<div class='precedent-card'>"
                        f"<div class='precedent-header'>Judgment {i+1} &nbsp;·&nbsp; {rel_label}</div>"
                        f"{doc[:800]}..."
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<div class='insight-body'>No matching judgments found in the database.</div>", unsafe_allow_html=True)

        with st.expander("Laws & Sections Cited"):
            section_pattern = r"\b(?:Section|Article|Order|Rule|Schedule)\s+\d+[\w\-]*(?:\s+\w+)?"
            sections = re.findall(section_pattern, context, flags=re.IGNORECASE)
            if sections:
                tags_html = "".join(f"<span class='section-tag'>{sec}</span>" for sec in sorted(set(sections)))
                st.markdown(f"<div style='padding:8px 0;'>{tags_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='insight-body'>No specific statutory provisions identified.</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='disclaimer'>This analysis is generated from publicly available Indian court judgments "
            "and is intended for informational and research purposes only. It does not constitute legal advice. "
            "Please consult a qualified advocate for guidance on your specific matter.</div>",
            unsafe_allow_html=True
        )

# ---------------------------------------------------
# PRECEDENT SEARCH
# ---------------------------------------------------

elif page == "Precedent Search":

    st.markdown("""
        <div class='page-header'>
            <div class='page-title'>Precedent Search</div>
            <div class='page-subtitle'>Find judgments relevant to your case facts or legal issue</div>
        </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Describe your case or legal issue",
        placeholder="e.g. Accused charged under Section 302 IPC, circumstantial evidence, no eyewitness...",
        label_visibility="collapsed"
    )

    search = st.button("Search Judgments")

    if search and query:

        with st.spinner("Searching the judgment database..."):
            query_embedding = embedder.encode([query]).tolist()

            results = collection.query(
                query_embeddings=query_embedding,
                n_results=10,
                include=["documents", "metadatas", "distances"]
            )

            raw_docs      = results.get("documents",  [[]])[0]
            raw_metas     = results.get("metadatas",  [[]])[0]
            raw_distances = results.get("distances",  [[]])[0]

            deduped_docs = mmr_deduplicate(raw_docs, raw_metas, n=5)

        if deduped_docs:
            st.markdown(
                f"<div style='font-size:13px;color:#475569;margin-bottom:20px;'>"
                f"Found {len(deduped_docs)} relevant judgment(s)</div>",
                unsafe_allow_html=True
            )
            for i, doc in enumerate(deduped_docs):
                try:
                    orig_idx   = raw_docs.index(doc)
                    similarity = round((1 - raw_distances[orig_idx]) * 100, 1)
                    meta       = raw_metas[orig_idx]
                    case_id    = meta.get("case_id", "—")
                    rel_label  = relevance_label(similarity)
                except (ValueError, IndexError):
                    rel_label = "Related"
                    case_id   = "—"

                with st.expander(f"Judgment {i+1}  ·  {rel_label}  ·  Ref: {case_id}"):
                    st.markdown(
                        f"<div class='insight-body'>{doc}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                "<div class='insight-body'>No matching judgments found. Try rephrasing with more specific facts or legal terms.</div>",
                unsafe_allow_html=True
            )

# ---------------------------------------------------
# CASE ASSESSMENT
# ---------------------------------------------------

elif page == "Case Assessment":

    st.markdown("""
        <div class='page-header'>
            <div class='page-title'>Case Assessment</div>
            <div class='page-subtitle'>Paste your case facts for a preliminary assessment of likely outcome and risk</div>
        </div>
    """, unsafe_allow_html=True)

    case_text = st.text_area(
        "Case facts",
        height=220,
        placeholder="Describe the facts of the case, charges, parties involved, and relief sought...",
        label_visibility="collapsed"
    )

    predict = st.button("Assess Case")

    if predict and case_text:

        with st.spinner("Reviewing case facts..."):

            prediction, confidence = run_classifier(case_text)
            case_embedding = embedder.encode([case_text[:500]]).tolist()

            sim_results = collection.query(
                query_embeddings=case_embedding,
                n_results=6,
                include=["documents", "metadatas", "distances"]
            )
            sim_docs    = sim_results.get("documents", [[]])[0]
            sim_metas   = sim_results.get("metadatas", [[]])[0]
            sim_deduped = mmr_deduplicate(sim_docs, sim_metas, n=3)

        pred_colour = {"Allowed": "#22c55e", "Dismissed": "#ef4444"}.get(prediction, "#f59e0b")

        verdict_class = {
            "Allowed": "verdict-allowed",
            "Dismissed": "verdict-dismissed"
        }.get(prediction, "verdict-partial")

        st.markdown(f"<span class='{verdict_class}'>{prediction}</span>", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"<div class='info-card'>"
                f"<div class='info-card-label'>Expected Verdict</div>"
                f"<div class='info-card-value' style='color:{pred_colour}'>{prediction}</div>"
                f"<div class='info-card-sub'>Based on pattern analysis of similar matters</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col2:
            if prediction == "Allowed":
                action = "File with strong documentary support and a clear prayer for relief."
            elif prediction == "Dismissed":
                action = "Review procedural compliance, limitation period, and jurisdictional basis before filing."
            else:
                action = "Strengthen the factual foundation and clarify the relief sought."

            st.markdown(
                f"<div class='info-card'>"
                f"<div class='info-card-label'>What to Do Next</div>"
                f"<div style='font-size:14px;color:#94A3B8;line-height:1.7;margin-top:4px;'>{action}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        if prediction == "Allowed":
            insight = "Courts have consistently granted relief in comparable matters where the petitioner demonstrated procedural compliance, substantive legal merit, and clear documentary evidence. The strength of the prayer for relief and the absence of delay were decisive factors."
        elif prediction == "Dismissed":
            insight = "In similar matters, courts have declined relief primarily on grounds of procedural defects, delay in approaching the court, lack of maintainability, or absence of a legally cognisable cause of action. Jurisdictional objections were also frequently upheld."
        else:
            insight = "Comparable cases resulted in partial relief, with courts exercising discretion based on the specific facts and equities of each matter. The quantum of relief granted varied significantly depending on the strength of individual factual elements."

        with st.expander("How Courts Have Ruled", expanded=True):
            st.markdown(f"<div class='insight-body'>{insight}</div>", unsafe_allow_html=True)

        with st.expander("Related Cases"):
            if sim_deduped:
                for i, doc in enumerate(sim_deduped):
                    st.markdown(
                        f"<div class='precedent-card'>"
                        f"<div class='precedent-header'>Related Case {i+1}</div>"
                        f"{doc[:600]}..."
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<div class='insight-body'>No related cases found.</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='disclaimer'>This assessment is generated algorithmically from historical judgment data "
            "and is provided for preliminary research purposes only. It does not constitute legal advice or a "
            "prediction of outcome in any specific case. Consult a qualified advocate before taking any legal action.</div>",
            unsafe_allow_html=True
        )