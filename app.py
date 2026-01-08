import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from duckduckgo_search import DDGS
import json

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="TennisBet AI", page_icon="🎾", layout="centered")

# --- CSS CUSTOM PER I GRAFICI ---
# Questo rende le barre di progresso più "spesse" e colorate
st.markdown("""
<style>
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #4CAF50, #8BC34A);
}
.big-font {
    font-size: 24px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎾 TennisBet AI")
    st.caption("Agente strategico per previsioni sportive")
    api_key = st.text_input("Groq API Key:", type="password")
    st.info("Motore: Llama 3.3-70B (Versatile)")

# --- FUNZIONI ---
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

def search_web(query):
    try:
        results = DDGS().text(query, max_results=3, backend="html")
        return str(results) if results else "Nessun dato trovato."
    except Exception as e:
        return f"Errore ricerca: {e}"

def run_tennis_agent(query, api_key):
    llm = ChatGroq(temperature=0.2, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    current_date = get_current_date()

    # 1. PLANNING
    with st.spinner("🧠 Analisi quote e statistiche..."):
        planner_prompt = ChatPromptTemplate.from_template(
            """Oggi è il {date}. Task: Previsione scommessa per "{query}".
            Genera 2 query di ricerca per trovare:
            1. Ultimi 5 match di entrambi i giocatori.
            2. Testa a testa (H2H) e performance sulla superficie specifica.
            Restituisci SOLO le query separate da virgola.
            """
        )
        search_queries = (planner_prompt | llm | StrOutputParser()).invoke({"date": current_date, "query": query}).split(',')

    # 2. RETRIEVAL
    context_data = ""
    with st.status("🕵️‍♂️ Scouting dei dati...", expanded=False):
        for q in search_queries:
            q = q.strip()
            st.write(f"🔍 Analizzo: **{q}**")
            res = search_web(q)
            context_data += f"\nQ: {q}\nA: {res}\n"

    # 3. GENERATION (JSON MODE)
    st.info("🤖 Calcolo probabilità vittoria...")
    
    final_prompt = ChatPromptTemplate.from_template(
        """Sei un esperto Betting Tipster di tennis. Data: {date}.
        
        DATI LIVE: {context}
        MATCH: {query}
        
        COMPITO:
        1. Identifica i due giocatori.
        2. Assegna una % di probabilità di vittoria (Somma deve fare 100).
        3. Spiega su chi scommettere e perché (analizzando forma, superficie, H2H).
        
        FORMATO JSON OBBLIGATORIO:
        {{
            "p1_name": "Nome Giocatore 1",
            "p1_score": 65,
            "p2_name": "Nome Giocatore 2",
            "p2_score": 35,
            "betting_advice": "Il consiglio è puntare su X perché..."
        }}
        
        Rispondi SOLO con il JSON.
        """
    )
    
    chain = final_prompt | llm | StrOutputParser()
    return chain.invoke({"date": current_date, "context": context_data, "query": query})

# --- UI PRINCIPALE ---
st.title("🏆 AI Match Predictor")
st.write("Inserisci il match per ottenere le probabilità percentuali.")

query = st.text_input("Match:", placeholder="Es: Musetti vs Wong Hong Kong")

if st.button("CALCOLA PREVISIONE") and api_key and query:
    try:
        raw_response = run_tennis_agent(query, api_key)
        
        # Pulizia JSON (a volte il modello mette ```json all'inizio)
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        
        # Estrazione dati
        p1 = data.get("p1_name")
        s1 = data.get("p1_score")
        p2 = data.get("p2_name")
        s2 = data.get("p2_score")
        advice = data.get("betting_advice")

        # Visualizzazione Grafica
        st.divider()
        
        col1, col2, col3 = st.columns([4, 1, 4])
        
        # Logica colori: Verde per chi ha > 50%, Rosso per chi ha < 50%
        color1 = "#4CAF50" if s1 >= s2 else "#FF5252"
        color2 = "#4CAF50" if s2 > s1 else "#FF5252"

        with col1:
            st.markdown(f"<div style='text-align: center; color: {color1}; font-size: 30px; font-weight: bold'>{s1}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-weight: bold'>{p1}</div>", unsafe_allow_html=True)
            st.progress(s1 / 100)

        with col2:
            st.markdown("<div style='text-align: center; font-size: 20px; padding-top: 20px'>VS</div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div style='text-align: center; color: {color2}; font-size: 30px; font-weight: bold'>{s2}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-weight: bold'>{p2}</div>", unsafe_allow_html=True)
            # Trucco per la barra rossa invertita (visivo) o standard
            st.progress(s2 / 100)

        st.divider()
        st.subheader("💡 L'Analisi dell'Esperto")
        st.info(advice)

    except Exception as e:
        st.error(f"Errore nell'elaborazione: {e}")
        st.write("Risposta grezza modello:", raw_response if 'raw_response' in locals() else "N/A")
