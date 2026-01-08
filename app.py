import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from duckduckgo_search import DDGS
import json
import time

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="TennisBet AI", page_icon="🎾", layout="wide")

st.markdown("""
<style>
/* VS Style */
.stProgress > div > div > div > div { background-image: linear-gradient(to right, #4CAF50, #8BC34A); }
/* Betting Style */
.bet-box { background-color: #007bff; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
.bet-risk { background-color: #ff9800; }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Ciao! Sono il tuo Tennis Advisor. Preferisci giocate Audaci (rischio alto) o Prudenti (sicure)?")]
if "scout_data" not in st.session_state: st.session_state.scout_data = None
if "single_data" not in st.session_state: st.session_state.single_data = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎾 TennisBet AI")
    api_key = st.text_input("Groq API Key:", type="password")
    st.divider()
    manual_profile = st.selectbox("Profilo:", ["Imposta da Chat", "🛡️ PRUDENTE", "🔥 AUDACE"])

# --- FUNZIONI ---
def search_web(query):
    try:
        results = DDGS().text(query, max_results=5, backend="html")
        return str(results) if results else ""
    except: return ""

def extract_json(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])
    except: return None

def run_advisor(mode, user_input, profile, key):
    llm = ChatGroq(temperature=0.2, groq_api_key=key, model_name="llama-3.3-70b-versatile")
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 1. ANALISI SINGOLA
    if mode == "single":
        with st.spinner(f"🔮 Analizzo {user_input}..."):
            web_data = search_web(f"{user_input} tennis match prediction H2H")
            
        prompt = """
        Sei un Esperto Tennis. Oggi: {date}.
        Richiesta: "{query}". Dati: {context}.
        
        Prevedi il risultato futuro.
        Rispondi SOLO JSON:
        {{
            "p1_name": "Nome1", "p1_score": 65,
            "p2_name": "Nome2", "p2_score": 35,
            "reason": "Motivazione tecnica..."
        }}
        """
        chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
        return chain.invoke({"date": today, "query": user_input, "context": web_data})
        
    # 2. LISTA SCOMMESSE (FIX TENNIS ONLY)
    else:
        with st.spinner("📡 Scarico quote TENNIS odierne..."):
            # Aggiunto "ATP WTA" per forzare il tennis nella ricerca
            web_data = search_web(f"ATP WTA Tennis matches schedule {today} order of play")
            
        # PROMPT BLINDATO CONTRO IL CALCIO
        prompt = """
        Sei un Bookmaker specializzato ESCLUSIVAMENTE in TENNIS (ATP, WTA, Challenger).
        Oggi: {date}. Profilo: {profile}.
        DATI CALENDARIO: {context}
        
        Compito:
        1. Estrai 6-8 match reali di oggi/domani.
        2. VIETATO INSERIRE CALCIO (Premier League, Serie A, ecc.). Se leggi "Manchester", "Liverpool", "Inter", SCARTALI.
        3. Voglio SOLO tennisti (es. Sinner, Draper, Berrettini, Sabalenka, ecc.).
        4. Calcola quote e percentuali.
        
        Rispondi SOLO JSON:
        {{
            "matches": [
                {{
                    "p1": "Sinner", "p2": "Djokovic",
                    "p1_perc": 45, "p2_perc": 55,
                    "bet_on": "2",
                    "odd_value": 1.85,
                    "reason": "Match equilibrato..."
                }}
            ]
        }}
        """
        chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
        return chain.invoke({"date": today, "profile": profile, "context": web_data})

# --- UI ---
st.title("🏆 TennisBet AI")
tab1, tab2 = st.tabs(["🔮 Previsione Match", "📋 Scommesse del Giorno"])

# TAB 1
with tab1:
    col_in, col_btn = st.columns([4, 1])
    match_input = col_in.text_input("Inserisci Match:", placeholder="Es: Sinner vs Alcaraz 10 Gennaio")
    
    if col_btn.button("ANALIZZA") and api_key and match_input:
        res = run_advisor("single", match_input, manual_profile, api_key)
        st.session_state.single_data = extract_json(res)

    if st.session_state.single_data and "p1_score" in st.session_state.single_data:
        d = st.session_state.single_data
        st.divider()
        c1, c2, c3 = st.columns([4, 1, 4])
        
        col_p1 = "#4CAF50" if d['p1_score'] >= d['p2_score'] else "#FF5252"
        col_p2 = "#4CAF50" if d['p2_score'] > d['p1_score'] else "#FF5252"
        
        with c1:
            st.markdown(f"<h1 style='text-align:center; color:{col_p1}'>{d['p1_score']}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center'>{d['p1_name']}</h3>", unsafe_allow_html=True)
            st.progress(d['p1_score']/100)
        with c2: st.markdown("<h2 style='text-align:center; padding-top:20px'>VS</h2>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<h1 style='text-align:center; color:{col_p2}'>{d['p2_score']}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center'>{d['p2_name']}</h3>", unsafe_allow_html=True)
            st.progress(d['p2_score']/100)
        st.info(d.get('reason'))

# TAB 2
with tab2:
    col_h, col_r = st.columns([5,1])
    col_h.write("Le migliori occasioni per oggi.")
    if col_r.button("🔄 Aggiorna"):
        st.session_state.scout_data = None
        st.rerun()

    if st.session_state.scout_data is None and api_key:
        prof = manual_profile if manual_profile != "Imposta da Chat" else "BILANCIATO"
        res = run_advisor("list", "scout", prof, api_key)
        st.session_state.scout_data = extract_json(res)

    if st.session_state.scout_data and "matches" in st.session_state.scout_data:
        h1, h2, h3 = st.columns([3, 4, 2])
        h1.caption("MATCH"); h2.caption("PROBABILITÀ"); h3.caption("BET")
        st.divider()
        
        for m in st.session_state.scout_data['matches']:
            r1, r2, r3 = st.columns([3, 4, 2])
            with r1:
                st.write(f"**{m['p1']}** vs **{m['p2']}**")
                st.caption(m.get('reason', '')[:50]+"...")
            with r2:
                st.progress(m['p1_perc']/100)
                st.progress(m['p2_perc']/100)
            with r3:
                css = "bet-risk" if m['odd_value'] > 2.0 else "bet-box"
                st.markdown(f"<div class='{css}' style='text-align:center; border-radius:5px; padding:5px; color:white;'><div>Punta {m['bet_on']}</div><div style='font-size:1.2em'>{m['odd_value']}</div></div>", unsafe_allow_html=True)
            st.divider()

# CHAT
st.divider()
st.subheader("💬 Chat Esperto")
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

if u := st.chat_input("Scrivi qui..."):
    if not api_key: st.error("Serve Key")
    else:
        st.session_state.chat_history.append(HumanMessage(content=u))
        st.chat_message("user").write(u)
        llm = ChatGroq(temperature=0.7, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
        ctx = json.dumps(st.session_state.scout_data) if st.session_state.scout_data else ""
        chain = ChatPromptTemplate.from_template("Ctx:{ctx}. User:{u}. Rispondi.") | llm | StrOutputParser()
        ans = chain.invoke({"ctx":ctx, "u":u})
        st.session_state.chat_history.append(AIMessage(content=ans))
        st.chat_message("assistant").write(ans)
