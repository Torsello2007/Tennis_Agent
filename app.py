import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from duckduckgo_search import DDGS
import json

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="TennisBet AI", page_icon="🎾", layout="wide")

st.markdown("""
<style>
.stProgress > div > div > div > div { background-image: linear-gradient(to right, #4CAF50, #8BC34A); }
.bet-box { background-color: #007bff; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
.bet-risk { background-color: #ff9800; }
.match-time { 
    font-size: 1.1em; 
    color: #ffcc00; 
    font-weight: bold; 
    font-family: monospace; 
    background-color: #222; 
    padding: 4px 8px; 
    border-radius: 4px; 
    margin-left: 10px; 
}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Ciao! Sono il tuo Tennis Advisor. Per darti i consigli migliori ho bisogno di conoscerti: preferisci giocate PRUDENTI (sicure) o AUDACI (rischiose)?")]
if "scout_data" not in st.session_state: st.session_state.scout_data = None
if "single_data" not in st.session_state: st.session_state.single_data = None

# --- SIDEBAR (PULITA) ---
with st.sidebar:
    st.title("🎾 TennisBet AI")
    api_key = st.text_input("Groq API Key:", type="password")
    st.divider()
    st.info("💡 Usa la chat per definire la tua strategia.")
    manual_profile = "BILANCIATO" 

# --- FUNZIONI ---
def search_web(query):
    try:
        results = DDGS().text(query, max_results=10, backend="html")
        return str(results) if results else ""
    except: return ""

def extract_json(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])
    except: return None

def run_advisor(mode, user_input, profile, key):
    llm = ChatGroq(temperature=0.1, groq_api_key=key, model_name="llama-3.3-70b-versatile")
    today = datetime.now().strftime("%Y-%m-%d")
    
    if mode == "single":
        with st.spinner(f"🔮 Analisi approfondita {user_input}..."):
            web_data = search_web(f"{user_input} tennis match preview prediction head to head")
        prompt = """
        Sei un Analista Tecnico. Oggi: {date}. Richiesta: "{query}". Dati Web: {context}.
        Compito: Previsione dettagliata (minimo 5 righe di motivazione).
        Rispondi SOLO JSON: {{ "p1_name": "A", "p1_score": 60, "p2_name": "B", "p2_score": 40, "reason": "..." }}
        """
        chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
        return chain.invoke({"date": today, "query": user_input, "context": web_data})
        
    else:
        with st.spinner("📡 Scarico il palinsesto reale (ATP/WTA)..."):
            web_data = search_web(f"tennis matches today {today} schedule betting tips odds")
        prompt = """
        Sei un Bookmaker. Oggi: {date}.
        DATI WEB: {context}
        
        COMPITO:
        1. Estrai ALMENO 5 match di tennis che si giocano OGGI o DOMANI.
        2. FORMATO ORARIO: Scrivi SOLO L'ORA (es. "14:30" o "19:00"). NON scrivere "Oggi", "Domani" o la data.
        3. VIETATO CALCIO.
        
        Rispondi SOLO JSON:
        {{
            "matches": [
                {{
                    "p1": "Nome A", "p2": "Nome B",
                    "p1_perc": 60, "p2_perc": 40,
                    "match_time": "14:30", 
                    "bet_on": "1",
                    "odd_value": 1.50,
                    "reason": "Motivo..."
                }}
            ]
        }}
        """
        chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
        return chain.invoke({"date": today, "context": web_data})

# --- UI ---
st.title("🏆 TennisBet AI")
tab1, tab2 = st.tabs(["🔮 Previsione Match", "📋 Scommesse del Giorno"])

# TAB 1
with tab1:
    col_in, col_btn = st.columns([4, 1])
    match_input = col_in.text_input("Inserisci Match:", placeholder="Es: Sinner vs Alcaraz")
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
        st.write("---")
        st.markdown(f"### 📝 Report Tecnico\n{d.get('reason')}")

# TAB 2
with tab2:
    col_h, col_r = st.columns([5,1])
    col_h.write("Le migliori occasioni (ATP/WTA).")
    if col_r.button("🔄 Aggiorna"):
        st.session_state.scout_data = None
        st.rerun()

    if st.session_state.scout_data is None and api_key:
        res = run_advisor("list", "scout", manual_profile, api_key)
        data = extract_json(res)
        if data and data.get("matches"): st.session_state.scout_data = data

    if st.session_state.scout_data and "matches" in st.session_state.scout_data:
        h1, h2, h3 = st.columns([3, 4, 2])
        h1.caption("MATCH / ORA"); h2.caption("PROBABILITÀ"); h3.caption("BET")
        st.divider()
        for m in st.session_state.scout_data['matches']:
            r1, r2, r3 = st.columns([3, 4, 2])
            with r1:
                st.write(f"**{m['p1']}** vs **{m['p2']}**")
                time_str = m.get('match_time', '')
                if time_str: st.markdown(f"<span class='match-time'>🕒 {time_str}</span>", unsafe_allow_html=True)
                st.caption(m.get('reason', '')[:60]+"...")
            with r2:
                st.progress(m['p1_perc']/100)
                st.progress(m['p2_perc']/100)
            with r3:
                css = "bet-risk" if m['odd_value'] > 2.0 else "bet-box"
                st.markdown(f"<div class='{css}' style='text-align:center; border-radius:5px; padding:5px; color:white;'><div>Punta {m['bet_on']}</div><div style='font-size:1.2em'>{m['odd_value']}</div></div>", unsafe_allow_html=True)
            st.divider()

# CHATBOT PROATTIVO
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
        
        matches_list = st.session_state.scout_data['matches'] if st.session_state.scout_data else []
        matches_str = json.dumps(matches_list) if matches_list else "NESSUNA LISTA."
        
        prompt_chat = """
        Sei un Consulente Scommesse molto curioso e professionale.
        LISTA SCOMMESSE DISPONIBILI: {matches}
        MESSAGGIO UTENTE: "{u}"
        
        ISTRUZIONI:
        1. Rispondi all'utente basandoti sulla lista o sulla sua richiesta.
        2. IMPORTANTE: Non chiudere MAI la conversazione. 
        3. OBBLIGATORIO: Termina OGNI tua risposta con una NUOVA DOMANDA per profilare meglio l'utente.
           Esempi di domande da fare:
           - "Ti piacciono le scommesse sui set?"
           - "Preferisci puntare sugli Over/Under?"
           - "Segui solo i Grandi Slam o anche i Challenger?"
           - "Ti fidi di più delle statistiche o dell'intuito?"
        """
        
        chain = ChatPromptTemplate.from_template(prompt_chat) | llm | StrOutputParser()
        ans = chain.invoke({"matches":matches_str, "u":u})
        
        st.session_state.chat_history.append(AIMessage(content=ans))
        st.chat_message("assistant").write(ans)
