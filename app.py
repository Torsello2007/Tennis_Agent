import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from duckduckgo_search import DDGS

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="TennisPredictor AI", page_icon="🎾")

# --- TITOLO E SIDEBAR ---
st.title("🎾 TennisPredictor AI Agent")
st.markdown(
    """
    *Agent basato sui principi del Capitolo 6 di AI Engineering.*
    Utilizza **Planning** e **Tool Use** (Web Search) per recuperare il contesto aggiornato.
    """
)

# Sidebar per la chiave API
api_key = st.sidebar.text_input("Inserisci la tua Groq API Key:", type="password")
st.sidebar.info("La chiave non viene salvata, serve solo per questa sessione.")

# --- FUNZIONI DELL'AGENTE ---

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

# Funzione di ricerca diretta
def search_web(query):
    try:
        # Cerca 3 risultati per ogni query
        results = DDGS().text(query, max_results=3)
        return str(results)
    except Exception as e:
        return f"Errore durante la ricerca web: {e}"

def run_tennis_agent(query, api_key):
    # 1. Setup LLM (AGGIORNATO ALL'ULTIMO MODELLO GROQ)
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
    
    current_date = get_current_date()

    # 2. FASE DI PLANNING (Reasoning)
    st.info("🧠 Generazione del piano di ricerca...")
    
    planner_prompt = ChatPromptTemplate.from_template(
        """Sei un AI Planner per un assistente di tennis. Oggi è il {date}.
        L'utente chiede: "{query}".
        
        Identifica 2 query di ricerca specifiche per trovare:
        1. Risultati recenti dei giocatori.
        2. Testa a testa (H2H) e superficie torneo.
        
        Restituisci SOLO le query separate da virgola, senza altro testo.
        """
    )
    
    planner_chain = planner_prompt | llm | StrOutputParser()
    search_queries_text = planner_chain.invoke({"date": current_date, "query": query})
    search_queries = [q.strip() for q in search_queries_text.split(',')]
    
    # 3. FASE DI TOOL EXECUTION (Retrieval)
    context_data = ""
    with st.status("🕵️‍♂️ Ricerca informazioni sul web...", expanded=True):
        for q in search_queries:
            st.write(f"Cerco: *{q}*")
            result = search_web(q)
            context_data += f"\n--- Risultati per '{q}' ---\n{result}\n"
    
    # 4. FASE DI GENERATION (RAG)
    st.info("🤖 Elaborazione della previsione...")
    
    final_prompt = ChatPromptTemplate.from_template(
        """Sei un esperto analista di tennis.
        
        DATA DI OGGI: {date}
        
        CONTESTO RECUPERATO (WEB):
        {context}
        
        DOMANDA UTENTE: {query}
        
        ISTRUZIONI:
        Analizza il contesto. Fai una previsione sul vincitore basandoti su:
        - Forma recente
        - Precedenti (H2H)
        - Adattabilità alla superficie
        
        Rispondi in italiano. Sii sintetico e professionale.
        """
    )
    
    final_chain = final_prompt | llm | StrOutputParser()
    response = final_chain.invoke({
        "date": current_date, 
        "context": context_data, 
        "query": query
    })
    
    return response

# --- INTERFACCIA UTENTE PRINCIPALE ---

query = st.text_input("Di quale match vuoi la previsione?", placeholder="Es: Chi vince tra Sinner e Alcaraz a Indian Wells?")

if st.button("Analizza Match"):
    if not api_key:
        st.error("⚠️ Per favore inserisci la API Key di Groq nella barra laterale.")
    elif not query:
        st.warning("⚠️ Inserisci una domanda.")
    else:
        try:
            result = run_tennis_agent(query, api_key)
            st.success("Analisi completata!")
            st.markdown("### 🏆 Previsione dell'Agente")
            st.markdown(result)
        except Exception as e:
            st.error(f"Si è verificato un errore: {e}")
