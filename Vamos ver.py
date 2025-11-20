# football_studio_professional.py
# App profissional unificado — Streamlit
# Janela de análise padrão: 15 (profissional)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------------- Configuração -----------------------------
st.set_page_config(page_title="Football Studio Analyzer - Profissional", layout="wide")
st.title("Football Studio Analyzer — Profissional (Modo: 15 cartas)")
st.markdown("Análise profissional unificada — força de cartas, padrões, detector de quebra, nível de manipulação (1-9) e previsão. Janela padrão: 15.")

# ----------------------------- Constantes -----------------------------
CARD_MAP = {
    "A": 14, "K": 13, "Q": 12, "J": 11,
    "10": 10, "9": 9, "8": 8, "7": 7,
    "6": 6, "5": 5, "4": 4, "3": 3, "2": 2
}
CARD_ORDER = ["A","K","Q","J","10","9","8","7","6","5","4","3","2"]

HIGH = {"A","K","Q","J"}
MEDIUM = {"10","9","8"}
LOW = {"7","6","5","4","3","2"}

# Strength for heuristics: 1..5
CARD_STRENGTH = {
    "A":5,"K":5,"Q":5,
    "J":4,"10":4,
    "9":3,"8":3,
    "7":2,"6":1,"5":1,"4":1,"3":1,"2":1
}

DEFAULT_WINDOW = 15
MAX_COLS = 9
MAX_LINES = 10

# ----------------------------- Utilitários -----------------------------
def card_value(label: str) -> int:
    return CARD_MAP.get(str(label), 0)

def card_group(label: str) -> str:
    if label in HIGH:
        return "alta"
    if label in MEDIUM:
        return "media"
    return "baixa"

def strength_of(label: str) -> int:
    return CARD_STRENGTH.get(label, 1)

# ----------------------------- Estado -----------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["timestamp","winner","card","value","value_class"])

# ----------------------------- Operações de histórico -----------------------------
def add_result(winner: str, card_label: str):
    now = datetime.now()
    v = card_value(card_label) if card_label != "T" else 0
    vc = card_group(card_label) if card_label != "T" else "tie"
    new = pd.DataFrame([{"timestamp":now,"winner":winner,"card":card_label,"value":v,"value_class":vc}])
    st.session_state.history = pd.concat([st.session_state.history, new], ignore_index=True)

def reset_history():
    st.session_state.history = pd.DataFrame(columns=["timestamp","winner","card","value","value_class"])

# ----------------------------- Sidebar / Config -----------------------------
with st.sidebar:
    st.header("Controles")
    if st.button("Resetar Histórico"):
        reset_history()
    st.write("---")
    st.markdown("Exportar / Configurações")
    csv = st.session_state.history.to_csv(index=False)
    st.download_button("Exportar histórico (CSV)", data=csv, file_name="history_football_studio.csv")
    st.write("---")
    show_timestamps = st.checkbox("Mostrar timestamps", value=False)
    show_confidence_bar = st.checkbox("Mostrar barra de confiança", value=True)
    window = st.slider("Janela de análise (nº de últimas jogadas)", min_value=5, max_value=50, value=DEFAULT_WINDOW, step=1)

# ----------------------------- Inserção rápida (botões coloridos) -----------------------------
st.subheader("Inserir Resultado — 1 clique (clique no valor da carta na coluna da cor)")

col_r, col_b, col_t = st.columns(3)

with col_r:
    st.markdown("<div style='text-align:center; color:#b30000; font-weight:bold;'>🔴 RED</div>", unsafe_allow_html=True)
    for c in CARD_ORDER:
        if st.button(c, key=f"r_{c}", use_container_width=True):
            add_result("red", c)

with col_b:
    st.markdown("<div style='text-align:center; color:#1f4fff; font-weight:bold;'>🔵 BLUE</div>", unsafe_allow_html=True)
    for c in CARD_ORDER:
        if st.button(c, key=f"b_{c}", use_container_width=True):
            add_result("blue", c)

with col_t:
    st.markdown("<div style='text-align:center; color:#c7a400; font-weight:bold;'>🟡 TIE</div>", unsafe_allow_html=True)
    if st.button("TIE", key="tie_btn", use_container_width=True):
        add_result("tie","T")

st.write("---")

# ----------------------------- Visualizar Histórico -----------------------------
st.subheader("Histórico (visualização)")

history = st.session_state.history.copy()
if history.empty:
    st.info("Sem resultados. Use os botões acima.")
else:
    # Limit view
    if len(history) > MAX_COLS * MAX_LINES:
        view = history.tail(MAX_COLS * MAX_LINES).reset_index(drop=True)
    else:
        view = history.copy()
    rows = [view.iloc[i:i+MAX_COLS] for i in range(0, len(view), MAX_COLS)]
    for row_df in rows:
        cols = st.columns(MAX_COLS)
        for idx in range(MAX_COLS):
            with cols[idx]:
                if idx < len(row_df):
                    item = row_df.iloc[idx]
                    if item["winner"] == "red":
                        label = f"🔴 {item['card']} ({item['value_class']})"
                    elif item["winner"] == "blue":
                        label = f"🔵 {item['card']} ({item['value_class']})"
                    else:
                        label = "🟡 TIE"
                    if show_timestamps:
                        st.caption(str(item["timestamp"]))
                    st.markdown(f"**{label}**")
                else:
                    st.write("")

# ----------------------------- Padrões e Heurísticas (UNIFICADAS) -----------------------------
def detect_pattern_unified(df: pd.DataFrame) -> str:
    """Padrão unificado usando sequência e classes."""
    if df.empty:
        return "indefinido"
    winners = df["winner"].tolist()
    classes = df["value_class"].tolist()

    # Repetição forte (últimos 3 iguais, não tie)
    if len(winners) >= 3 and winners[-1] == winners[-2] == winners[-3] and winners[-1] != "tie":
        return "repetição"

    # Alternância ABAB nos últimos 4
    if len(winners) >= 4 and winners[-1] == winners[-3] and winners[-2] == winners[-4] and winners[-1] != winners[-2]:
        return "alternância"

    # Degrau: AA BB AA (simples heurística)
    if len(winners) >= 6:
        seq = winners[-6:]
        if seq[0]==seq[1] and seq[2]==seq[3] and seq[4]==seq[5] and seq[0]==seq[4]:
            return "degrau"

    # Quebra controlada por classes: baixa, baixa, alta
    if len(classes) >= 3 and classes[-3]=="baixa" and classes[-2]=="baixa" and classes[-1]=="alta":
        return "quebra controlada"

    return "indefinido"

def compute_manipulation_level_unified(df: pd.DataFrame) -> int:
    """Nível 1..9: heurística que junta runs de baixas, alternações e poucas altas."""
    if df.empty:
        return 1
    vals = df["value_class"].tolist()
    winners = df["winner"].tolist()
    n = len(df)

    score = 0.0
    # runs of low values
    run = 0
    low_runs = 0
    for v in vals:
        if v == "baixa":
            run += 1
        else:
            if run >= 2:
                low_runs += 1
            run = 0
    if run >= 2:
        low_runs += 1
    score += low_runs * 1.6

    # alternation rate
    alternations = sum(1 for i in range(1,n) if winners[i] != winners[i-1])
    alternation_rate = alternations / max(1, (n-1))
    score += alternation_rate * 3.0

    # proportion of highs reduces score
    high_count = sum(1 for v in vals if v == "alta")
    high_rate = high_count / max(1, n)
    score -= high_rate * 1.6

    level = int(min(9, max(1, round(score))))
    return level

def detect_break_unified(df: pd.DataFrame) -> dict:
    """Detecta probabilidade de quebra na próxima jogada com razão/justificativa."""
    if df.empty:
        return {"break_expected": False, "reason": ""}

    # get recent window
    recent = df.tail(window).reset_index(drop=True)
    classes = recent["value_class"].tolist()
    winners = recent["winner"].tolist()

    # Heurística 1: sequência com muitas baixas nas últimas 5
    last5 = classes[-5:] if len(classes) >= 5 else classes
    low_count = sum(1 for x in last5 if x == "baixa")
    if low_count >= 3:
        return {"break_expected": True, "reason": f"{low_count}/5 baixas recentes"}

    # Heurística 2: última carta baixa
    if classes and classes[-1] == "baixa":
        return {"break_expected": True, "reason": "última carta baixa"}

    # Heurística 3: alternância acelerada
    last_w = winners[-6:] if len(winners) >= 6 else winners
    if len(last_w) >= 4:
        alt = sum(1 for i in range(1,len(last_w)) if last_w[i] != last_w[i-1])
        if alt >= (len(last_w)-1) * 0.75:
            return {"break_expected": True, "reason": "alternância acelerada (alto risco de quebra/empate)"}

    return {"break_expected": False, "reason": ""}

def weighted_probabilities_unified(df: pd.DataFrame, window_size: int) -> dict:
    """Calcula probabilidades RED/BLUE/TIE usando janela com pesos e força das cartas.
       Entrega probabilidades e confiança (0..100)."""
    if df.empty:
        return {"red":49.0,"blue":49.0,"tie":2.0,"confidence":0.0}
    sub = df.tail(window_size).reset_index(drop=True)
    m = len(sub)
    # weights: exponential decay giving more weight to recent ones
    decay = 0.85
    weights = np.array([decay**(m-1-i) for i in range(m)], dtype=float)
    weights = weights / weights.sum()

    score = {"red":0.0,"blue":0.0,"tie":0.0}
    for i, row in sub.iterrows():
        w = weights[i]
        winner = row["winner"]
        # factor based on card strength 0.2..1.0
        if row["card"] == "T":
            factor = 0.3
        else:
            factor = strength_of(row["card"]) / 5.0
        # winner contribution: base influenced by factor
        contrib = w * (0.5 + 0.5*factor)  # 0.5..1.0 scaled by factor
        if winner == "red":
            score["red"] += contrib
        elif winner == "blue":
            score["blue"] += contrib
        else:
            # tie stronger when factor low (many low cards)
            score["tie"] += w * (0.4 + 0.6*(1-factor))

    # normalize to percent
    total = score["red"] + score["blue"] + score["tie"]
    # smoothing
    total = total if total > 0 else 1.0
    probs = {k: round((v/total)*100,1) for k,v in score.items()}

    # confidence: peakness measured by normalized max
    vals = np.array(list(score.values()))
    peakness = vals.max() / max(1e-9, vals.sum())
    confidence = min(0.99, max(0.05, peakness)) * 100  # map to 5%..99%
    return {"red":probs["red"],"blue":probs["blue"],"tie":probs["tie"],"confidence":round(confidence,1)}

def make_final_suggestion(probs: dict, break_info: dict, manip_level: int, df: pd.DataFrame) -> str:
    """Gera sugestão única com justificativa interna (apenas uma saída)."""
    # If break expected -> favor opposite of recent dominant
    if break_info.get("break_expected"):
        # if tie probability high -> suggest tie
        if probs["tie"] >= 12:
            return "apostar TIE (🟡) — alto risco de empate"
        # determine recent dominant color
        recent = df.tail(5)["winner"].value_counts()
        if recent.empty:
            dominant = None
        else:
            dominant = recent.idxmax()
        if dominant == "red":
            return "apostar BLUE (🔵) — quebra provável"
        elif dominant == "blue":
            return "apostar RED (🔴) — quebra provável"
        else:
            # fallback to highest prob
            if probs["red"] > probs["blue"]:
                return "apostar RED (🔴) — quebra provável"
            else:
                return "apostar BLUE (🔵) — quebra provável"

    # if tie prob high, suggest tie
    if probs["tie"] >= 12:
        return "apostar TIE (🟡)"

    # otherwise pick top color with thresholds influenced by manipulation level
    top_color = "red" if probs["red"] > probs["blue"] else "blue"
    top_val = max(probs["red"], probs["blue"])

    # if high manipulation level, be more conservative
    threshold = 60
    if manip_level >= 6:
        threshold = 66
    if top_val >= threshold:
        return f"apostar {top_color.upper()} ({'🔴' if top_color=='red' else '🔵'})"
    # otherwise wait
    return "aguardar (sem entrada segura)"

# ----------------------------- Execução unificada da análise -----------------------------
st.subheader("Análise Profissional (unificada)")

pattern = detect_pattern_unified(history) if not history.empty else "indefinido"
manip_level = compute_manipulation_level_unified(history)
break_info = detect_break_unified(history)
probs = weighted_probabilities_unified(history, window)
suggestion = make_final_suggestion(probs, break_info, manip_level, history)

colA, colB = st.columns([2,1])
with colA:
    st.markdown(f"**Padrão detectado:** {pattern}")
    st.markdown(f"**Nível de manipulação (1–9):** {manip_level}")
    st.markdown(f"**Sugestão:** {suggestion}")
    st.markdown(f"**Justificativa (break):** {break_info['reason']}" if break_info["break_expected"] else "")
    st.markdown(f"**Confiança estimada:** {probs['confidence']} %")
with colB:
    st.metric("🔴 RED", f"{probs['red']} %")
    st.metric("🔵 BLUE", f"{probs['blue']} %")
    st.metric("🟡 TIE", f"{probs['tie']} %")

if show_confidence_bar:
    st.progress(int(min(100, probs["confidence"])))

st.markdown("---")

# ----------------------------- Resumo e interpretação -----------------------------
st.subheader("Resumo das últimas jogadas (últimas 15)")
st.dataframe(st.session_state.history.tail(DEFAULT_WINDOW).reset_index(drop=True))

st.markdown("**Interpretação por valor da carta**")
st.markdown("""
- A, K, Q: ALTAS — favorecem repetição.
- J, 10: MÉDIAS — transição provável.
- 9, 8: MÉDIAS para baixa — risco de quebra em 1–2 jogadas.
- 7–2: BAIXAS — alta probabilidade de quebra / manipulação.
""")

st.markdown("**Estratégia operacional sintetizada**")
st.markdown("""
1. Use a janela de 15 jogadas para leitura profissional (padrões + manipulação).  
2. Não entre sem que sugestão e confiança concordem (prob >= threshold).  
3. Em manipulação alta (níveis 6–9) seja mais conservador.  
4. Empates aparecem quando muitas baixas e alternância acelerada.
""")

# ----------------------------- Relatório / Export -----------------------------
st.markdown("---")
st.header("Ferramentas")

if st.button("Gerar relatório TXT"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txt = "Football Studio Analyzer - Relatório\n"
    txt += f"Gerado em: {now}\n"
    txt += f"Padrão: {pattern}\n"
    txt += f"Nível de manipulação: {manip_level}\n"
    txt += f"Sugestão: {suggestion}\n"
    txt += f"Probabilidades: RED {probs['red']}%, BLUE {probs['blue']}%, TIE {probs['tie']}%\n"
    txt += f"Confiança: {probs['confidence']}%\n\n"
    txt += "Últimas jogadas (até 50):\n"
    txt += st.session_state.history.tail(50).to_csv(index=False)
    st.download_button("Baixar relatório (TXT)", data=txt, file_name="relatorio_football_studio.txt")

st.caption("App profissional: análise unificada. Probabilidades são estimativas heurísticas; aposte com responsabilidade.")
