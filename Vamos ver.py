# REINICIO_COMPLETO.py
# Football Studio Card Analyzer - Versão PROFISSIONAL (Completa e Corrigida)
# Histórico vertical, inserção automática, análise de duas cartas, sugestões e exportação.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------------- Configurações -----------------------------
st.set_page_config(
    page_title="Football Studio Analyzer - Profissional",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Football Studio Analyzer - Profissional (Vertical • Auto Inserção)")
st.markdown("Registro das 2 cartas por rodada (BLUE + RED). Sistema replica sua versão original com melhorias essenciais.")

# ----------------------------- Constantes -----------------------------
CARD_MAP = {
    'A': 14, 'K': 13, 'Q': 12, 'J': 11,
    '10': 10, '9': 9, '8': 8, '7': 7, '6': 6,
    '5': 5, '4': 4, '3': 3, '2': 2
}

HIGH = {'A', 'K', 'Q', 'J'}
MEDIUM = {'10', '9', '8'}
LOW = {'7', '6', '5', '4', '3', '2'}

# ----------------------------- Funções Utilitárias -----------------------------
def card_value(card_label: str) -> int:
    return CARD_MAP.get(str(card_label), 0)

def classify_card(card_label: str) -> str:
    if card_label in HIGH:
        return 'alta'
    if card_label in MEDIUM:
        return 'media'
    if card_label in LOW:
        return 'baixa'
    return 'indefinido'

def strength_of_duel(v_blue: int, v_red: int) -> str:
    if v_blue == 0 or v_red == 0:
        return 'indefinido'
    diff = abs(v_blue - v_red)
    if diff <= 2:
        return 'fraco'
    if diff <= 4:
        return 'medio'
    return 'forte'

def determine_winner(v_blue: int, v_red: int) -> str:
    if v_blue == v_red:
        return 'tie'
    return 'blue' if v_blue > v_red else 'red'

# ----------------------------- Histórico -----------------------------
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'timestamp', 'blue_card', 'value_blue', 'value_class_blue',
        'red_card', 'value_red', 'value_class_red',
        'winner', 'diff', 'strength'
    ])

def add_result(blue_card: str, red_card: str):
    now = datetime.now()
    vb = card_value(blue_card)
    vr = card_value(red_card)
    vc_blue = classify_card(blue_card)
    vc_red = classify_card(red_card)
    winner = determine_winner(vb, vr)
    diff = abs(vb - vr)
    strength = strength_of_duel(vb, vr)
    new_row = {
        'timestamp': now,
        'blue_card': blue_card,
        'value_blue': vb,
        'value_class_blue': vc_blue,
        'red_card': red_card,
        'value_red': vr,
        'value_class_red': vc_red,
        'winner': winner,
        'diff': diff,
        'strength': strength
    }
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)

def reset_history():
    st.session_state.history = pd.DataFrame(columns=[
        'timestamp', 'blue_card', 'value_blue', 'value_class_blue',
        'red_card', 'value_red', 'value_class_red',
        'winner', 'diff', 'strength'
    ])

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header('Controles & Export')
    if st.button('Resetar Histórico'):
        reset_history()
    st.markdown('Exportar Histórico')
    csv_data = st.session_state.history.to_csv(index=False)
    st.download_button('Exportar CSV', data=csv_data, file_name='history_football_studio.csv')
    st.write('---')
    show_timestamps = st.checkbox('Mostrar timestamps', value=False)
    show_confidence_bar = st.checkbox('Mostrar barras de confiança', value=True)

# ----------------------------- Inserção Vertical -----------------------------
st.subheader("Inserir Resultado (Auto-inserção)")

for i in range(1):  # linha única de inserção
    blue_card = st.selectbox("Carta BLUE", options=list(CARD_MAP.keys()), key=f"blue_card_{i}")
    red_card = st.selectbox("Carta RED", options=list(CARD_MAP.keys()), key=f"red_card_{i}")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("🔵 BLUE", key=f"btn_blue_{i}"):
            add_result(blue_card, red_card)
    with col2:
        if st.button("🔴 RED", key=f"btn_red_{i}"):
            add_result(blue_card, red_card)
    with col3:
        if st.button("🟡 TIE", key=f"btn_tie_{i}"):
            add_result(blue_card, red_card)

st.markdown('---')

# ----------------------------- Histórico Vertical -----------------------------
st.subheader("Histórico das Rodadas")

history = st.session_state.history.copy()

if history.empty:
    st.info("Sem resultados ainda.")
else:
    for idx, row in history.iterrows():
        label = ""
        if row['winner'] == 'red':
            label = f"🔴 RED {row['red_card']} vs {row['blue_card']} ({row['strength']})"
        elif row['winner'] == 'blue':
            label = f"🔵 BLUE {row['blue_card']} vs {row['red_card']} ({row['strength']})"
        else:
            label = f"🟡 TIE {row['blue_card']}|{row['red_card']} ({row['strength']})"
        if show_timestamps:
            label += f"  [{row['timestamp']}]"
        st.markdown(f"**{label}**")

# ----------------------------- Análise -----------------------------
def pattern_of_sequence(history):
    if history.empty:
        return "indefinido"
    winners = history['winner'].tolist()
    if len(winners) >= 3 and winners[-1] == winners[-2] == winners[-3]:
        return "repetição"
    if len(winners) >= 4 and winners[-1] == winners[-3] and winners[-2] == winners[-4] and winners[-1] != winners[-2]:
        return "alternância"
    if len(winners) >= 6:
        seq = winners[-6:]
        if seq[0]==seq[1] and seq[2]==seq[3] and seq[4]==seq[5] and seq[0]==seq[4]:
            return "degrau"
    return "indefinido"

def analyze_tendency(history):
    if history.empty:
        return {"pattern":"indefinido","prob_red":0,"prob_blue":0,"prob_tie":0,"suggestion":"aguardar","confidence":0}
    last = history.iloc[-1]
    pattern = pattern_of_sequence(history)
    winner = last['winner']
    strength = last['strength']
    prob_red = prob_blue = prob_tie = 0
    conf = 0.5
    if strength=="forte":
        if winner=="red": prob_red=78; prob_blue=20; prob_tie=2
        elif winner=="blue": prob_red=20; prob_blue=78; prob_tie=2
        else: prob_red=15; prob_blue=15; prob_tie=70
        conf=0.78
    elif strength=="medio":
        if winner=="red": prob_red=62; prob_blue=35; prob_tie=3
        elif winner=="blue": prob_red=35; prob_blue=62; prob_tie=3
        else: prob_red=48; prob_blue=48; prob_tie=4
        conf=0.58
    else:  # fraco
        if winner=="red": prob_red=24; prob_blue=74; prob_tie=2
        elif winner=="blue": prob_red=74; prob_blue=24; prob_tie=2
        else: prob_red=47; prob_blue=47; prob_tie=6
        conf=0.72
    # ajuste simples por padrão
    if pattern=="repetição":
        if winner=="red": prob_red=min(97, prob_red+12)
        elif winner=="blue": prob_blue=min(97, prob_blue+12)
        conf=max(conf,0.78)
    suggestion="aguardar"
    top_prob=max(prob_red, prob_blue, prob_tie)
    if top_prob>=60 or conf>=0.7:
        if top_prob==prob_red: suggestion="apostar RED (🔴)"
        elif top_prob==prob_blue: suggestion="apostar BLUE (🔵)"
        else: suggestion="apostar TIE (🟡)"
    return {"pattern":pattern,"prob_red":prob_red,"prob_blue":prob_blue,"prob_tie":prob_tie,"suggestion":suggestion,"confidence":round(conf*100,1)}

def manipulation_level(history):
    if history.empty: return 1
    winners = history['winner'].tolist()
    strengths = history['strength'].tolist()
    score = 0
    weak_run = sum(1 for s in strengths if s=="fraco")
    score += weak_run*1.5
    altern = sum(1 for i in range(1,len(winners)) if winners[i]!=winners[i-1] and winners[i]!="tie")
    score += altern*1.5
    return min(9,max(1,int(round(score))))

st.subheader("Análise e Previsão")
analysis = analyze_tendency(history)
level = manipulation_level(history)

colA, colB = st.columns([2,1])
with colA:
    st.markdown(f"**Padrão detectado:** {analysis['pattern'].capitalize()}")
    st.markdown(f"**Nível de manipulação:** {level}")
    st.markdown(f"**Sugestão:** {analysis['suggestion']}")
    st.markdown(f"**Confiança do modelo:** {analysis['confidence']}%")
    if show_confidence_bar: st.progress(int(analysis['confidence']))

with colB:
    st.markdown("Últimas 10 jogadas")
    st.dataframe(history.tail(10).reset_index(drop=True))

# ----------------------------- Exportação TXT -----------------------------
st.markdown('---')
st.header("Exportação de Relatório")
if st.button("Exportar TXT"):
    txt = f"Football Studio Analyzer - Relatório\nGerado em {datetime.now()}\n"
    txt += f"Padrão: {analysis['pattern']}\nNível de manipulação: {level}\nSugestão: {analysis['suggestion']}\n"
    txt += f"Probabilidades: RED {analysis['prob_red']}%, BLUE {analysis['prob_blue']}%, TIE {analysis['prob_tie']}%\n"
    txt += "\nÚltimas 10 jogadas:\n"
    txt += history.tail(10).to_string(index=False)
    st.download_button("Baixar Relatório", data=txt, file_name="relatorio_football_studio.txt")

st.caption("Sistema de análise de Football Studio. Heurísticas baseadas em cartas e padrões. Aposte com responsabilidade.")
