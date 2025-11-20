# REINICIO.py
# Football Studio Card Analyzer - Profissional (Completo)
# - Inserção por grade de cartas (sem selects)
# - Mantém e estende seu BKM
# - Detecta brechas, prevê múltiplos caminhos, calcula nível 1-9
# - Export, desfazer, destaque etc.
#
# Execute: streamlit run REINICIO.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

# ----------------------------- Config -----------------------------
st.set_page_config(page_title="Football Studio Analyzer - Completo", layout="wide", initial_sidebar_state="expanded")
st.title("Football Studio Analyzer — Versão Completa")
st.markdown("Inserção por cartas (grade), análise multi-caminho, detecção de brechas e nível de manipulação (1–9).")

# ----------------------------- Constantes -----------------------------
CARD_MAP = {
    'A': 14, 'K': 13, 'Q': 12, 'J': 11,
    '10': 10, '9': 9, '8': 8, '7': 7, '6': 6,
    '5': 5, '4': 4, '3': 3, '2': 2
}
CARD_ORDER = ['A','K','Q','J','10','9','8','7','6','5','4','3','2']

HIGH = {'A','K','Q','J'}
MEDIUM = {'10','9','8'}
LOW = {'7','6','5','4','3','2'}

MAX_COLS = 9
MAX_LINES = 10
MAX_DISPLAY = MAX_COLS * MAX_LINES

# ----------------------------- Utilitários BKM (mantidos e expandidos) -----------------------------
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

# Padrões (preservados e ajustáveis)
def pattern_of_sequence(history: pd.DataFrame) -> str:
    if history.empty:
        return 'indefinido'
    winners = history['winner'].tolist()
    strengths = history['strength'].tolist() if 'strength' in history.columns else []
    if len(winners) >= 3 and winners[-1] == winners[-2] == winners[-3]:
        return 'repetição'
    if len(winners) >= 4 and winners[-1] == winners[-3] and winners[-2] == winners[-4] and winners[-1] != winners[-2]:
        return 'alternância'
    if len(winners) >= 6:
        seq = winners[-6:]
        if seq[0] == seq[1] and seq[2] == seq[3] and seq[4] == seq[5] and seq[0] == seq[4] and seq[1] == seq[5]:
            return 'degrau'
    if 'strength' in history.columns and len(history) >= 3:
        s = strengths
        if len(s) >= 3 and s[-1] == 'forte' and s[-2] == 'fraco' and s[-3] == 'fraco':
            return 'quebra controlada'
    return 'indefinido'

# Heurística de análise principal (preservando sua base)
def analyze_tendency(history: pd.DataFrame) -> dict:
    if history.empty:
        return {'pattern': 'indefinido','prob_red': 0.0,'prob_blue': 0.0,'prob_tie': 0.0,'suggestion':'aguardar','confidence':0.0}
    last = history.iloc[-1]
    pattern = pattern_of_sequence(history)
    prob = {'red':0.0,'blue':0.0,'tie':0.0}
    confidence = 0.0
    last_strength = last.get('strength','indefinido')
    last_winner = last['winner']
    # base
    if last_strength == 'forte':
        repeat_prob = 0.78; other_prob = 1 - repeat_prob
        if last_winner == 'red':
            prob['red'] = repeat_prob; prob['blue'] = other_prob*0.95
        elif last_winner == 'blue':
            prob['blue'] = repeat_prob; prob['red'] = other_prob*0.95
        else:
            prob['tie'] = 0.7; prob['red'] = 0.15; prob['blue'] = 0.15
        confidence = 0.78
    elif last_strength == 'medio':
        base = 0.62
        if last_winner == 'red':
            prob['red'] = base; prob['blue'] = 1 - base - 0.03
        elif last_winner == 'blue':
            prob['blue'] = base; prob['red'] = 1 - base - 0.03
        else:
            prob['tie'] = 0.04; prob['red'] = 0.48; prob['blue'] = 0.48
        confidence = 0.58
    elif last_strength == 'fraco':
        break_prob = 0.74
        if last_winner == 'red':
            prob['blue'] = break_prob; prob['red'] = max(0.0, 1 - break_prob - 0.03)
        elif last_winner == 'blue':
            prob['red'] = break_prob; prob['blue'] = max(0.0, 1 - break_prob - 0.03)
        else:
            prob['tie'] = 0.06; prob['red'] = 0.47; prob['blue'] = 0.47
        confidence = 0.72
    else:
        prob = {'red':0.49,'blue':0.49,'tie':0.02}; confidence = 0.4
    # ajustes por padrão
    if pattern == 'repetição':
        if last_winner == 'red': prob['red'] = min(0.97, prob['red'] + 0.12)
        elif last_winner == 'blue': prob['blue'] = min(0.97, prob['blue'] + 0.12)
        confidence = max(confidence, 0.78)
    elif pattern == 'alternância':
        if last_winner == 'red':
            prob['blue'] = max(prob['blue'], 0.58); prob['red'] = 1 - prob['blue'] - prob.get('tie',0)
        elif last_winner == 'blue':
            prob['red'] = max(prob['red'], 0.58); prob['blue'] = 1 - prob['red'] - prob.get('tie',0)
        confidence = max(confidence, 0.62)
    elif pattern == 'degrau':
        if len(history) >= 2 and history.iloc[-2]['winner'] == last_winner:
            if last_winner == 'red': prob['red'] = max(prob['red'], 0.72)
            else: prob['blue'] = max(prob['blue'], 0.72)
            confidence = max(confidence, 0.72)
    elif pattern == 'quebra controlada':
        prob['tie'] = max(prob.get('tie',0.03), 0.06)
        if last_winner == 'red': prob['red'] = max(prob['red'], 0.62)
        else: prob['blue'] = max(prob['blue'], 0.62)
        confidence = max(confidence, 0.68)
    # ties recentes
    recent_ties = history['winner'].tail(4).tolist().count('tie')
    if recent_ties >= 1:
        confidence = min(0.85, confidence * 0.9)
        prob['tie'] = max(prob.get('tie',0.03), 0.03 + recent_ties*0.02)
    # normaliza
    total = prob['red'] + prob['blue'] + prob.get('tie',0.0)
    if total <= 0:
        prob = {'red':0.49,'blue':0.49,'tie':0.02}; total = 1.0
    for k in prob: prob[k] = prob[k] / total
    prob_pct = {k: round(v*100,1) for k,v in prob.items()}
    sorted_probs = sorted(prob_pct.items(), key=lambda x: x[1], reverse=True)
    top_label, top_val = sorted_probs[0]
    suggestion = 'aguardar'
    if top_val >= 60 or confidence >= 0.7:
        if top_label == 'red': suggestion = 'apostar RED (🔴)'
        elif top_label == 'blue': suggestion = 'apostar BLUE (🔵)'
        else: suggestion = 'apostar TIE (🟡)'
    return {'pattern':pattern,'prob_red':prob_pct['red'],'prob_blue':prob_pct['blue'],'prob_tie':prob_pct['tie'],'suggestion':suggestion,'confidence':round(confidence*100,1)}

# Manipulation level (ampliado)
def manipulation_level(history: pd.DataFrame) -> int:
    if history.empty:
        return 1
    vals_blue = history['value_blue'].tolist()
    vals_red = history['value_red'].tolist()
    winners = history['winner'].tolist()
    strengths = history['strength'].tolist()
    score = 0.0
    # runs fracas
    weak_runs = 0; run = 0
    for s in strengths:
        if s == 'fraco': run += 1
        else:
            if run >= 2: weak_runs += 1
            run = 0
    if run >= 2: weak_runs += 1
    score += weak_runs * 1.6
    # alternações
    alternations = sum(1 for i in range(1, len(winners)) if winners[i] != winners[i-1] and winners[i] != 'tie' and winners[i-1] != 'tie')
    alternation_rate = alternations / max(1, (len(winners)-1))
    score += alternation_rate * 3.4
    # vitórias por baixas
    low_win_count = 0
    for idx, w in enumerate(winners):
        if w == 'red':
            if classify_card(history.iloc[idx]['red_card']) == 'baixa' and classify_card(history.iloc[idx]['blue_card']) != 'alta':
                low_win_count += 1
        elif w == 'blue':
            if classify_card(history.iloc[idx]['blue_card']) == 'baixa' and classify_card(history.iloc[idx]['red_card']) != 'alta':
                low_win_count += 1
    low_rate = low_win_count / max(1, len(winners))
    score += low_rate * 3.2
    # ties reduzem
    tie_rate = winners.count('tie') / max(1, len(winners))
    score -= tie_rate * 1.6
    # muitas cartas altas reduzem suspeita
    high_count = sum(1 for i in range(len(vals_blue)) if vals_blue[i] >= 11 or vals_red[i] >= 11)
    high_rate = high_count / max(1, len(vals_blue))
    score -= high_rate * 2.2
    level = int(min(9, max(1, round(score))))
    return level

# Breach detection (heurísticas)
def detect_breaches(history: pd.DataFrame) -> List[Dict]:
    flags = []
    if history.empty:
        return flags
    # 1) sequência de vitórias por cartas baixas
    low_wins = 0
    for idx in range(len(history)):
        row = history.iloc[idx]
        if row['winner'] == 'red':
            if classify_card(row['red_card']) == 'baixa':
                low_wins += 1
            else:
                low_wins = 0
        elif row['winner'] == 'blue':
            if classify_card(row['blue_card']) == 'baixa':
                low_wins += 1
            else:
                low_wins = 0
        else:
            low_wins = 0
        if low_wins >= 3:
            flags.append({'type':'low_win_streak','index':idx,'desc':'3+ vitórias consecutivas por cartas baixas'})
            break
    # 2) carta baixa vencendo carta alta (suspeita)
    for idx in range(len(history)):
        row = history.iloc[idx]
        vb = row.get('value_blue',0); vr = row.get('value_red',0)
        if row['winner'] == 'red' and classify_card(row['red_card']) == 'baixa' and vb >= 11:
            flags.append({'type':'low_beat_high','index':idx,'desc':'Carta baixa venceu carta alta (RED venceu uma carta alta BLUE)'})
        if row['winner'] == 'blue' and classify_card(row['blue_card']) == 'baixa' and vr >= 11:
            flags.append({'type':'low_beat_high','index':idx,'desc':'Carta baixa venceu carta alta (BLUE venceu uma carta alta RED)'})
    # 3) alternância perfeita (ABAB) repetida -> possível manipulação
    winners = history['winner'].tolist()
    for i in range(len(winners)-3):
        seq = winners[i:i+4]
        if seq[0] == seq[2] and seq[1] == seq[3] and seq[0] != seq[1]:
            flags.append({'type':'perfect_alternation','index':i,'desc':'Alternância ABAB detectada'})
            break
    return flags

# Stake suggestion (conservador)
def stake_suggestion(confidence_pct: float, manipulation_lvl: int) -> float:
    # retorna porcentagem do bankroll sugerida (conservador)
    # confiança 0-100; maniplevel 1-9
    base = max(0.5, min(5.0, confidence_pct / 20.0))  # 0.5% a 5% pela confiança
    # penalizar stake se manipulação alta
    penalty = 1.0 - ((manipulation_lvl - 1) / 12.0)  # nível 9 reduz stake ~ 66%
    stake = base * penalty
    # garantir mínimo e máximo
    stake = round(max(0.25, min(5.0, stake)), 2)
    return stake

# ----------------------------- Storage inicial (mantém BKM) -----------------------------
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'timestamp','blue_card','value_blue','value_class_blue',
        'red_card','value_red','value_class_red',
        'winner','diff','strength'
    ])

# UI state
if 'ui_mobile' not in st.session_state:
    st.session_state.ui_mobile = False
if 'record_tie_cards' not in st.session_state:
    st.session_state.record_tie_cards = True

# ----------------------------- Funções de manipulação -----------------------------
def add_round(blue_card: str, red_card: str):
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

def add_round_tie(optional_blue: str=None, optional_red: str=None):
    now = datetime.now()
    vb = card_value(optional_blue) if (optional_blue and st.session_state.record_tie_cards) else 0
    vr = card_value(optional_red) if (optional_red and st.session_state.record_tie_cards) else 0
    vc_blue = classify_card(optional_blue) if (optional_blue and st.session_state.record_tie_cards) else 'indefinido'
    vc_red = classify_card(optional_red) if (optional_red and st.session_state.record_tie_cards) else 'indefinido'
    diff = abs(vb - vr)
    strength = strength_of_duel(vb, vr) if vb and vr else 'indefinido'
    new_row = {
        'timestamp': now,
        'blue_card': optional_blue if (optional_blue and st.session_state.record_tie_cards) else '0',
        'value_blue': vb,
        'value_class_blue': vc_blue,
        'red_card': optional_red if (optional_red and st.session_state.record_tie_cards) else '0',
        'value_red': vr,
        'value_class_red': vc_red,
        'winner': 'tie',
        'diff': diff,
        'strength': strength
    }
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)

def reset_history():
    st.session_state.history = pd.DataFrame(columns=[
        'timestamp','blue_card','value_blue','value_class_blue',
        'red_card','value_red','value_class_red',
        'winner','diff','strength'
    ])

def remove_last():
    if len(st.session_state.history) > 0:
        st.session_state.history = st.session_state.history.iloc[:-1].reset_index(drop=True)

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Controles & Export")
    if st.button("Resetar Histórico"):
        reset_history()
    if st.button("Remover último"):
        remove_last()
    st.write('---')
    csv = st.session_state.history.to_csv(index=False)
    st.download_button("Exportar histórico (CSV)", data=csv, file_name='history_football_studio.csv')
    if st.button("Exportar relatório (TXT)"):
        # gerar relatório simples
        txt = "Football Studio Analyzer - Relatório\n"
        txt += f"Gerado em: {datetime.now()}\n\n"
        if not st.session_state.history.empty:
            txt += st.session_state.history.tail(50).to_string(index=False)
        else:
            txt += "Sem dados.\n"
        st.download_button("Download TXT", data=txt, file_name='relatorio_football_studio.txt')
    st.write('---')
    st.session_state.ui_mobile = st.checkbox("Modo Mobile (botões maiores)", value=st.session_state.ui_mobile)
    st.session_state.record_tie_cards = st.checkbox("Registrar cartas no TIE (se selecionadas)", value=st.session_state.record_tie_cards)
    st.write('---')
    st.caption("Clique na carta e depois no botão da cor para gravar (ou use os botões rápidos de inserção).")

# ----------------------------- Inserção por grade (sem selects) -----------------------------
st.subheader("Inserir Resultados — Grade de Cartas (sem selects)")

# estilo de botões maior via CSS
big_btn = """
<style>
div.stButton > button { height:58px; font-size:16px; }
</style>
"""
big_btn_mobile = """
<style>
div.stButton > button { height:84px; font-size:20px; }
</style>
"""
st.markdown(big_btn_mobile if st.session_state.ui_mobile else big_btn, unsafe_allow_html=True)

# área: escolhe carta e registra com cor
col_blue_area, col_middle_area, col_red_area = st.columns([4,1,4])

with col_blue_area:
    st.markdown("**🔵 Escolha a carta BLUE (clique na carta, depois em 'Inserir BLUE')**")
    ccols = st.columns(7)
    # armazenar seleção localmente em session_state
    if 'sel_blue' not in st.session_state:
        st.session_state.sel_blue = None
    for i, c in enumerate(CARD_ORDER):
        if ccols[i%7].button(c, key=f"b_{c}"):
            st.session_state.sel_blue = c
    st.write(f"Selecionada BLUE: **{st.session_state.sel_blue or '-'}**")
    if st.button("🔵 Inserir BLUE (usa RED selecionada se houver)", key="insert_blue_btn"):
        sel_red = st.session_state.get('sel_red', CARD_ORDER[0])
        if not st.session_state.sel_blue:
            st.warning("Selecione a carta BLUE antes de inserir.")
        else:
            add_round(st.session_state.sel_blue, sel_red)

with col_middle_area:
    st.markdown(" ")
    st.write(" ")
    # botão Tie rápido (opcionalmente grava cartas selecionadas)
    if st.button("🟡 Inserir TIE (opcional cartas selecionadas)", key="insert_tie_btn"):
        add_round_tie(st.session_state.get('sel_blue', None), st.session_state.get('sel_red', None))

with col_red_area:
    st.markdown("**🔴 Escolha a carta RED (clique na carta, depois em 'Inserir RED')**")
    rcols = st.columns(7)
    if 'sel_red' not in st.session_state:
        st.session_state.sel_red = None
    for i, c in enumerate(CARD_ORDER):
        if rcols[i%7].button(c, key=f"r_{c}"):
            st.session_state.sel_red = c
    st.write(f"Selecionada RED: **{st.session_state.sel_red or '-'}**")
    if st.button("🔴 Inserir RED (usa BLUE selecionada se houver)", key="insert_red_btn"):
        sel_blue = st.session_state.get('sel_blue', CARD_ORDER[0])
        if not st.session_state.sel_red:
            st.warning("Selecione a carta RED antes de inserir.")
        else:
            add_round(sel_blue, st.session_state.sel_red)

st.write("---")

# ----------------------------- Histórico visual (9x10) -----------------------------
st.subheader("Histórico — Visual 9×10 (últimos resultados)")

history = st.session_state.history.copy()
total_len = len(st.session_state.history)

if total_len == 0:
    st.info("Nenhum resultado inserido ainda.")
else:
    # limitar para visual
    hist_vis = history.tail(MAX_DISPLAY).reset_index(drop=True)
    rows = [hist_vis.iloc[i:i+MAX_COLS] for i in range(0, len(hist_vis), MAX_COLS)]
    # destaque último 3 (global index)
    for r_idx, row_df in enumerate(rows):
        cols = st.columns(MAX_COLS)
        for c_idx in range(MAX_COLS):
            with cols[c_idx]:
                idx = r_idx*MAX_COLS + c_idx
                if idx < len(row_df):
                    item = row_df.iloc[idx]
                    # label
                    if item['winner'] == 'red':
                        label = f"🔴 {item['red_card']} vs {item['blue_card']}\n({item['strength']})"
                    elif item['winner'] == 'blue':
                        label = f"🔵 {item['blue_card']} vs {item['red_card']}\n({item['strength']})"
                    else:
                        label = f"🟡 TIE {item['blue_card']}|{item['red_card']}\n({item['strength']})"
                    # compute global index in history
                    global_idx = total_len - len(hist_vis) + idx
                    highlight = (global_idx >= total_len - 3)
                    if highlight:
                        st.markdown(f"<div style='background:#fff7a8;border-radius:6px;padding:6px'><b>{label}</b></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='padding:4px'>{label}</div>", unsafe_allow_html=True)
                else:
                    st.write("")

st.write("---")

# ----------------------------- Análise principal + multi-caminho -----------------------------
st.subheader("Análise, Previsões Multi-Caminho e Detecção de Brechas")

analysis = analyze_tendency(st.session_state.history)
level = manipulation_level(st.session_state.history)
breaches = detect_breaches(st.session_state.history)

# cenário atual
st.markdown("### Situação Atual")
st.markdown(f"**Padrão detectado:** {analysis['pattern'].capitalize()}")
st.markdown(f"**Probabilidades (heur.):** RED {analysis['prob_red']}% • BLUE {analysis['prob_blue']}% • TIE {analysis['prob_tie']}%")
st.markdown(f"**Sugestão:** {analysis['suggestion']}")
st.markdown(f"**Confiança:** {analysis['confidence']}%")
st.markdown(f"**Nível de manipulação (1–9):** {level}")
stake_pct = stake_suggestion(analysis['confidence'], level)
st.markdown(f"**Stake sugerido (conservador):** {stake_pct}% do bankroll (ex.: se R$100 -> R${round(100*stake_pct/100,2)})")
st.caption("Sugestão conservadora — ajuste conforme sua gestão de banca. Aposte com responsabilidade.")

# multi-caminho: simular próxima rodada sendo RED, BLUE, TIE e recalcular análise condicional
st.markdown("### Previsões Condicionais (se o próximo for RED / BLUE / TIE) — Multi-caminho")
def simulate_next_and_analyze(history_df: pd.DataFrame, next_winner: str, next_blue_card=None, next_red_card=None):
    # cria cópia e adiciona hipotético
    h = history_df.copy()
    if next_winner == 'red':
        bc = next_blue_card if next_blue_card else h.iloc[-1]['blue_card'] if not h.empty else CARD_ORDER[0]
        rc = next_red_card if next_red_card else CARD_ORDER[0]
        vb = card_value(bc); vr = card_value(rc)
        winner = 'red'
    elif next_winner == 'blue':
        bc = next_blue_card if next_blue_card else CARD_ORDER[0]
        rc = next_red_card if next_red_card else h.iloc[-1]['red_card'] if not h.empty else CARD_ORDER[0]
        vb = card_value(bc); vr = card_value(rc)
        winner = 'blue'
    else:
        # tie
        bc = next_blue_card if next_blue_card else (h.iloc[-1]['blue_card'] if not h.empty else CARD_ORDER[0])
        rc = next_red_card if next_red_card else (h.iloc[-1]['red_card'] if not h.empty else CARD_ORDER[0])
        vb = card_value(bc); vr = card_value(rc)
        winner = 'tie'
    # compute fields
    strength = strength_of_duel(vb, vr)
    new_row = {
        'timestamp': datetime.now(),
        'blue_card': bc,
        'value_blue': vb,
        'value_class_blue': classify_card(bc),
        'red_card': rc,
        'value_red': vr,
        'value_class_red': classify_card(rc),
        'winner': winner,
        'diff': abs(vb-vr),
        'strength': strength
    }
    h = pd.concat([h, pd.DataFrame([new_row])], ignore_index=True)
    return analyze_tendency(h), manipulation_level(h)

# compute scenarios
if not st.session_state.history.empty:
    scen_red = simulate_next_and_analyze(st.session_state.history, 'red')
    scen_blue = simulate_next_and_analyze(st.session_state.history, 'blue')
    scen_tie = simulate_next_and_analyze(st.session_state.history, 'tie')
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Se próximo = RED**")
        st.write(f"Prob RED {scen_red[0]['prob_red']}% • Suggest: {scen_red[0]['suggestion']} • Conf: {scen_red[0]['confidence']}%")
        st.write(f"Manip lvl (após) : {scen_red[1]}")
    with cols[1]:
        st.markdown("**Se próximo = BLUE**")
        st.write(f"Prob BLUE {scen_blue[0]['prob_blue']}% • Suggest: {scen_blue[0]['suggestion']} • Conf: {scen_blue[0]['confidence']}%")
        st.write(f"Manip lvl (após) : {scen_blue[1]}")
    with cols[2]:
        st.markdown("**Se próximo = TIE**")
        st.write(f"Prob TIE {scen_tie[0]['prob_tie']}% • Suggest: {scen_tie[0]['suggestion']} • Conf: {scen_tie[0]['confidence']}%")
        st.write(f"Manip lvl (após) : {scen_tie[1]}")
else:
    st.info("Insira ao menos 1 rodada para ver previsões condicionais.")

# Breaches
st.markdown("### Brechas / Sinais de Suspeita detectados")
if breaches:
    for b in breaches:
        st.warning(f"[{b['type']}] índice {b['index']}: {b['desc']}")
else:
    st.success("Nenhuma brecha importante detectada nas heurísticas atuais.")

st.write("---")

# ----------------------------- Últimas jogadas e tabela detalhada -----------------------------
st.subheader("Últimas 10 jogadas (detalhado)")
if st.session_state.history.empty:
    st.info("Sem dados.")
else:
    st.dataframe(st.session_state.history.tail(10).reset_index(drop=True), use_container_width=True)

# ----------------------------- Interpretação e instruções operacionais -----------------------------
st.subheader("Interpretação (resumo operacional)")
st.markdown("""
- Registre **sempre as 2 cartas** (BLUE e RED) para cada rodada; o sistema usa ambas para calcular força e padrões.
- Diferença ≤2 => rodada **fraca** (alto risco de quebra).  
- Diferença 3–4 => rodada **média**.  
- Diferença ≥5 => rodada **forte** (tende a repetir o vencedor).
- **Nível de manipulação (1–9)**: quanto maior, maior a suspeita de padrões artificiais — reduza stake.
- Use a **Stake sugerida** como referência conservadora, não instrução absoluta.
""")

st.caption("Este software aplica heurísticas. Não existe garantia de lucro. Aposte com responsabilidade.")

# ----------------------------- EOF -----------------------------
