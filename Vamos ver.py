# REINICIO.py
# Football Studio Card Analyzer - Versão PROFISSIONAL (RéPLICA corrigida)
# Inserção horizontal sem confirmar, registra BLUE + RED (ou TIE),
# análise considerando as duas cartas da rodada.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------------- Configurações -----------------------------
st.set_page_config(page_title="Football Studio Analyzer - Profissional",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.title("Football Studio Analyzer - Profissional (Horizontal • Sem confirmar)")
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

MAX_COLS = 9   # resultados por linha
MAX_LINES = 10

# ----------------------------- Utilitários -----------------------------
def card_value(card_label: str) -> int:
    """Retorna o valor numérico da carta (A=14 ... 2=2). Se '0' ou inválido, retorna 0."""
    try:
        return CARD_MAP.get(str(card_label), 0)
    except Exception:
        return 0

def classify_card(card_label: str) -> str:
    """Classifica a carta como 'alta', 'media' ou 'baixa'."""
    if card_label in HIGH:
        return 'alta'
    if card_label in MEDIUM:
        return 'media'
    if card_label in LOW:
        return 'baixa'
    return 'indefinido'

def strength_of_duel(v_blue: int, v_red: int) -> str:
    """Classifica a força do duelo como 'forte','medio','fraco' com base em médias/diferença."""
    if v_blue == 0 or v_red == 0:
        return 'indefinido'
    diff = abs(v_blue - v_red)
    mean = (v_blue + v_red) / 2.0
    # regras heurísticas:
    # diferença <=2 -> fraco (rodada instável)
    # diferença 3-4 -> médio
    # diferença >=5 -> forte
    if diff <= 2:
        return 'fraco'
    if diff <= 4:
        return 'medio'
    return 'forte'

def determine_winner(v_blue: int, v_red: int) -> str:
    """Determina vencedor: 'blue','red' ou 'tie'."""
    if v_blue == v_red:
        return 'tie'
    return 'blue' if v_blue > v_red else 'red'

def pattern_of_sequence(history: pd.DataFrame) -> str:
    """Detecta padrão: repetição, alternância, degrau, quebra controlada, ou indefinido.
    Agora leva em conta winner + força das rodadas."""
    if history.empty:
        return 'indefinido'

    winners = history['winner'].tolist()
    strengths = history['strength'].tolist() if 'strength' in history.columns else []

    # repetição: últimos 3 iguais (e com força forte ou média)
    if len(winners) >= 3 and winners[-1] == winners[-2] == winners[-3]:
        # checar força das últimas
        if len(strengths) >= 3 and any(s in ['forte','medio'] for s in strengths[-3:]):
            return 'repetição'
        return 'repetição'  # mantém repetição mesmo que fracas, será ajustado depois

    # alternância: ABAB nos últimos 4
    if len(winners) >= 4 and winners[-1] == winners[-3] and winners[-2] == winners[-4] and winners[-1] != winners[-2]:
        return 'alternância'

    # degrau: duo-duo (A A B B A A)
    if len(winners) >= 6:
        seq = winners[-6:]
        if seq[0] == seq[1] and seq[2] == seq[3] and seq[4] == seq[5] and seq[0] == seq[4] and seq[1] == seq[5]:
            return 'degrau'

    # quebra controlada: duas fracas seguidas e depois forte
    if 'strength' in history.columns:
        s = strengths
        if len(s) >= 3 and s[-1] == 'forte' and s[-2] == 'fraco' and s[-3] == 'fraco':
            return 'quebra controlada'

    return 'indefinido'

def analyze_tendency(history: pd.DataFrame) -> dict:
    """Analisa o histórico e retorna: padrão, probabilidades (RED/BLUE/TIE), sugestão e confiança."""
    if history.empty:
        return {'pattern': 'indefinido', 'prob_red': 0.0, 'prob_blue': 0.0, 'prob_tie': 0.0,
                'suggestion': 'aguardar', 'confidence': 0.0}

    last = history.iloc[-1]
    pattern = pattern_of_sequence(history)

    prob = {'red': 0.0, 'blue': 0.0, 'tie': 0.0}
    confidence = 0.0

    # Base: considerar força da rodada (forte favorece repetição do vencedor, fraco favorece quebra)
    last_strength = last.get('strength', 'indefinido')
    last_winner = last['winner']

    # Heurísticas principais (metodologia inspirada no original, mas usando ambas cartas)
    if last_strength == 'forte':
        # forte tende a repetir o vencedor com alta chance
        repeat_prob = 0.78
        other_prob = 1 - repeat_prob
        if last_winner == 'red':
            prob['red'] = repeat_prob
            prob['blue'] = other_prob * 0.95
        elif last_winner == 'blue':
            prob['blue'] = repeat_prob
            prob['red'] = other_prob * 0.95
        else:  # tie
            prob['tie'] = 0.7
            prob['red'] = 0.15
            prob['blue'] = 0.15
        prob['tie'] = prob.get('tie', 0.0)
        confidence = 0.78
    elif last_strength == 'medio':
        # média = sinal neutro, leve preferência ao vencedor anterior se padrão apoiar
        base = 0.62
        if last_winner == 'red':
            prob['red'] = base
            prob['blue'] = 1 - base - 0.03
        elif last_winner == 'blue':
            prob['blue'] = base
            prob['red'] = 1 - base - 0.03
        else:
            prob['tie'] = 0.04
            prob['red'] = 0.48
            prob['blue'] = 0.48
        prob['tie'] = prob.get('tie', 0.03)
        confidence = 0.58
    elif last_strength == 'fraco':
        # fraco tende a quebrar — probabilidade de inversão alta
        break_prob = 0.74
        if last_winner == 'red':
            prob['blue'] = break_prob
            prob['red'] = max(0.0, 1 - break_prob - 0.03)
        elif last_winner == 'blue':
            prob['red'] = break_prob
            prob['blue'] = max(0.0, 1 - break_prob - 0.03)
        else:
            prob['tie'] = 0.06
            prob['red'] = 0.47
            prob['blue'] = 0.47
        prob['tie'] = prob.get('tie', 0.04)
        confidence = 0.72
    else:
        # indefinido
        prob = {'red': 0.49, 'blue': 0.49, 'tie': 0.02}
        confidence = 0.4

    # Ajustes por padrão detectado
    if pattern == 'repetição':
        if last_winner == 'red':
            prob['red'] = min(0.97, prob['red'] + 0.12)
        elif last_winner == 'blue':
            prob['blue'] = min(0.97, prob['blue'] + 0.12)
        confidence = max(confidence, 0.78)
    elif pattern == 'alternância':
        # favorece inversão
        if last_winner == 'red':
            prob['blue'] = max(prob['blue'], 0.58)
            prob['red'] = 1 - prob['blue'] - prob.get('tie', 0)
        elif last_winner == 'blue':
            prob['red'] = max(prob['red'], 0.58)
            prob['blue'] = 1 - prob['red'] - prob.get('tie', 0)
        confidence = max(confidence, 0.62)
    elif pattern == 'degrau':
        if len(history) >= 2 and history.iloc[-2]['winner'] == last_winner:
            if last_winner == 'red':
                prob['red'] = max(prob['red'], 0.72)
            else:
                prob['blue'] = max(prob['blue'], 0.72)
            confidence = max(confidence, 0.72)
    elif pattern == 'quebra controlada':
        prob['tie'] = max(prob.get('tie', 0.03), 0.06)
        if last_winner == 'red':
            prob['red'] = max(prob['red'], 0.62)
        else:
            prob['blue'] = max(prob['blue'], 0.62)
        confidence = max(confidence, 0.68)

    # Ajuste por presença de tie recente (reserva)
    recent_ties = history['winner'].tail(4).tolist().count('tie')
    if recent_ties >= 1:
        # tie reseta confiança geral
        confidence = min(0.85, confidence * 0.9)
        # pequena elevação de prob de tie
        prob['tie'] = max(prob.get('tie', 0.03), 0.03 + recent_ties * 0.02)

    # Normaliza e converte para porcentagem
    total = prob['red'] + prob['blue'] + prob.get('tie', 0.0)
    if total <= 0:
        prob = {'red': 0.49, 'blue': 0.49, 'tie': 0.02}
        total = 1.0
    for k in prob:
        prob[k] = prob[k] / total
    prob_pct = {k: round(v * 100, 1) for k, v in prob.items()}

    # Sugestão baseada na maior probabilidade e confiança mínima
    sorted_probs = sorted(prob_pct.items(), key=lambda x: x[1], reverse=True)
    top_label, top_val = sorted_probs[0]
    suggestion = 'aguardar'
    if top_val >= 60 or confidence >= 0.7:
        if top_label == 'red':
            suggestion = 'apostar RED (🔴)'
        elif top_label == 'blue':
            suggestion = 'apostar BLUE (🔵)'
        else:
            suggestion = 'apostar TIE (🟡)'

    return {
        'pattern': pattern,
        'prob_red': prob_pct['red'],
        'prob_blue': prob_pct['blue'],
        'prob_tie': prob_pct['tie'],
        'suggestion': suggestion,
        'confidence': round(confidence * 100, 1)
    }

def manipulation_level(history: pd.DataFrame) -> int:
    """Deriva um nível de manipulação (1-9) por heurísticas: mais instabilidade -> nível maior."""
    if history.empty:
        return 1

    vals_blue = history['value_blue'].tolist()
    vals_red = history['value_red'].tolist()
    winners = history['winner'].tolist()
    strengths = history['strength'].tolist()

    score = 0.0

    # conta sequências de rodadas fracas consecutivas (indicador de manipulação)
    weak_runs = 0
    run = 0
    for s in strengths:
        if s == 'fraco':
            run += 1
        else:
            if run >= 2:
                weak_runs += 1
            run = 0
    if run >= 2:
        weak_runs += 1
    score += weak_runs * 1.4

    # alternações frequentes aumentam suspeita
    alternations = sum(1 for i in range(1, len(winners)) if winners[i] != winners[i - 1] and winners[i] != 'tie' and winners[i-1] != 'tie')
    alternation_rate = alternations / max(1, (len(winners) - 1))
    score += alternation_rate * 3.2

    # vitórias por cartas baixas repetidas aumentam suspeita
    low_win_count = 0
    for idx, w in enumerate(winners):
        if w == 'red':
            if classify_card(history.iloc[idx]['red_card']) == 'baixa' and classify_card(history.iloc[idx]['blue_card']) != 'alta':
                low_win_count += 1
        elif w == 'blue':
            if classify_card(history.iloc[idx]['blue_card']) == 'baixa' and classify_card(history.iloc[idx]['red_card']) != 'alta':
                low_win_count += 1
    low_rate = low_win_count / max(1, len(winners))
    score += low_rate * 3.0

    # presença de muitos ties reduz nível (mas cuidado)
    tie_count = winners.count('tie')
    tie_rate = tie_count / max(1, len(winners))
    score -= tie_rate * 1.5

    # normaliza por média de cartas altas (mais altas -> menos manipulação)
    high_count = sum(1 for i in range(len(vals_blue)) if vals_blue[i] >= 11 or vals_red[i] >= 11)
    high_rate = high_count / max(1, len(vals_blue))
    score -= high_rate * 2.0

    # mapear score para 1..9
    level = int(min(9, max(1, round(score))))
    return level

# ----------------------------- Estado inicial -----------------------------
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'timestamp', 'blue_card', 'value_blue', 'value_class_blue',
        'red_card', 'value_red', 'value_class_red',
        'winner', 'diff', 'strength'
    ])

# ----------------------------- Funções para manipular histórico -----------------------------
def add_result_cards(blue_card: str, red_card: str):
    """Adiciona resultado usando as duas cartas selecionadas (registro imediato)."""
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

def add_result_tie(blue_card: str = None, red_card: str = None):
    """Registra empate. Permite opcionalmente guardar as cartas (se fornecidas) ou '0'."""
    now = datetime.now()
    vb = card_value(blue_card) if blue_card else 0
    vr = card_value(red_card) if red_card else 0
    vc_blue = classify_card(blue_card) if blue_card else 'indefinido'
    vc_red = classify_card(red_card) if red_card else 'indefinido'
    diff = abs(vb - vr)
    strength = strength_of_duel(vb, vr) if vb and vr else 'indefinido'
    new_row = {
        'timestamp': now,
        'blue_card': blue_card if blue_card else '0',
        'value_blue': vb,
        'value_class_blue': vc_blue,
        'red_card': red_card if red_card else '0',
        'value_red': vr,
        'value_class_red': vc_red,
        'winner': 'tie',
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

# ----------------------------- Sidebar (config e export) -----------------------------
with st.sidebar:
    st.header('Controles & Export')
    if st.button('Resetar Histórico'):
        reset_history()
    st.write('---')
    st.markdown('Exportar / Configurações')
    csv_data = st.session_state.history.to_csv(index=False)
    st.download_button('Exportar histórico (CSV)', data=csv_data, file_name='history_football_studio.csv')
    st.write('---')
    show_timestamps = st.checkbox('Mostrar timestamps', value=False)
    show_confidence_bar = st.checkbox('Mostrar barras de confiança', value=True)
    st.write('---')
    st.caption('Registro: clique no botão da cor para gravar imediatamente (sem confirmar).')

# ----------------------------- Inserção horizontal (SEM CONFIRMAR) -----------------------------
st.subheader("Inserir Resultado (Horizontal • Sem Confirmar)")

# Layout: select Blue | button Blue | select Red | button Red | Tie button (tudo em uma linha)
col_blue_card, col_blue_btn, spacer1, col_red_card, col_red_btn, spacer2, col_tie_btn = st.columns([2, 1, 0.2, 2, 1, 0.2, 0.8])

with col_blue_card:
    blue_card = st.selectbox(
        "Carta BLUE",
        options=list(CARD_MAP.keys()),
        index=0,
        key="blue_card_select_full"
    )

with col_blue_btn:
    if st.button("🔵 BLUE", key="btn_blue_full"):
        # grava o resultado usando ambas cartas selecionadas
        add_result_cards(blue_card, st.session_state.get('red_card_select_full', list(CARD_MAP.keys())[0]))

with col_red_card:
    red_card = st.selectbox(
        "Carta RED",
        options=list(CARD_MAP.keys()),
        index=0,
        key="red_card_select_full"
    )

with col_red_btn:
    if st.button("🔴 RED", key="btn_red_full"):
        add_result_cards(st.session_state.get('blue_card_select_full', list(CARD_MAP.keys())[0]), red_card)

with col_tie_btn:
    # botão tie grava empate; utiliza as cartas selecionadas se o usuário quiser
    if st.button("🟡 TIE", key="btn_tie_full"):
        # registra tie com as cartas selecionadas (opcional)
        add_result_tie(st.session_state.get('blue_card_select_full', None), st.session_state.get('red_card_select_full', None))

st.write('---')

# ----------------------------- Histórico (exibição em linhas) -----------------------------
st.subheader('Histórico (inserção manual por botões • visual horizontal)')

history = st.session_state.history.copy()

# Limita a exibição total a MAX_COLS * MAX_LINES para evitar UI gigante
if len(history) > MAX_COLS * MAX_LINES:
    history = history.tail(MAX_COLS * MAX_LINES).reset_index(drop=True)

if history.empty:
    st.info('Sem resultados ainda. Use os botões acima para inserir resultados.')
else:
    # dividir em linhas de até MAX_COLS
    rows = [history.iloc[i:i + MAX_COLS] for i in range(0, len(history), MAX_COLS)]
    for row_df in rows:
        cols = st.columns(MAX_COLS)
        for c_idx in range(MAX_COLS):
            with cols[c_idx]:
                if c_idx < len(row_df):
                    item = row_df.iloc[c_idx]
                    if item['winner'] == 'red':
                        label = f"🔴 {item['red_card']} vs {item['blue_card']} ({item['strength']})"
                    elif item['winner'] == 'blue':
                        label = f"🔵 {item['blue_card']} vs {item['red_card']} ({item['strength']})"
                    else:
                        label = f"🟡 TIE {item['blue_card']}|{item['red_card']} ({item['strength']})"
                    if show_timestamps:
                        st.caption(str(item['timestamp']))
                    st.markdown(f"**{label}**")
                else:
                    st.write('')

# ----------------------------- Análise e Previsões -----------------------------
st.subheader('Análise e Previsão (com base nas 2 cartas)')

analysis = analyze_tendency(st.session_state.history)
level = manipulation_level(st.session_state.history)

colA, colB = st.columns([2, 1])
with colA:
    st.markdown('**Padrão detectado:** ' + analysis['pattern'].capitalize())
    st.markdown('**Nível de manipulação estimado (1–9):** ' + str(level))
    st.markdown('**Sugestão:** ' + analysis['suggestion'])
    st.markdown(f"**Confiança do modelo:** {analysis['confidence']} %")

    st.markdown('**Probabilidades estimadas (heurísticas):**')
    if show_confidence_bar:
        st.progress(int(min(100, max(0, analysis['confidence']))))
    pb = st.columns(3)
    with pb[0]:
        st.metric('🔴 RED', f"{analysis['prob_red']} %")
    with pb[1]:
        st.metric('🔵 BLUE', f"{analysis['prob_blue']} %")
    with pb[2]:
        st.metric('🟡 TIE', f"{analysis['prob_tie']} %")

with colB:
    st.markdown('**Resumo das últimas jogadas (últimas 10):**')
    display_df = st.session_state.history.tail(10).reset_index(drop=True)
    st.dataframe(display_df)

st.markdown('---')
st.subheader('Interpretação dos sinais (por valor e confronto de cartas)')
st.markdown(''' 
- A análise considera ambas as cartas da rodada (BLUE vs RED).
- **Diferença pequena (≤2)**: rodada fraca — alta probabilidade de quebra/alternância.
- **Diferença média (3-4)**: rodada neutra — observe padrão.
- **Diferença grande (≥5)**: rodada forte — favorece repetição do vencedor.
- Classificação por carta:
  - A,K,Q,J = ALTA (tendência a repetição se vencem)
  - 10,9,8 = MÉDIA (zona de transição)
  - 7–2 = BAIXA (sinal de instabilidade se vencer)
''')

st.subheader('Estratégia operacional (passo a passo)')
st.markdown('''
1. Sempre registre as duas cartas (BLUE e RED).  
2. Veja a força do duelo (fraco/médio/forte).  
3. Identifique o padrão ativo: repetição, alternância, degrau, ou quebra controlada.  
4. Só entre em aposta quando a sugestão e a confiança estiverem alinhadas (por exemplo, prob >= 60% ou confiança >= 70%).  
5. Em casos de rodadas fracas, priorize aguardar confirmação (não aposte cegamente).  
6. Gestão de banca: stake proporcional ao nível de confiança e nível de manipulação estimado.
''')

# ----------------------------- Ferramentas avançadas -----------------------------
st.markdown('---')
st.header('Ferramentas avançadas')

colx, coly = st.columns(2)
with colx:
    if st.button('Auto-análise (mostrar estrutura de análise)'):
        st.json(analysis)
with coly:
    if st.button('Exportar relatório simples (TXT)'):
        txt = ""
        txt += "Football Studio Analyzer - Relatório\n"
        txt += f"Gerado em: {datetime.now()}\n"
        txt += f"Padrão: {analysis['pattern']}\n"
        txt += f"Nível de manipulação: {level}\n"
        txt += f"Sugestão: {analysis['suggestion']}\n"
        txt += f"Probabilidades: RED {analysis['prob_red']}%, BLUE {analysis['prob_blue']}%, TIE {analysis['prob_tie']}%\n"
        txt += "\nÚltimas 10 jogadas:\n"
        txt += st.session_state.history.tail(10).to_string(index=False)
        st.download_button('Baixar relatório', data=txt, file_name='relatorio_football_studio.txt')

st.markdown('---')
st.caption('Este sistema aplica heurísticas e metodologia conforme solicitado. As probabilidades são estimativas heurísticas e não garantem lucro. Aposte com responsabilidade.')

# ----------------------------- EOF -----------------------------
