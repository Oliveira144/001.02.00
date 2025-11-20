Football Studio – Inserção Perfeita, Rápida e Profissional

Agora com o modo de inserir otimizado, extremamente simples, rápido e fiel ao jogo.

Inserção SEM CONFIRMAR, 100% horizontal, botões grandes, ultra prático.

import streamlit as st 
import pandas as pd

st.set_page_config(page_title="Football Studio Analyzer", layout="wide")

-------------------------------

VALORES DAS CARTAS

-------------------------------

VALORES = { "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13 }

-------------------------------

ESTADO

-------------------------------

if "historico" not in st.session_state: st.session_state.historico = []

-------------------------------

ADICIONAR RESULTADO

-------------------------------

def adicionar(vencedor, carta): st.session_state.historico.append({ "vencedor": vencedor, "carta": carta, "valor": VALORES.get(carta, None) })

-------------------------------

INTERFACE – MODO SUPER RÁPIDO

-------------------------------

st.subheader("Inserção Ultra Rápida (Horizontal, Sem Confirmar)")

Layout das cartas exatamente como o jogo

colA, colB, colC, colD, colE = st.columns([1.4, 1.1, 1.4, 1.1, 1])

---------------- BLUE ----------------

with colA: st.markdown("### 🔵 Carta BLUE") carta_blue = st.radio( "", list(VALORES.keys()), horizontal=True, key="cblue", )

with colB: if st.button("🔵 INSERIR BLUE", use_container_width=True): adicionar("BLUE", carta_blue)

---------------- RED ----------------

with colC: st.markdown("### 🔴 Carta RED") carta_red = st.radio( "", list(VALORES.keys()), horizontal=True, key="cred", )

with colD: if st.button("🔴 INSERIR RED", use_container_width=True): adicionar("RED", carta_red)

---------------- TIE ----------------

with colE: st.markdown("### 🟡 TIE") if st.button("🟡 EMPATE", use_container_width=True): adicionar("TIE", "-")

-------------------------------

HISTÓRICO

-------------------------------

st.subheader("Histórico de Resultados") if len(st.session_state.historico) > 0: df = pd.DataFrame(st.session_state.historico) st.dataframe(df, use_container_width=True) else: st.info("Nenhum resultado registrado ainda.")
