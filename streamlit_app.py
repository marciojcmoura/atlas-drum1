"""
streamlit_app.py — Interface web do ATLAS (Accelerated Testing & Lifetime
Analysis System), modulo do ecossistema DRUM, implementando o modelo
Bayesiano de Van Dorp & Mazzuchi (2004) via o binario C++ validado em cpp/
(ver PosteriorAproximation.cpp, LikelihoodFunction.cpp, PriorDistribution.cpp).

Identidade visual alinhada ao ecossistema DRUM / Risk Management (CEERMA-UFPE
/ Petrobras): paleta azul royal + azul-petroleo, cards brancos arredondados,
tabelas com cabecalho uppercase cinza -- ver detalhes no CSS injetado abaixo.
"""

import os

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from alt_io import (
    AltRunError,
    TestItem,
    build_input_text,
    parse_output,
    run_alt_cli,
    summarize,
)
from build_cli import ensure_binary

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(APP_DIR, "cpp")

# --------------------------------------------------------------------------
# Paleta / identidade visual DRUM (ver memoria "drum-visual-identity")
# --------------------------------------------------------------------------
DRUM = {
    "blue_primary": "#1F4E9F",
    "blue_sidebar": "#2F4CDD",
    "teal": "#155E75",
    "bg_page": "#F8F9FA",
    "bg_card": "#FFFFFF",
    "text_body": "#1A1A1A",
    "text_secondary": "#6B7280",
    "border": "#E5E7EB",
    "badge_bg": "#F1F3F5",
}

ENV_COLORS = ["#1F4E9F", "#155E75", "#0F6E56", "#993C1D", "#712B13", "#534AB7", "#993556"]

PAPER_EXAMPLE = {
    "K": 5,
    "N": 12,
    "S": 5,
    "beta_weibull": 1.0,
    "time_mission": 1000.0,
    "index_environment_use": 1,
    "reliability_quantile_environment_use": 0.26,
    "samples": 300000,
    "samples_burn_in": 30000,
    "n_skip": 2,
    "intervals": 200,
    "censored_flag": 1,
    "reliability_prior": [0.95, 0.90, 0.56, 0.24, 0.02],
}

_ENV_PATTERNS = {
    (1, 2, 3): [1, 2, 3, 4, 5],
    (4, 5, 6): [3, 3, 3, 3, 3],
    (7, 8, 9): [5, 4, 3, 2, 1],
    (10, 11, 12): [2, 3, 1, 5, 4],
}
_T_BOUNDARIES = [0, 720, 1440, 2160, 2880, 3600]
_RAMP = [0, 6, 6, 6, 6]
_FAILURE_TIMES = [3598, 2153, 716, 2879, 3600, 3600, 3596, 3600, 3592, 2157, 3591, 1438]


def paper_example_items():
    items = []
    for j in range(1, 13):
        seq = next(pattern for items_range, pattern in _ENV_PATTERNS.items() if j in items_range)
        items.append(
            {
                "item": j,
                **{f"env_{i+1}": seq[i] for i in range(5)},
                **{f"t_{i}": _T_BOUNDARIES[i] for i in range(6)},
                **{f"ramp_{i+1}": _RAMP[i] for i in range(5)},
                "failure_time": _FAILURE_TIMES[j - 1],
            }
        )
    return pd.DataFrame(items)


# --------------------------------------------------------------------------
# Setup da pagina + CSS
# --------------------------------------------------------------------------
st.set_page_config(page_title="ATLAS — DRUM", page_icon="📈", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {DRUM['bg_page']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {DRUM['blue_sidebar']};
    }}
    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.25);
    }}
    .drum-logo {{
        font-weight: 700;
        font-size: 26px;
        letter-spacing: 2px;
        color: #FFFFFF;
        margin-bottom: 0px;
    }}
    .drum-logo-sub {{
        font-size: 12px;
        letter-spacing: 1px;
        color: rgba(255,255,255,0.75);
        text-transform: uppercase;
        margin-top: -6px;
    }}
    .app-header-title {{
        font-weight: 700;
        font-size: 28px;
        color: {DRUM['text_body']};
        margin-bottom: 0px;
    }}
    .app-header-sub {{
        font-size: 14px;
        color: {DRUM['text_secondary']};
        margin-top: 2px;
    }}
    .app-breadcrumb {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {DRUM['teal']};
        font-weight: 600;
        margin-bottom: 4px;
    }}
    div[data-testid="stMetric"] {{
        background-color: {DRUM['bg_card']};
        border: 1px solid {DRUM['border']};
        border-radius: 12px;
        padding: 14px 16px;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {DRUM['blue_primary']};
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        color: {DRUM['text_secondary']};
    }}
    .env-badge {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        background-color: {DRUM['badge_bg']};
        color: {DRUM['text_body']};
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="drum-logo">ATLAS</div>', unsafe_allow_html=True)
    st.markdown('<div class="drum-logo-sub">Accelerated Testing &amp; Lifetime Analysis System</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Ecossistema DRUM**")
    st.caption("Risk Management · CEERMA-UFPE · Petrobras")
    st.markdown("---")
    st.markdown(
        "Inferência Bayesiana para testes de vida acelerados, com prior "
        "Dirichlet ordenada e MCMC (Metropolis-Hastings em bloco), conforme "
        "Van Dorp & Mazzuchi (2004), *JSPI* 119, 55–74."
    )
    st.markdown("---")
    if st.button("↻ Carregar exemplo do artigo", use_container_width=True):
        st.session_state["_load_example"] = True
        st.rerun()

# --------------------------------------------------------------------------
# Estado inicial
# --------------------------------------------------------------------------
def init_state():
    for k, v in PAPER_EXAMPLE.items():
        if k == "reliability_prior":
            continue
        st.session_state.setdefault(k, v)
    for i, val in enumerate(PAPER_EXAMPLE["reliability_prior"]):
        st.session_state.setdefault(f"prior_{i}", val)
    st.session_state.setdefault("test_plan_df", paper_example_items())


init_state()

# NOTA IMPORTANTE sobre Streamlit + session_state: um widget nao pode receber
# um parametro "value=" explicito quando sua "key" ja possui um valor em
# st.session_state (isso levanta StreamlitAPIException a partir da segunda
# execucao do script em diante). Por isso, TODOS os widgets abaixo usam
# apenas "key=" (nunca "value="), e o valor inicial/"exemplo do artigo" e
# sempre escrito em st.session_state ANTES do widget correspondente ser
# criado (aqui, ou em init_state() acima) -- nunca depois.
if st.session_state.pop("_load_example", False):
    for k, v in PAPER_EXAMPLE.items():
        if k == "reliability_prior":
            continue
        st.session_state[k] = v
    for i, val in enumerate(PAPER_EXAMPLE["reliability_prior"]):
        st.session_state[f"prior_{i}"] = val
    st.session_state["test_plan_df"] = paper_example_items()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="app-breadcrumb">DRUM · Risk Management · ATLAS</div>', unsafe_allow_html=True)
st.markdown('<div class="app-header-title">ATLAS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-header-sub">Accelerated Testing &amp; Lifetime Analysis System — modelo Bayesiano '
    "exponencial geral para inferência em testes de vida acelerados, Van Dorp &amp; Mazzuchi (2004)</div>",
    unsafe_allow_html=True,
)
st.write("")

tab_config, tab_plan, tab_run, tab_manual = st.tabs(
    ["1 · Configuração", "2 · Plano de teste", "3 · Executar & Resultados", "4 · Manual"]
)

# --------------------------------------------------------------------------
# Tab 1 — Configuração
# --------------------------------------------------------------------------
with tab_config:
    st.markdown("#### Ambientes e missão")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Número de ambientes (K)", min_value=2, max_value=10, step=1, key="K")
    with c2:
        st.number_input("Número de itens testados (N)", min_value=1, max_value=200, step=1, key="N")
    with c3:
        st.number_input("Número de passos de estresse (S)", min_value=1, max_value=10, step=1, key="S")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Forma Weibull (β)", min_value=0.01, key="beta_weibull")
    with c2:
        st.number_input("Tempo de missão (τ)", min_value=0.01, key="time_mission")
    with c3:
        censored_label = st.selectbox(
            "Tipo de dado",
            options=["Tipo I (censura fixa)", "Intervalar"],
            index=0 if st.session_state["censored_flag"] == 1 else 1,
        )
        st.session_state["censored_flag"] = 1 if censored_label == "Tipo I (censura fixa)" else 0

    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Ambiente de uso nominal (índice)", min_value=1, max_value=int(st.session_state["K"]),
            step=1, key="index_environment_use"
        )
    with c2:
        st.number_input(
            "Confiabilidade no quantil 5% (uso nominal)", min_value=0.0, max_value=1.0,
            key="reliability_quantile_environment_use"
        )

    st.markdown("#### Confiabilidade elicitada a priori, por ambiente")
    st.caption("Estimativa de confiabilidade no tempo de missão para cada ambiente (do menos ao mais severo).")
    K = int(st.session_state["K"])
    cols = st.columns(K)
    new_prior = []
    for i, c in enumerate(cols):
        st.session_state.setdefault(f"prior_{i}", 0.5)
        with c:
            val = st.number_input(
                f"Ambiente {i + 1}", min_value=0.0, max_value=1.0,
                key=f"prior_{i}", format="%.3f"
            )
            new_prior.append(val)
    st.session_state["reliability_prior"] = new_prior

    st.markdown("#### Amostragem MCMC")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Iterações totais", min_value=1000, step=10000, key="samples")
    with c2:
        st.number_input("Burn-in", min_value=100, step=1000, key="samples_burn_in")
    with c3:
        st.number_input("Thinning (nSkip)", min_value=1, key="n_skip")
    with c4:
        st.number_input("Faixas do CDF (intervals)", min_value=10, key="intervals")
    st.caption(
        "Mais iterações ⇒ resultado mais preciso, porém mais lento. "
        "A largura do passo da proposta se ajusta automaticamente durante o burn-in."
    )

# --------------------------------------------------------------------------
# Tab 2 — Plano de teste
# --------------------------------------------------------------------------
with tab_plan:
    st.markdown("#### Plano de teste (um item por linha)")
    st.caption(
        "env_i: índice do ambiente (1..K) no passo i · t_i: marco de tempo acumulado do passo i "
        "(t_0 = 0) · ramp_i: tempo de rampa no início do passo i · failure_time: tempo de falha "
        "ou censura do item."
    )

    S = int(st.session_state["S"])
    N = int(st.session_state["N"])
    df = st.session_state["test_plan_df"]

    expected_cols = (
        ["item"]
        + [f"env_{i+1}" for i in range(S)]
        + [f"t_{i}" for i in range(S + 1)]
        + [f"ramp_{i+1}" for i in range(S)]
        + ["failure_time"]
    )
    if list(df.columns) != expected_cols or len(df) != N:
        rows = []
        for j in range(1, N + 1):
            row = {"item": j}
            row.update({f"env_{i+1}": 1 for i in range(S)})
            row.update({f"t_{i}": 0.0 for i in range(S + 1)})
            row.update({f"ramp_{i+1}": 0.0 for i in range(S)})
            row["failure_time"] = 0.0
            rows.append(row)
        df = pd.DataFrame(rows, columns=expected_cols)
        st.session_state["test_plan_df"] = df

    edited = st.data_editor(
        st.session_state["test_plan_df"],
        num_rows="fixed",
        use_container_width=True,
        disabled=["item"],
        key="test_plan_editor",
    )
    st.session_state["test_plan_df"] = edited

# --------------------------------------------------------------------------
# Tab 3 — Executar & Resultados
# --------------------------------------------------------------------------
with tab_run:
    st.markdown("#### Executar o modelo")
    run_clicked = st.button("▶ Executar análise", type="primary")

    if run_clicked:
        try:
            with st.spinner("Compilando o motor de cálculo (primeira execução pode levar alguns segundos)..."):
                binary_path = ensure_binary(CPP_DIR)

            K = int(st.session_state["K"])
            N = int(st.session_state["N"])
            S = int(st.session_state["S"])
            df = st.session_state["test_plan_df"]

            items = []
            for _, row in df.iterrows():
                items.append(
                    TestItem(
                        envs=[int(row[f"env_{i+1}"]) for i in range(S)],
                        times=[float(row[f"t_{i}"]) for i in range(S + 1)],
                        ramps=[float(row[f"ramp_{i+1}"]) for i in range(S)],
                        failure_time=float(row["failure_time"]),
                    )
                )

            input_text = build_input_text(
                K=K,
                N=N,
                S=S,
                beta_weibull=float(st.session_state["beta_weibull"]),
                time_mission=float(st.session_state["time_mission"]),
                index_environment_use=int(st.session_state["index_environment_use"]),
                reliability_quantile_environment_use=float(st.session_state["reliability_quantile_environment_use"]),
                samples=int(st.session_state["samples"]),
                samples_burn_in=int(st.session_state["samples_burn_in"]),
                n_skip=int(st.session_state["n_skip"]),
                intervals=int(st.session_state["intervals"]),
                censored_flag=int(st.session_state["censored_flag"]),
                reliability_prior=[float(x) for x in st.session_state["reliability_prior"]],
                items=items,
            )

            with st.spinner("Rodando MCMC (Metropolis-Hastings em bloco)..."):
                output_text = run_alt_cli(binary_path, input_text, timeout_s=300)

            st.session_state["last_output_text"] = output_text
            st.session_state["last_envs"] = parse_output(output_text)
            st.success("Análise concluída.")

        except AltRunError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro inesperado: {e}")

    envs = st.session_state.get("last_envs")
    if envs:
        st.markdown("#### Resumo por ambiente")
        summary_rows = []
        for env_idx in sorted(envs):
            post_rel = envs[env_idx].get("posterior", {}).get("reliability")
            prior_rel = envs[env_idx].get("prior", {}).get("reliability")
            s_post = summarize(post_rel)
            s_prior = summarize(prior_rel)
            summary_rows.append(
                {
                    "Ambiente": env_idx,
                    "Confiabilidade priori (média)": s_prior["mean"],
                    "Confiabilidade posteriori (média)": s_post["mean"],
                    "Confiabilidade posteriori (mediana)": s_post["median"],
                    "Confiabilidade posteriori (moda)": s_post["mode"],
                }
            )
        summary_df = pd.DataFrame(summary_rows).round(4)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("#### Distribuição da confiabilidade — priori vs. posteriori")
        env_choice = st.selectbox(
            "Ambiente", options=sorted(envs), format_func=lambda e: f"Ambiente {e}"
        )
        prior_df = envs[env_choice].get("prior", {}).get("reliability")
        post_df = envs[env_choice].get("posterior", {}).get("reliability")

        chart_rows = []
        if prior_df is not None:
            for _, r in prior_df.iterrows():
                chart_rows.append({"Confiabilidade": r["value"], "CDF": r["cdf"], "Distribuição": "Priori"})
        if post_df is not None:
            for _, r in post_df.iterrows():
                chart_rows.append({"Confiabilidade": r["value"], "CDF": r["cdf"], "Distribuição": "Posteriori"})

        if chart_rows:
            chart_df = pd.DataFrame(chart_rows)
            chart = (
                alt.Chart(chart_df)
                .mark_line(point=False)
                .encode(
                    x=alt.X("Confiabilidade:Q", title="Confiabilidade no tempo de missão"),
                    y=alt.Y("CDF:Q", title="CDF"),
                    color=alt.Color(
                        "Distribuição:N",
                        scale=alt.Scale(domain=["Priori", "Posteriori"], range=[DRUM["text_secondary"], DRUM["blue_primary"]]),
                    ),
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

        st.download_button(
            "⬇ Baixar saída completa (.txt)",
            data=st.session_state["last_output_text"],
            file_name="atlas_resultado.txt",
            mime="text/plain",
        )
    else:
        st.info("Configure os parâmetros nas abas anteriores e clique em **Executar análise**.")

# --------------------------------------------------------------------------
# Tab 4 — Manual
# --------------------------------------------------------------------------
with tab_manual:
    manual_path = os.path.join(APP_DIR, "manual.html")
    with open(manual_path, "r", encoding="utf-8") as f:
        manual_html = f.read()
    components.html(manual_html, height=1500, scrolling=True)
