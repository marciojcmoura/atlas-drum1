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
    "green_prior": "#2F855A",  # tonalidade verde p/ graficos que so tem priori (Estagio 1)
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


# --------------------------------------------------------------------------
# Modelo de "Perfis de teste" + "Atribuição de itens" (aba 2, redesenho)
#
# Em vez de configurar env/t/ramp item a item, o usuário define poucos
# PERFIS reutilizáveis (sequência de passos de estresse) e depois só diz,
# para cada item testado, qual perfil ele seguiu e seu tempo de
# falha/censura. Isso é convertido para a lista de TestItem (envs/times/
# ramps/failure_time) exigida por build_input_text na hora de executar.
# --------------------------------------------------------------------------
_STEP_DURATION = 720.0  # duração default de cada passo (h), igual ao artigo

# Monta os 4 perfis-exemplo (A/B/C/D) diretamente dos padrões do artigo.
DEFAULT_PROFILES = {}
for name, seq in zip(["A", "B", "C", "D"], _ENV_PATTERNS.values()):
    DEFAULT_PROFILES[name] = [
        {"env": int(seq[i]), "dur": float(_STEP_DURATION), "ramp": float(_RAMP[i])}
        for i in range(len(seq))
    ]

_PROFILE_FOR_ITEM = {}
for items_range, _seq in _ENV_PATTERNS.items():
    profile_name = ["A", "B", "C", "D"][list(_ENV_PATTERNS.keys()).index(items_range)]
    for j in items_range:
        _PROFILE_FOR_ITEM[j] = profile_name


def default_assignment_df():
    rows = []
    for j in range(1, 13):
        ft = _FAILURE_TIMES[j - 1]
        rows.append(
            {
                "item": j,
                "perfil": _PROFILE_FOR_ITEM[j],
                "tempo_falha_censura": float(ft),
                "censurado": bool(ft == _T_BOUNDARIES[-1]),
            }
        )
    return pd.DataFrame(rows)


def ensure_profiles_state(S: int, K: int):
    """Garante que st.session_state['profiles'] existe e que cada perfil tem
    exatamente S passos, com env dentro de 1..K."""
    profiles = st.session_state.get("profiles")
    if not profiles:
        profiles = {name: [dict(step) for step in steps] for name, steps in DEFAULT_PROFILES.items()}
    fixed = {}
    for name, steps in profiles.items():
        steps = [dict(s) for s in steps]
        if len(steps) < S:
            last = dict(steps[-1]) if steps else {"env": 1, "dur": _STEP_DURATION, "ramp": 0.0}
            steps = steps + [dict(last) for _ in range(S - len(steps))]
        elif len(steps) > S:
            steps = steps[:S]
        for s in steps:
            s["env"] = int(min(max(int(s.get("env", 1)), 1), K))
            s["dur"] = float(s.get("dur", _STEP_DURATION))
            s["ramp"] = float(s.get("ramp", 0.0))
        fixed[name] = steps
    st.session_state["profiles"] = fixed


def ensure_assignment_state(N: int):
    """Garante que st.session_state['item_assignments'] tem N linhas e que
    toda referência a 'perfil' aponta para um perfil que ainda existe."""
    profile_names = list(st.session_state["profiles"].keys())
    df = st.session_state.get("item_assignments")
    if df is None:
        df = default_assignment_df()
    df = df.copy()
    if len(df) < N:
        extra = pd.DataFrame(
            [
                {
                    "item": len(df) + i + 1,
                    "perfil": profile_names[0],
                    "tempo_falha_censura": 0.0,
                    "censurado": False,
                }
                for i in range(N - len(df))
            ]
        )
        df = pd.concat([df, extra], ignore_index=True)
    elif len(df) > N:
        df = df.iloc[:N].reset_index(drop=True)
    df["item"] = range(1, N + 1)
    df["perfil"] = df["perfil"].apply(lambda p: p if p in profile_names else profile_names[0])
    st.session_state["item_assignments"] = df


def assemble_test_items(profiles: dict, assignment_df: pd.DataFrame) -> list:
    """Expande perfil + tempo de falha/censura de cada item em um TestItem
    completo (envs/times/ramps/failure_time), pronto para build_input_text."""
    items = []
    for _, row in assignment_df.iterrows():
        steps = profiles[row["perfil"]]
        envs = [int(s["env"]) for s in steps]
        times = [0.0]
        for s in steps:
            times.append(times[-1] + float(s["dur"]))
        ramps = [float(s["ramp"]) for s in steps]
        censurado = bool(row["censurado"])
        failure_time = times[-1] if censurado else float(row["tempo_falha_censura"])
        items.append(TestItem(envs=envs, times=times, ramps=ramps, failure_time=failure_time))
    return items


def build_profile_chart(steps: list, K: int, min_ramp_frac: float = 0.03):
    """Gráfico de degraus (Altair) do perfil de estresse: ambiente no eixo Y,
    tempo acumulado no eixo X. Segmentos de rampa usam uma largura MÍNIMA de
    exibição fixa (proporção do tempo total do perfil) mesmo quando o valor
    real da rampa é pequeno perto da duração total -- caso contrário a rampa
    fica visualmente indistinguível de um salto instantâneo. O valor real
    (em horas) é sempre mostrado como rótulo de texto sobre o trecho de
    rampa; a duração real (não distorcida) é o que efetivamente entra no
    cálculo -- a distorção é só para o desenho."""
    total_dur = sum(s["dur"] for s in steps) or 1.0
    ramp_display_w = min_ramp_frac * total_dur

    points = []
    labels = []
    x = 0.0
    prev_y = None
    for s in steps:
        y = s["env"]
        if prev_y is not None and s["ramp"] > 0:
            points.append((x, prev_y))
            x += ramp_display_w
            points.append((x, y))
            labels.append((x - ramp_display_w / 2, max(y, prev_y) + 0.18, f'rampa {s["ramp"]:g}h'))
        else:
            points.append((x, y))
        x_end = x + s["dur"]
        points.append((x_end, y))
        x = x_end
        prev_y = y

    line_df = pd.DataFrame(points, columns=["x", "y"])
    label_df = pd.DataFrame(labels, columns=["x", "y", "text"])

    line = (
        alt.Chart(line_df)
        .mark_line(interpolate="linear", strokeWidth=3, color=DRUM["blue_primary"])
        .encode(
            x=alt.X("x:Q", title="tempo acumulado (h) — rampas com largura mínima de exibição"),
            y=alt.Y(
                "y:Q",
                title="ambiente",
                scale=alt.Scale(domain=[0.4, K + 0.6]),
                axis=alt.Axis(tickMinStep=1),
            ),
        )
    )
    chart = line
    if not label_df.empty:
        text = (
            alt.Chart(label_df)
            .mark_text(fontSize=10, color=DRUM["text_secondary"], fontWeight="bold")
            .encode(x="x:Q", y="y:Q", text="text:N")
        )
        chart = line + text
    return chart.properties(height=220)


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
    st.session_state.setdefault(
        "profiles", {name: [dict(step) for step in steps] for name, steps in DEFAULT_PROFILES.items()}
    )
    st.session_state.setdefault("item_assignments", default_assignment_df())
    st.session_state.setdefault("plan_mode", "Tabela")


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
    st.session_state["profiles"] = {
        name: [dict(step) for step in steps] for name, steps in DEFAULT_PROFILES.items()
    }
    st.session_state["item_assignments"] = default_assignment_df()

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
# Tab 2 — Plano de teste (perfis de teste + atribuição de itens)
# --------------------------------------------------------------------------
with tab_plan:
    K = int(st.session_state["K"])
    S = int(st.session_state["S"])
    N = int(st.session_state["N"])

    ensure_profiles_state(S, K)
    ensure_assignment_state(N)

    st.markdown("#### 1 · Perfis de teste")
    st.caption(
        "Um perfil é uma sequência de passos de estresse (ambiente, duração e tempo de rampa) "
        "que pode ser reutilizada por vários itens — como os 4 planos (A/B/C/D) do artigo. "
        "Defina os perfis primeiro; depois, na seção 2, diga qual perfil cada item seguiu."
    )

    mode = st.radio(
        "Como configurar os perfis",
        options=["Tabela", "Upload Excel", "Construtor visual"],
        horizontal=True,
        key="plan_mode",
    )

    profile_names = list(st.session_state["profiles"].keys())

    # -- criação / remoção de perfis (comum aos 3 modos) --------------------
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    with cc1:
        new_profile_name = st.text_input(
            "Nome do novo perfil", key="new_profile_name", placeholder="ex.: E", label_visibility="collapsed"
        )
    with cc2:
        if st.button("+ criar perfil", use_container_width=True):
            name = st.session_state.get("new_profile_name", "").strip()
            if name and name not in st.session_state["profiles"]:
                st.session_state["profiles"][name] = [
                    {"env": 1, "dur": _STEP_DURATION, "ramp": 0.0} for _ in range(S)
                ]
                st.rerun()
    with cc3:
        if len(profile_names) > 1:
            remove_choice = st.selectbox(
                "Remover", options=["—"] + profile_names, label_visibility="collapsed", key="remove_profile_choice"
            )
            if remove_choice != "—" and st.button("Remover perfil", use_container_width=True):
                remaining = [p for p in profile_names if p != remove_choice]
                del st.session_state["profiles"][remove_choice]
                assign_df = st.session_state["item_assignments"]
                assign_df.loc[assign_df["perfil"] == remove_choice, "perfil"] = remaining[0]
                st.session_state["item_assignments"] = assign_df
                st.rerun()

    profile_names = list(st.session_state["profiles"].keys())

    if mode == "Tabela":
        sel_profile = st.selectbox("Perfil a editar", options=profile_names, key="table_profile_select")
        steps = st.session_state["profiles"][sel_profile]
        steps_df = pd.DataFrame(steps)
        steps_df.insert(0, "passo", range(1, S + 1))
        edited_steps = st.data_editor(
            steps_df,
            num_rows="fixed",
            use_container_width=True,
            disabled=["passo"],
            column_config={
                "env": st.column_config.NumberColumn("Ambiente", min_value=1, max_value=K, step=1),
                "dur": st.column_config.NumberColumn("Duração (h)", min_value=0.0),
                "ramp": st.column_config.NumberColumn("Rampa (h)", min_value=0.0),
            },
            key=f"steps_editor_{sel_profile}",
        )
        st.session_state["profiles"][sel_profile] = edited_steps[["env", "dur", "ramp"]].to_dict("records")
        st.altair_chart(build_profile_chart(st.session_state["profiles"][sel_profile], K), use_container_width=True)

    elif mode == "Upload Excel":
        st.caption(
            "Planilha com duas abas: **Perfis** (colunas: perfil, passo, env, dur, ramp) e "
            "**Atribuicao** (colunas: item, perfil, tempo_falha_censura, censurado)."
        )
        template_rows = []
        for name, steps in st.session_state["profiles"].items():
            for i, s in enumerate(steps, start=1):
                template_rows.append({"perfil": name, "passo": i, "env": s["env"], "dur": s["dur"], "ramp": s["ramp"]})
        template_perfis = pd.DataFrame(template_rows)
        template_assign = st.session_state["item_assignments"]

        import io as _io

        buf = _io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            template_perfis.to_excel(writer, sheet_name="Perfis", index=False)
            template_assign.to_excel(writer, sheet_name="Atribuicao", index=False)
        st.download_button(
            "⬇ Baixar template (com os dados atuais)",
            data=buf.getvalue(),
            file_name="atlas_plano_de_teste_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        uploaded = st.file_uploader("Carregar planilha preenchida (.xlsx)", type=["xlsx"], key="plan_excel_uploader")
        if uploaded is not None:
            try:
                sheets = pd.read_excel(uploaded, sheet_name=["Perfis", "Atribuicao"])
                perfis_df, assign_df = sheets["Perfis"], sheets["Atribuicao"]
                new_profiles = {}
                for name, group in perfis_df.sort_values("passo").groupby("perfil"):
                    new_profiles[name] = [
                        {"env": int(r["env"]), "dur": float(r["dur"]), "ramp": float(r["ramp"])}
                        for _, r in group.iterrows()
                    ]
                assign_df = assign_df.copy()
                assign_df["censurado"] = assign_df["censurado"].astype(bool)
                st.session_state["profiles"] = new_profiles
                st.session_state["item_assignments"] = assign_df
                st.success("Planilha carregada com sucesso.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Não foi possível ler a planilha: {e}")

    else:  # Construtor visual
        sel_profile = st.radio("Perfil a editar", options=profile_names, horizontal=True, key="visual_profile_select")
        steps = st.session_state["profiles"][sel_profile]
        st.caption("Ambiente, duração e rampa de cada passo do perfil selecionado:")
        for i, s in enumerate(steps):
            col_a, col_b, col_c = st.columns([3, 1, 1])
            env_key, dur_key, ramp_key = (
                f"visual_env_{sel_profile}_{i}",
                f"visual_dur_{sel_profile}_{i}",
                f"visual_ramp_{sel_profile}_{i}",
            )
            # Mesma regra do restante do app: nunca combinar "value=/default="
            # explicito com uma "key=" ja populada em session_state -- por
            # isso o valor inicial e escrito ANTES do widget, via setdefault.
            st.session_state.setdefault(env_key, s["env"])
            st.session_state.setdefault(dur_key, float(s["dur"]))
            st.session_state.setdefault(ramp_key, float(s["ramp"]))
            with col_a:
                env_val = st.segmented_control(
                    f"Passo {i+1} — ambiente",
                    options=list(range(1, K + 1)),
                    key=env_key,
                )
                if env_val is not None:
                    s["env"] = int(env_val)
            with col_b:
                s["dur"] = st.number_input("Duração (h)", min_value=0.0, key=dur_key)
            with col_c:
                s["ramp"] = st.number_input("Rampa (h)", min_value=0.0, key=ramp_key)
        st.session_state["profiles"][sel_profile] = steps
        st.altair_chart(build_profile_chart(steps, K), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 2 · Atribuir corpos de prova aos perfis")
    st.caption(
        "Uma linha por item testado: escolha o perfil seguido e informe o tempo de falha "
        "(ou marque **censurado** para um item que sobreviveu ao teste — o tempo final do "
        "perfil é usado automaticamente nesse caso)."
    )
    profile_names = list(st.session_state["profiles"].keys())
    edited_assign = st.data_editor(
        st.session_state["item_assignments"],
        num_rows="fixed",
        use_container_width=True,
        disabled=["item"],
        column_config={
            "item": st.column_config.NumberColumn("Item"),
            "perfil": st.column_config.SelectboxColumn("Perfil", options=profile_names),
            "tempo_falha_censura": st.column_config.NumberColumn("Tempo de falha (h)", min_value=0.0),
            "censurado": st.column_config.CheckboxColumn("Censurado"),
        },
        key="assignment_editor",
    )
    st.session_state["item_assignments"] = edited_assign

# --------------------------------------------------------------------------
# Tab 3 — Executar & Resultados
# --------------------------------------------------------------------------
with tab_run:
    st.markdown("#### Executar o modelo")
    st.caption(
        "**Estágio 1** usa só a confiabilidade elicitada de especialistas (a priori) — rápido, "
        "não usa os dados de teste. **Estágio 2** combina a priori com os dados de teste reais "
        "via MCMC, produzindo a confiabilidade posteriori. Rode o Estágio 1 primeiro para "
        "conferir a priori antes de comprometer os itens ao teste."
    )
    bc1, bc2 = st.columns(2)
    with bc1:
        stage1_clicked = st.button("① Estágio 1: Análise Priori", use_container_width=True)
    with bc2:
        stage2_clicked = st.button("② Estágio 2: Análise Posteriori", type="primary", use_container_width=True)

    def _run_stage(mode: str) -> bool:
        try:
            with st.spinner("Compilando o motor de cálculo (primeira execução pode levar alguns segundos)..."):
                binary_path = ensure_binary(CPP_DIR)

            K = int(st.session_state["K"])
            N = int(st.session_state["N"])
            S = int(st.session_state["S"])

            items = assemble_test_items(st.session_state["profiles"], st.session_state["item_assignments"])

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

            spinner_msg = (
                "Amostrando a priori (Dirichlet ordenada)..."
                if mode == "prior"
                else "Rodando MCMC (Metropolis-Hastings em bloco)..."
            )
            with st.spinner(spinner_msg):
                output_text = run_alt_cli(binary_path, input_text, timeout_s=300, mode=mode)

            st.session_state["last_output_text"] = output_text
            st.session_state["last_envs"] = parse_output(output_text)
            st.session_state["last_stage"] = mode  # "prior" (Estagio 1) ou "full" (Estagio 2)
            # Ajusta quais distribuicoes aparecem selecionadas na secao de
            # graficos: no Estagio 1 so existe priori (posteriori nem foi
            # calculada), no Estagio 2 restaura as duas.
            if mode == "prior":
                st.session_state["result_show_dists"] = ["Priori"]
            else:
                current = st.session_state.get("result_show_dists", ["Priori", "Posteriori"])
                if "Posteriori" not in current:
                    current = current + ["Posteriori"]
                st.session_state["result_show_dists"] = current
            st.success("Análise concluída.")
            return True
        except AltRunError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Erro inesperado: {e}")
        return False

    if stage1_clicked:
        if _run_stage("prior"):
            # Sem isto, o grafico Vega-Lite as vezes nao aparece na PRIMEIRA
            # renderizacao logo apos uma computacao pesada dentro de
            # st.spinner -- so aparecia depois de outra interacao qualquer
            # (ex.: mudar as faixas do histograma). Forcar mais um rerun aqui
            # garante que os resultados apareçam imediatamente apos o clique.
            st.rerun()
    if stage2_clicked:
        if _run_stage("full"):
            st.rerun()

    envs = st.session_state.get("last_envs")
    stage = st.session_state.get("last_stage")
    if envs:
        st.markdown("#### Resumo por ambiente")
        summary_rows = []
        for env_idx in sorted(envs):
            prior_rel = envs[env_idx].get("prior", {}).get("reliability")
            s_prior = summarize(prior_rel)
            if stage == "prior":
                summary_rows.append(
                    {
                        "Ambiente": env_idx,
                        "Confiabilidade priori (média)": s_prior["mean"],
                        "Confiabilidade priori (mediana)": s_prior["median"],
                        "Confiabilidade priori (moda)": s_prior["mode"],
                    }
                )
            else:
                post_rel = envs[env_idx].get("posterior", {}).get("reliability")
                s_post = summarize(post_rel)
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

        if stage == "prior":
            st.markdown("#### Distribuição da confiabilidade — priori")
        else:
            st.markdown("#### Distribuição da confiabilidade — priori vs. posteriori")

        all_envs = sorted(envs)
        st.session_state.setdefault("result_envs_select", all_envs[:1])
        # remove da seleção salva qualquer ambiente que não exista mais (ex.: K mudou entre execuções)
        st.session_state["result_envs_select"] = [e for e in st.session_state["result_envs_select"] if e in all_envs] or all_envs[:1]
        st.session_state.setdefault("result_value_type", "PDF")
        # Padrao 25 faixas: valor que mais se aproximou visual/numericamente das
        # Figuras 3(A)/5(A) do artigo (Van Dorp & Mazzuchi, 2004) na validacao
        # feita em 2026-07-24 -- ver memoria "alt-software-validation".
        st.session_state.setdefault("result_n_bins", 25)

        # No Estagio 1 (so priori), a UNICA opcao disponivel e "Priori" -- a
        # posteriori nem foi calculada, entao nao faz sentido oferece-la.
        dist_options = ["Priori"] if stage == "prior" else ["Priori", "Posteriori"]
        st.session_state["result_show_dists"] = [
            d for d in st.session_state.get("result_show_dists", dist_options) if d in dist_options
        ] or dist_options

        cc1, cc2, cc3, cc4 = st.columns([2, 1, 1, 1])
        with cc1:
            env_choices = st.multiselect(
                "Ambientes", options=all_envs, format_func=lambda e: f"Ambiente {e}", key="result_envs_select"
            )
        with cc2:
            value_type = st.radio("Distribuição", options=["PDF", "CDF"], key="result_value_type", horizontal=True)
        with cc3:
            show_dists = st.multiselect(
                "Mostrar", options=dist_options, key="result_show_dists"
            )
        with cc4:
            n_bins = st.number_input(
                "Faixas do histograma", min_value=5, max_value=200, step=5, key="result_n_bins"
            )
        st.caption(
            "A altura do pico do histograma depende do número de faixas: menos faixas "
            "acumulam mais massa de probabilidade por barra (pico mais alto), mais faixas "
            "se aproximam da resolução bruta do modelo (pico mais baixo). Priori e "
            "posteriori são divididas nesse número de faixas cada uma dentro do seu "
            "próprio intervalo de confiabilidade — como a posteriori normalmente ocupa "
            "uma faixa bem mais estreita (o teste reduz a incerteza), suas barras ficam "
            "mais finas que as da priori, mesmo com o mesmo número de faixas."
        )

        value_col = "pdf" if value_type == "PDF" else "cdf"
        value_axis_title = "Densidade (PDF)" if value_type == "PDF" else "Probabilidade acumulada (CDF)"

        # IMPORTANTE (pedido do usuario, 2026-07-24): priori e posteriori NAO
        # compartilham mais a mesma grade de faixas. Antes, as duas usavam o
        # mesmo bin fixo sobre [0,1] -- como a posteriori normalmente ocupa uma
        # faixa de confiabilidade bem mais estreita que a priori (o MCMC reduz
        # a incerteza ao incorporar os dados de teste), ela acabava so "usando"
        # uma fracao das faixas configuradas (ex.: com 25 faixas configuradas,
        # a posteriori do Ambiente 1 so preenchia ~7 delas, porque seus dados
        # ocupam so ~25% do eixo 0-1). Agora cada (Ambiente, Distribuicao) e
        # rebinado independentemente sobre o SEU PROPRIO intervalo [min, max]
        # em exatamente n_bins faixas -- isso garante ~n_bins faixas visiveis
        # para AMBAS, priori e posteriori, em todo ambiente, sem precisar de
        # dois controles de faixas separados. O eixo X do grafico continua
        # fixo em [0,1] (a POSICAO de cada faixa reflete o dado real); so a
        # LARGURA da faixa passa a variar por serie/ambiente.
        n_bins = int(n_bins)
        dist_key_map = {"Priori": "prior", "Posteriori": "posterior"}
        chart_rows = []
        for env_idx in env_choices:
            for dist_label in show_dists:
                dist_df = envs[env_idx].get(dist_key_map[dist_label], {}).get("reliability")
                if dist_df is None or dist_df.empty:
                    continue
                vmin = float(dist_df["value"].min())
                vmax = float(dist_df["value"].max())
                span = (vmax - vmin) or 1e-9
                step = span / n_bins
                # indice da faixa de CADA ponto bruto dentro do intervalo proprio
                # dessa serie (nao do eixo 0-1 inteiro) -- por isso o rebinning e
                # feito aqui em Python/pandas, e nao com alt.Bin (que so sabe
                # binar sobre um unico intervalo compartilhado por todo o campo).
                bin_idx = ((dist_df["value"] - vmin) / step).clip(upper=n_bins - 1e-9).astype(int)
                tmp = pd.DataFrame({"bin_idx": bin_idx, "v": dist_df[value_col]})
                # Mesma regra de sempre: PDF soma (preserva massa total = 1),
                # CDF usa o maximo (e cumulativa, nao somavel) por faixa.
                agg = tmp.groupby("bin_idx")["v"].sum() if value_type == "PDF" else tmp.groupby("bin_idx")["v"].max()
                for idx, val in agg.items():
                    bin_left = vmin + idx * step
                    chart_rows.append(
                        {
                            "Ambiente": f"Ambiente {env_idx}",
                            "Distribuição": dist_label,
                            "bin_left": bin_left,
                            "bin_right": bin_left + step,
                            "Valor": float(val),
                        }
                    )

        if not env_choices or not show_dists:
            st.info("Selecione ao menos um ambiente e uma distribuição para ver o gráfico.")
        elif chart_rows:
            chart_df = pd.DataFrame(chart_rows)
            chart_df["y0"] = 0.0
            # Escala de cor construida so com as distribuicoes REALMENTE marcadas em
            # "Mostrar" -- um dominio fixo faria a legenda listar "Priori"/"Posteriori"
            # sempre, mesmo quando so uma das duas foi selecionada, dando a falsa
            # impressao de que a outra ainda estava sendo desenhada no grafico.
            # A priori usa tonalidade VERDE (pedido do usuario, para diferenciar
            # visualmente do azul da posteriori e reforcar que o Estagio 1 e uma
            # etapa distinta, so com opiniao de especialistas).
            _dist_colors = {"Priori": DRUM["green_prior"], "Posteriori": DRUM["blue_primary"]}
            color_domain = [d for d in ["Priori", "Posteriori"] if d in show_dists]
            color_range = [_dist_colors[d] for d in color_domain]
            # Um UNICO grafico por ambiente (facet por Ambiente) -- design estavel,
            # testado e aprovado; NAO usar paineis separados por distribuicao
            # (row+column facet ja quebrou a renderizacao no componente Vega-Lite
            # do Streamlit numa rodada anterior).
            #
            # Priori e posteriori agora tem faixas de LARGURA DIFERENTE (ver bloco
            # acima), entao um xOffset lado-a-lado deixaria de fazer sentido visual
            # (as barras nao teriam a mesma largura para "encaixar" uma do lado da
            # outra). Em vez disso, as duas sao desenhadas SOBREPOSTAS no mesmo
            # eixo X com opacidade parcial (mark_bar(opacity=0.75)) -- onde priori e
            # posteriori ocupam a mesma regiao de confiabilidade, as cores se
            # misturam visualmente; onde nao, cada uma aparece na sua cor pura.
            #
            # Encoding x/x2 (em vez de bin=alt.Bin) porque cada faixa ja foi
            # calculada manualmente acima com seus proprios limites (bin_left/
            # bin_right); alt.Bin so sabe binar declarativamente sobre UM unico
            # intervalo compartilhado por todo o campo, o que e exatamente o que
            # NAO queremos mais aqui. Barra completa exige y=0 (coluna auxiliar
            # "y0") e y2=Valor -- especificar so x/x2 sem y2 faz o Vega-Lite tratar
            # a marca como um segmento fino tipo Gantt (achatado), nao uma barra
            # cheia do zero ate o valor (confirmado visualmente com vl-convert).
            chart = (
                alt.Chart(chart_df)
                .mark_bar(opacity=0.75)
                .encode(
                    x=alt.X("bin_left:Q", title="Confiabilidade no tempo de missão", scale=alt.Scale(domain=[0, 1])),
                    x2="bin_right:Q",
                    y=alt.Y("y0:Q", title=value_axis_title),
                    y2="Valor:Q",
                    color=alt.Color(
                        "Distribuição:N",
                        scale=alt.Scale(domain=color_domain, range=color_range),
                    ),
                )
                .properties(width=200, height=200)
                .facet(column=alt.Column("Ambiente:N", title=None), columns=3)
                .resolve_scale(y="independent")
            )
            # "key" muda a cada combinacao de ambiente/distribuicao/mostrar -- sem
            # isso, o Streamlit tenta reaproveitar o componente Vega-Lite anterior
            # e faz um patch incremental que, em graficos com facetas, pode deixar
            # marcas da renderizacao anterior (ex.: a serie cinza da Priori) atras
            # da nova, mesmo depois de desmarcar essa distribuicao em "Mostrar".
            # Uma key distinta forca o componente a ser remontado do zero.
            chart_key = (
                "result_chart_" + "-".join(str(e) for e in env_choices) + "_" + value_type
                + "_" + "-".join(show_dists) + f"_bins{n_bins}"
            )
            st.altair_chart(chart, use_container_width=False, key=chart_key)

        st.download_button(
            "⬇ Baixar saída completa (.txt)",
            data=st.session_state["last_output_text"],
            file_name="atlas_resultado.txt",
            mime="text/plain",
        )
    else:
        st.info(
            "Configure os parâmetros nas abas anteriores e clique em **① Estágio 1** "
            "(para ver só a priori) ou **② Estágio 2** (priori + posteriori)."
        )

# --------------------------------------------------------------------------
# Tab 4 — Manual
# --------------------------------------------------------------------------
with tab_manual:
    manual_path = os.path.join(APP_DIR, "manual.html")
    with open(manual_path, "r", encoding="utf-8") as f:
        manual_html = f.read()
    components.html(manual_html, height=1500, scrolling=True)
