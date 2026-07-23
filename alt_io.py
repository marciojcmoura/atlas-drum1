"""
alt_io.py — camada de integracao entre a interface Streamlit e o binario
compilado alt_cli (modelo ALT de Van Dorp & Mazzuchi, 2004).

Responsabilidades:
  1) build_input_text(...):  monta o arquivo de entrada em texto plano no
     formato esperado por Exec.cpp (ver cabecalho de cpp/Exec.cpp).
  2) run_alt_cli(...):       escreve o arquivo de entrada, roda o binario via
     subprocess e le o arquivo de saida.
  3) parse_output(...):      converte o texto de saida (blocos "### ENVIRONMENT
     n ###" com tabelas Prior/Posterior de taxa de falha e confiabilidade) em
     um dicionario de DataFrames, um por ambiente, facil de plotar no app.

Nenhuma logica estatistica mora aqui -- isso e so a "cola" de I/O em torno do
executavel C++ ja validado.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List

import pandas as pd


class AltRunError(RuntimeError):
    """Erro ao compilar ou executar o alt_cli."""
    pass


@dataclass
class TestItem:
    """Um item de teste ALT: sequencia de ambientes, marcos de tempo, tempos
    de rampa e tempo de falha/censura -- uma linha da Tabela 2 do artigo."""
    envs: List[int]       # S valores (ambiente em cada passo, 1..K)
    times: List[float]    # S+1 valores (t_0=0, t_1..t_S); marcos de tempo
    ramps: List[float]    # S valores (tempo de rampa no INICIO de cada passo)
    failure_time: float   # tempo de falha (ou censura, se == t_S)


def build_input_text(
    K: int,
    N: int,
    S: int,
    beta_weibull: float,
    time_mission: float,
    index_environment_use: int,
    reliability_quantile_environment_use: float,
    samples: int,
    samples_burn_in: int,
    n_skip: int,
    intervals: int,
    censored_flag: int,
    reliability_prior: List[float],
    items: List[TestItem],
) -> str:
    """Monta o texto do arquivo de entrada do alt_cli.

    O formato (ver cpp/Exec.cpp) e:
        K N S betaWeibull timeMission indexEnvironmentUse reliabilityQuantileEnvironmentUse
        samples samplesBurnIn nSkip intervals censoredData
        R_1 ... R_K
        -- por item j=1..N --
        env_0 env_1 ... env_S   (env_0 e sempre 0, dummy)
        t_0 t_1 ... t_S         (t_0 e sempre 0)
        sigma_1 ... sigma_S
        failureTime_j
    """
    if len(reliability_prior) != K:
        raise AltRunError(f"reliability_prior precisa ter {K} valores (tem {len(reliability_prior)}).")
    if len(items) != N:
        raise AltRunError(f"Esperados {N} itens de teste, recebidos {len(items)}.")

    lines = []
    lines.append(f"{K} {N} {S}")
    lines.append(f"{beta_weibull} {time_mission} {index_environment_use} {reliability_quantile_environment_use}")
    lines.append(f"{samples} {samples_burn_in} {n_skip} {intervals} {censored_flag}")
    lines.append(" ".join(str(x) for x in reliability_prior))

    for item in items:
        if len(item.envs) != S:
            raise AltRunError(f"Item com {len(item.envs)} ambientes, esperado {S}.")
        if len(item.times) != S + 1:
            raise AltRunError(f"Item com {len(item.times)} marcos de tempo, esperado {S + 1}.")
        if len(item.ramps) != S:
            raise AltRunError(f"Item com {len(item.ramps)} tempos de rampa, esperado {S}.")

        lines.append("0 " + " ".join(str(x) for x in item.envs))
        lines.append(" ".join(str(x) for x in item.times))
        lines.append(" ".join(str(x) for x in item.ramps))
        lines.append(str(item.failure_time))

    return "\n".join(lines) + "\n"


def run_alt_cli(binary_path: str, input_text: str, timeout_s: int = 300) -> str:
    """Escreve input_text em um arquivo temporario, roda alt_cli e retorna o
    conteudo do arquivo de saida (texto). Levanta AltRunError em caso de falha
    (binario ausente, timeout, saida vazia, codigo de retorno != 0)."""
    if not os.path.isfile(binary_path) or not os.access(binary_path, os.X_OK):
        raise AltRunError(f"Binario alt_cli nao encontrado ou nao executavel em: {binary_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.txt")
        out_path = os.path.join(tmpdir, "output.txt")
        with open(in_path, "w") as f:
            f.write(input_text)

        try:
            result = subprocess.run(
                [binary_path, in_path, out_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise AltRunError(
                f"O modelo nao terminou em {timeout_s}s. Tente reduzir o numero de iteracoes (amostras)."
            )

        if result.returncode != 0:
            raise AltRunError(
                f"alt_cli terminou com erro (codigo {result.returncode}).\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )

        if not os.path.isfile(out_path):
            raise AltRunError("alt_cli rodou mas nao produziu arquivo de saida.\n" + result.stderr[-4000:])

        with open(out_path, "r") as f:
            output_text = f.read()

        if not output_text.strip():
            raise AltRunError("Arquivo de saida do alt_cli esta vazio.\n" + result.stderr[-4000:])

        return output_text


_ENV_HEADER_RE = re.compile(r"### ENVIRONMENT (\d+) ###")
_SECTION_HEADERS = {
    "Failure Rate Prior": ("prior", "rate"),
    "Reliability Prior": ("prior", "reliability"),
    "Failure Rate Posterior": ("posterior", "rate"),
    "Reliability Posterior": ("posterior", "reliability"),
}


def parse_output(output_text: str) -> dict:
    """Converte o texto de saida do alt_cli em:
        { env_index: { "prior": {"rate": df, "reliability": df},
                       "posterior": {"rate": df, "reliability": df} } }
    onde cada df tem colunas [value, pdf, cdf].
    """
    envs = {}
    cur_env = None
    cur_key = None  # (which, kind)
    buffer = []

    def flush():
        if cur_env is not None and cur_key is not None and buffer:
            which, kind = cur_key
            df = pd.DataFrame(buffer, columns=["value", "pdf", "cdf"])
            envs.setdefault(cur_env, {}).setdefault(which, {})[kind] = df

    for raw_line in output_text.splitlines():
        line = raw_line.rstrip("\n")

        m = _ENV_HEADER_RE.match(line)
        if m:
            flush()
            cur_env = int(m.group(1))
            cur_key = None
            buffer = []
            continue

        matched_header = None
        for header, key in _SECTION_HEADERS.items():
            if line.startswith(header):
                matched_header = key
                break
        if matched_header is not None:
            flush()
            cur_key = matched_header
            buffer = []
            continue

        if line.strip() == "":
            continue

        parts = line.split("\t")
        if len(parts) == 3:
            try:
                x, p, F = float(parts[0]), float(parts[1]), float(parts[2])
                buffer.append((x, p, F))
            except ValueError:
                pass

    flush()
    return envs


def summarize(df: pd.DataFrame) -> dict:
    """Media, mediana (primeiro ponto com CDF>=0.5) e moda (maior PDF) de uma
    tabela (value, pdf, cdf)."""
    if df is None or df.empty:
        return {"mean": None, "median": None, "mode": None}
    mean = float((df["value"] * df["pdf"]).sum())
    median_rows = df[df["cdf"] >= 0.5]
    median = float(median_rows.iloc[0]["value"]) if not median_rows.empty else None
    mode = float(df.loc[df["pdf"].idxmax(), "value"])
    return {"mean": mean, "median": median, "mode": mode}
