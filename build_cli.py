"""
build_cli.py — compila o binario atlas_cli (motor de calculo do ATLAS --
Accelerated Testing & Lifetime Analysis System, modulo DRUM) a partir do
codigo-fonte em cpp/, em tempo de execucao (usado pelo streamlit_app.py via
st.cache_resource, para compilar so uma vez por sessao/container).

Por que compilar em runtime em vez de commitar o binario no repositorio:
alt_cli e um binario nativo (arquitetura/SO especificos); commitar o binario
prenderia o deploy a uma unica plataforma. Compilar no Streamlit Community
Cloud (Linux) via g++ (instalado a partir de packages.txt) garante que o
binario sempre bate com o ambiente onde vai rodar.
"""

import os
import subprocess

SOURCES = [
    "DistributionPoint.cpp",
    "Exec.cpp",
    "LikelihoodFunction.cpp",
    "PosteriorAproximation.cpp",
    "PriorDistribution.cpp",
    "dcdflib.cpp",
    "ipmpar.cpp",
    "linpack.cpp",
    "ranlib.cpp",
    "com.cpp",
]

BINARY_NAME = "atlas_cli"


def ensure_binary(cpp_dir: str) -> str:
    """Garante que cpp_dir/atlas_cli existe e e executavel, compilando com g++
    se necessario. Retorna o caminho absoluto do binario.
    Levanta RuntimeError com a saida do compilador em caso de falha."""
    # IMPORTANTE: normalizamos para caminho absoluto e NAO usamos o parametro
    # "cwd" do subprocess (bug encontrado nesta implementacao: passar cwd=cpp_dir
    # ENQUANTO os caminhos de origem/objeto ja incluiam cpp_dir via os.path.join
    # fazia o compilador procurar "cpp_dir/cpp_dir/arquivo.cpp" -- duplicando o
    # prefixo do diretorio e falhando com "No such file or directory"). Com
    # caminhos absolutos e sem cwd, o comando funciona independente de qual e
    # o diretorio de trabalho de quem chamou ensure_binary().
    cpp_dir = os.path.abspath(cpp_dir)
    binary_path = os.path.join(cpp_dir, BINARY_NAME)

    newest_source_mtime = max(
        os.path.getmtime(os.path.join(cpp_dir, src)) for src in SOURCES
    )
    needs_build = (
        not os.path.isfile(binary_path)
        or not os.access(binary_path, os.X_OK)
        or os.path.getmtime(binary_path) < newest_source_mtime
    )

    if not needs_build:
        return binary_path

    object_files = []
    for src in SOURCES:
        src_path = os.path.join(cpp_dir, src)
        obj_path = os.path.join(cpp_dir, src.replace(".cpp", ".o"))
        compile_cmd = ["g++", "-std=c++03", "-O2", "-c", src_path, "-o", obj_path]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Falha ao compilar {src}:\n{result.stderr[-4000:]}"
            )
        object_files.append(obj_path)

    link_cmd = ["g++", "-std=c++03", "-O2", "-o", binary_path] + object_files + ["-lm"]
    result = subprocess.run(link_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Falha ao linkar {BINARY_NAME}:\n{result.stderr[-4000:]}"
        )

    os.chmod(binary_path, 0o755)
    return binary_path
