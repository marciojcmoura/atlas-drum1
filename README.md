# ATLAS — Accelerated Testing & Lifetime Analysis System

Módulo DRUM / Risk Management para inferência Bayesiana em testes de vida
acelerados (ALT), implementando o modelo de Van Dorp & Mazzuchi (2004), *A
general Bayes exponential inference model for accelerated life testing*,
JSPI 119, 55–74.

O motor de cálculo é o código C++ original (2005), validado e corrigido
contra o artigo, compilado como um binário (`atlas_cli`) e chamado pela
interface via subprocess.

## Estrutura

```
app-streamlit/
  streamlit_app.py     interface (formulário, execução, resultados, manual)
  manual.html            conteúdo da aba "Manual" (renderizado via components.html)
  alt_io.py              monta o arquivo de entrada do atlas_cli e lê a saída
  build_cli.py           compila cpp/ em atlas_cli na primeira execução
  requirements.txt       dependências Python
  packages.txt           dependências de sistema (g++) para o Streamlit Cloud
  .streamlit/config.toml tema base
  cpp/                   código-fonte C++ (validado; NÃO commitar binários)
```

## Rodando localmente

```bash
pip install -r requirements.txt
# precisa de g++ instalado (build-essential no Linux, Xcode CLT no macOS)
streamlit run streamlit_app.py
```

Na primeira execução, o app compila `cpp/` automaticamente (pode levar alguns
segundos); nas execuções seguintes, reaproveita o binário já compilado.

## Deploy no Streamlit Community Cloud

1. Suba esta pasta (`app-streamlit/`) para um repositório no GitHub.
2. Em [share.streamlit.io](https://share.streamlit.io), crie um novo app
   apontando para o repositório e para `streamlit_app.py`.
3. O `packages.txt` já está configurado para instalar `g++`/`build-essential`
   no ambiente do Streamlit Cloud — não é necessário nenhum passo manual
   de build.

## Uso

1. **Configuração**: número de ambientes/itens/passos de estresse,
   parâmetros da Weibull/missão, confiabilidade elicitada a priori por
   ambiente, e parâmetros do MCMC (iterações, burn-in, thinning).
2. **Plano de teste**: uma linha por item testado (sequência de ambientes,
   marcos de tempo, tempos de rampa, tempo de falha/censura).
3. **Executar & Resultados**: roda o modelo e mostra, por ambiente, a
   confiabilidade a priori vs. a posteriori (médias, mediana, moda e a
   distribuição CDF completa), com download da saída bruta.
4. **Manual**: guia de uso completo, embutido no próprio app (aba 4).

Use o botão **"Carregar exemplo do artigo"** (barra lateral) para preencher
tudo automaticamente com os dados de exemplo do artigo (Tabelas 1 e 2,
Seção 5) e ver o app funcionando de ponta a ponta.

## Notas sobre o modelo

- A amostragem MCMC usa uma cadeia única conjunta sobre os K ambientes
  (`MetropolisSamplerJoint`), com proposta adaptativa durante o burn-in
  (ajuste automático da largura do passo visando taxa de aceitação de
  20–50%). Ver comentários em `cpp/PosteriorAproximation.cpp` para o
  histórico completo de correções em relação ao código original de 2005.
- Iterações totais maiores (300k+) dão resultados mais estáveis, ao custo de
  mais tempo de execução (a barra de progresso mostra o andamento).
