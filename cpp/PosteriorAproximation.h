// $Id: PosteriorAproximationSignatures.h,v 1.4 2005/03/01 13:20:52 marcio Exp $

#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include "DistributionPoint.h"
using namespace std;

// Arquivo de saida compartilhado. Aberto explicitamente em Exec.cpp::main()
// com o caminho recebido por linha de comando (era hardcoded como "mean.xls").
extern ofstream fout;

// Implementa o modelo de inferencia Bayesiana para ALT (Accelerated Life
// Testing) de Van Dorp & Mazzuchi (2004), "A general Bayes exponential
// inference model for accelerated life testing", JSPI 119, 55-74.
//
// Visao geral do fluxo (ver comentarios de cada metodo para detalhes):
//   1) A PRIORI: PriorDistribution resolve os parametros (beta, alpha[]) da
//      Dirichlet ordenada (eq. 16/25 do artigo) a partir das confiabilidades
//      elicitadas (Tabela 1). generateCandidatePrior/PriorMetropolisSampler
//      (ou a variante *Joint, ver abaixo) fazem Gibbs sampling dessa priori.
//   2) A POSTERIORI: dado o teste ALT (ambientes/tempos/rampas/falhas em
//      LikelihoodFunction), MetropolisSampler (ou *Joint) roda um
//      Metropolis-Hastings em bloco sobre o vetor ordenado u[1..K] (as K
//      confiabilidades transformadas dos K ambientes), usando a priori acima
//      como densidade a priori conjunta.
//   3) REDUCAO: computePriorParameters/computePosteriorParameters convertem
//      as amostras (valor, peso) em uma distribuicao discretizada (CDF) por
//      faixas ("reduce"/"reducePrior"), escrita em "fout".
class PosteriorAproximation
{
public:
	double computeLogDensityPrior(int number_environments, double beta, double alpha[], double X[], double Y[]);
	double computeBetaLogDensityPosterior(int number_environments, double X[], double Y[]);
	double computeDirichletLogDensity (int number_environments, double u[], double alpha[]);
	double computeMoveProbabilityPrior(int number_environments, double X[], double Y[], double alpha[]);
	double computeMoveProbabilityPosterior (int number_items, int number_steps, int number_environments, int index, double betaWeibull, double c, double failureTimes[], double X[], double Y[], double beta, double alpha[], bool censoredData);
	double computeBetaPriorQuantile (int number_environments, int index, double c, double beta, double alpha[]);
	double generateUniformSample();
	void PriorMetropolisSampler (int index, int number_environments, int numberThin, int samplesBurnIn, double beta, double alpha[]);
	double generateUniformDeviate();
	bool setDensityMode(int mode);
	bool increaseVariation();
	bool decreaseVariation();
	void setMinimumVariation();
	void setMaximumVariation();
	double generateCandidatePrior(int number_environments, int index, double beta, double alpha[]);
	double *generateCandidatePosterior(int number_environments, double X[], double Y[]);
	void MetropolisSampler (int samples, int pSamplesBurnIn, int intervals, int nSkip, int index, int number_items, int number_steps, int number_environments, double betaWeibull, double timeMission, int indexEnvironmentUse, double reliabilityQuantileEnvironmentUse, double reliabilityPrior[], double failureTimes[], bool censoredData);
	void QuickSort(double A[], int l, int r);
	void QuickSort(double A[],double P[], int l, int r);
	void orderPoints (int samplesBurnIn, int samplesThin, double var[], double varOrder[]);
	void orderTable (int size, double pointsX[], double pointsY[]);
	void computePriorParameters (int intervals, double t);
	void computePosteriorParameters (int nSkip, int numberThin, int intervals, double t);
	void computePriorCDF(int numberElements, int samplesBurnIn);
	void computePosteriorCDF(int samplesBurnIn);
	void reduce(int intervals, int size, bool logScale, double varOrder[], double probabilityPosterior[]);
	void reduceInterval(int start, int end, int size, double maxLogWeight, double & mean, double & prob, double varOrder[], double logWeightValues[]);
	void reducePrior(int intervals, int size, bool logScale, double varOrder[], double probabilityPosterior[]);
	void reduceIntervalPrior(int start, int end, int size, double & mean, double & prob, double varOrder[], double probability[]);

	// ------------------------------------------------------------------
	// Amostragem CONJUNTA (adicionada nesta refatoracao, 2026-07).
	//
	// Por que existe: PriorMetropolisSampler/MetropolisSampler (acima, ORIGINAIS
	// e mantidas sem nenhuma alteracao de comportamento) rodam o processo de
	// amostragem MCMC uma vez POR AMBIENTE -- o parametro "index" so controla
	// qual coordenada do vetor u[]/X[] (que e ordenado e sempre atualizado por
	// INTEIRO a cada passo, para todos os K ambientes) e gravada em
	// valuesPrior/valuesPosterior; as outras K-1 coordenadas geradas na mesma
	// iteracao sao descartadas. Isso e estatisticamente valido (a marginal de
	// cada ambiente sai correta), mas: (a) desperdica trabalho, rodando o
	// mesmo custo de simulacao K vezes ao inves de uma; e (b) e mais sensivel
	// a ruido de amostragem entre ambientes vizinhos, pois cada ambiente vem
	// de uma cadeia (e burn-in) INDEPENDENTE das demais. Isso foi observado na
	// pratica: mesmo aumentando as iteracoes de 100 mil para 1 milhao, uma
	// pequena inversao na ordenacao esperada da confiabilidade entre dois
	// ambientes adjacentes persistia -- so mudava DE QUAL PAR de ambientes,
	// nunca desaparecia, um sintoma classico de comparar marginais de cadeias
	// distintas em vez de uma unica cadeia conjunta.
	//
	// O que as funcoes abaixo fazem: rodam a MESMA cadeia (reaproveitando
	// generateCandidatePrior/generateCandidatePosterior/
	// computeMoveProbabilityPosterior sem nenhuma modificacao -- essas ja
	// operam sobre o vetor de K ambientes de uma vez so) uma UNICA vez,
	// gravando as K coordenadas de cada amostra aceita. computePriorParameters
	// /computePosteriorParameters (originais, tambem sem alteracao) sao entao
	// reaproveitadas por ambiente atraves dos wrappers
	// computePriorParametersForEnv/computePosteriorParametersForEnv, que apenas
	// apontam os campos sizePrior/valuesPrior (ou sizePosterior/valuesPosterior)
	// para os dados do ambiente pedido antes de delegar para a funcao original.
	void PriorMetropolisSamplerJoint (int number_environments, int numberThin, int samplesBurnIn, double beta, double alpha[]);
	void MetropolisSamplerJoint (int samples, int samplesBurnIn, int intervals, int nSkip, int number_items, int number_steps, int number_environments, double betaWeibull, double timeMission, int indexEnvironmentUse, double reliabilityQuantileEnvironmentUse, double reliabilityPrior[], double failureTimes[], bool censoredData);
	// "Estagio 1" da interface (analise SO da priori, sem dados de teste): faz
	// a mesma configuracao que MetropolisSamplerJoint faz antes de chamar
	// PriorMetropolisSamplerJoint (alocar prior->counterTotalEnvironmentValue,
	// calcular alpha[] a partir da confiabilidade elicitada, etc.) mas SEM
	// alocar as estruturas do lado da posteriori (_valuesPosteriorByEnv), que
	// sao desnecessarias e caras aqui. PriorMetropolisSamplerJoint sozinha
	// NAO e autossuficiente -- depende dessa configuracao previa (confirmado
	// com AddressSanitizer: chama-la isolada sem isso da segfault em
	// prior->counterTotalEnvironmentValue, que fica nulo).
	void PriorOnlyAnalysis (int number_environments, int samples, int samplesBurnIn, int intervals, double timeMission, int indexEnvironmentUse, double reliabilityQuantileEnvironmentUse, double reliabilityPrior[]);
	void computePriorParametersForEnv (int env, int intervals, double t);
	void computePosteriorParametersForEnv (int env, int nSkip, int numberThin, int intervals, double t);

	~PosteriorAproximation ();


private:
	int _intervals;      // numero de faixas usado na discretizacao (reduce/reducePrior)
	int _samples;        // total de iteracoes do MH (usado so para dimensionar arrays no destrutor)
	int _samplesBurnIn;  // iteracoes de burn-in descartadas (idem)
	int numberThin;      // tamanho da cadeia "thinned" (sub-amostrada) usada na priori
	int sizePrior;        // quantas linhas validas ha em valuesPrior (preenchido por PriorMetropolisSampler ou pelo wrapper *ForEnv)
	int sizePosterior;    // quantas linhas validas ha em valuesPosterior (idem, por MetropolisSampler ou *ForEnv)
	double c;             // fator de transformacao (eq. 24 do artigo) entre confiabilidade e taxa de falha
	double sumOfProbabilities; // acumulador auxiliar usado por reduce()/reduceInterval() para normalizar o CDF
	double normalizer;        // idem, usado por reducePrior()/reduceIntervalPrior()
	double _densityPopulation; // controla a concentracao (passo) da proposta Beta local em generateCandidatePosterior/computeBetaLogDensityPosterior; ajustado adaptativamente durante o burn-in via increaseVariation()/decreaseVariation() (ver MetropolisSampler/MetropolisSamplerJoint)
	double _rejected;     // nao usado no fluxo atual (mantido do codigo original)
	double *u;            // vetor ordenado das confiabilidades transformadas (u[0]=1 e u[K+1]=0 sao "sentinelas" fixas; u[1..K] sao os K ambientes)
	double *X;            // estado ATUAL da cadeia de Metropolis-Hastings (mesma estrutura de u[])
	double *Y;            // proposta candidata gerada a cada iteracao (idem)
	double *alphaElicitated; // parametros elicitados acumulados (eq. 25, "Step 5" do artigo) antes da diferenciacao
	double *alpha;        // parametros de forma da Dirichlet ordenada por ambiente (eq. 25, ja diferenciados)
	double *uPriorMean;   // media a priori de cada u[e], estimada via PriorMetropolisSampler/PriorMetropolisSamplerJoint, usada como ponto de partida da cadeia posteriori
	double *uPrior;        // nao usado diretamente (mantido do codigo original; ver valuesPrior)
	double *ratePrior;     // taxas de falha a priori (transformadas de u via c), ordenadas para o CDF
	double *relPrior;      // confiabilidades a priori (exp(-taxa*tempoMissao)), ordenadas
	double *ratePosterior; // idem, a posteriori
	double *relPosterior;  // idem, a posteriori
	double *probabilityPrior;     // pesos associados a ratePrior/relPrior (mesma ordem, apos QuickSort conjunto)
	double *probabilityPosterior; // idem, a posteriori
	double **valuesPrior;     // [linha][0]=valor de u amostrado, [linha][1]=peso (1/numberThin); preenchido por PriorMetropolisSampler (ORIGINAL, por ambiente) ou apontado para _valuesPriorByEnv[env] pelo wrapper computePriorParametersForEnv
	double **valuesPosterior; // [linha][0]=valor de X[index] no momento do aceite, [linha][1]=log(peso) (repeticoes do mesmo estado); preenchido por MetropolisSampler (ORIGINAL) ou apontado para _valuesPosteriorByEnv[env] pelo wrapper computePosteriorParametersForEnv
	double **valuesReduced;   // buffer de trabalho de reduce()/reducePrior(): [faixa][0]=media, [faixa][1]=peso agregado da faixa

	// --- Dados da amostragem CONJUNTA (ver MetropolisSamplerJoint/PriorMetropolisSamplerJoint no topo da classe) ---
	// Guardam, para cada um dos "numero_environments" ambientes (indices
	// 1..K), o mesmo tipo de tabela (valor, peso) que valuesPrior/
	// valuesPosterior guardam para UM ambiente so -- mas todos preenchidos a
	// partir de uma UNICA execucao da cadeia, garantindo que os K ambientes
	// comparados entre si venham da mesma amostra (sem ruido extra de
	// cadeias/burn-ins independentes).
	double ***_valuesPriorByEnv;      // _valuesPriorByEnv[e] tem o mesmo formato de valuesPrior, para o ambiente e (1..K)
	int _sizePriorJoint;              // tamanho comum a todos os ambientes (a cadeia a priori e a mesma para todos)
	double ***_valuesPosteriorByEnv;  // _valuesPosteriorByEnv[e] tem o mesmo formato de valuesPosterior, para o ambiente e (1..K)
	int _sizePosteriorJoint;          // tamanho comum a todos os ambientes (decisao de aceite/rejeicao e conjunta)
	int _numberEnvironmentsJoint;     // K usado na ultima chamada a *Joint (para alocar/liberar os arrays acima)
};