// Driver de linha de comando do modelo ALT (Van Dorp & Mazzuchi, 2004).
//
// Reescrito para ler a configuracao de um arquivo texto (em vez dos arrays
// hardcoded do Exec.cpp original) e escrever os resultados -- prior e
// posteriori de taxa de falha/confiabilidade, para CADA ambiente de teste --
// em um unico arquivo de saida, delimitado por marcadores "### ENVIRONMENT n ###".
//
// Uso:
//   alt_cli <arquivo_de_entrada> <arquivo_de_saida>
//
// Formato do arquivo de entrada (tokens separados por espaco/quebra de linha,
// a formatacao em linhas abaixo e apenas para legibilidade):
//
//   K N S betaWeibull timeMission indexEnvironmentUse reliabilityQuantileEnvironmentUse
//   samples samplesBurnIn nSkip intervals censoredData(0=intervalar,1=tipoI)
//   R_1 R_2 ... R_K                                  (confiabilidade priori por ambiente)
//   -- para cada item j = 1..N --
//   env_0 env_1 ... env_S                            (S+1 indices; env_0 e sempre 0/dummy)
//   t_0 t_1 ... t_S                                   (S+1 tempos, t_0 = 0)
//   sigma_1 ... sigma_S                                (S tempos de rampa)
//   failureTime_j                                      (tempo de falha/censura do item j)
//
// Todos os tempos devem estar na MESMA unidade (recomendado: horas, para
// bater com a convencao do artigo).

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <time.h>
#include <stdlib.h>
#include "PosteriorAproximation.h"
#include "LikelihoodFunction.h"

using namespace std;

int main(int argc, char **argv)
{
	if (argc < 3) {
		cerr << "Uso: alt_cli <arquivo_de_entrada> <arquivo_de_saida> [envStart envEnd]" << endl;
		cerr << "  envStart/envEnd (opcionais): processa so um subconjunto de ambientes"
			 << " (1-based, inclusive), mantendo o modelo conjunto (K ambientes) intacto --"
			 << " util para dividir uma rodada longa em varias chamadas dentro do limite"
			 << " de tempo do sandbox." << endl;
		return 1;
	}

	ifstream fin(argv[1]);
	if (!fin.is_open()) {
		cerr << "Nao foi possivel abrir o arquivo de entrada: " << argv[1] << endl;
		return 1;
	}

	int K, N, S;
	double betaWeibull, timeMission, reliabilityQuantileEnvironmentUse;
	int indexEnvironmentUse;
	int samples, samplesBurnIn, nSkip, intervals, censoredFlag;

	fin >> K >> N >> S;
	fin >> betaWeibull >> timeMission >> indexEnvironmentUse >> reliabilityQuantileEnvironmentUse;
	fin >> samples >> samplesBurnIn >> nSkip >> intervals >> censoredFlag;

	double *reliabilityPrior = new double[K];
	for (int e = 0; e < K; e++) fin >> reliabilityPrior[e];

	int **env = new int*[N];
	double **times = new double*[N];
	double **ramp = new double*[N];
	double *failureTimes = new double[N];

	for (int j = 0; j < N; j++) {
		env[j] = new int[S + 1];
		times[j] = new double[S + 1];
		ramp[j] = new double[S];
		for (int i = 0; i <= S; i++) fin >> env[j][i];
		for (int i = 0; i <= S; i++) fin >> times[j][i];
		for (int i = 0; i < S; i++) fin >> ramp[j][i];
		fin >> failureTimes[j];
	}
	fin.close();

	setTestPlan(N, S, env, times, ramp);

	fout.open(argv[2]);
	if (!fout.is_open()) {
		cerr << "Nao foi possivel abrir o arquivo de saida: " << argv[2] << endl;
		return 1;
	}

	bool censoredData = (censoredFlag != 0);
	long tStart = time(0);

	int envStart = 1, envEnd = K;
	if (argc >= 5) {
		envStart = atoi(argv[3]);
		envEnd = atoi(argv[4]);
	}

	// ANTES: um novo PosteriorAproximation era criado e MetropolisSampler
	// (a versao ORIGINAL, por-ambiente) era chamada uma vez POR AMBIENTE
	// dentro deste loop -- ou seja, o modelo de K ambientes inteiro era
	// re-simulado do zero K vezes, descartando a cada rodada as K-1
	// coordenadas nao gravadas. Isso e valido estatisticamente mas gera
	// cadeias independentes cujo ruido de amostragem pode quebrar a
	// ordenacao esperada entre ambientes vizinhos mesmo com muitas
	// iteracoes (observado empiricamente: a inversao persistia, so mudando
	// de par, de 100 mil a 1 milhao de iteracoes).
	//
	// AGORA: MetropolisSamplerJoint roda a MESMA cadeia MH uma UNICA vez
	// (custo ~K vezes menor pelo mesmo total de iteracoes) e grava as K
	// coordenadas de cada amostra aceita internamente; o loop abaixo so
	// itera sobre os resultados JA COLETADOS para escrever a saida de cada
	// ambiente (computePriorParametersForEnv/computePosteriorParametersForEnv
	// reaproveitam a logica original de reducao/CDF sem modifica-la).
	PosteriorAproximation *posterior = new PosteriorAproximation();

	posterior->MetropolisSamplerJoint(samples, samplesBurnIn, intervals, nSkip,
		N, S, K, betaWeibull, timeMission, indexEnvironmentUse,
		reliabilityQuantileEnvironmentUse, reliabilityPrior, failureTimes, censoredData);

	int numberThin = samples / nSkip;

	for (int e = envStart; e <= envEnd; e++) {
		fout << "### ENVIRONMENT " << e << " ###" << endl;

		posterior->computePriorParametersForEnv(e, intervals, timeMission);
		posterior->computePosteriorParametersForEnv(e, nSkip, numberThin, intervals, timeMission);

		cerr << "Ambiente " << e << " de " << K << " concluido." << endl;
	}

	fout.close();

	freeTestPlan(N);
	delete [] reliabilityPrior;
	delete [] failureTimes;
	for (int j = 0; j < N; j++) {
		delete [] env[j];
		delete [] times[j];
		delete [] ramp[j];
	}
	delete [] env;
	delete [] times;
	delete [] ramp;

	long tEnd = time(0);
	cerr << "Fim. Tempo total: " << (tEnd - tStart) << "s" << endl;
	return 0;
}
