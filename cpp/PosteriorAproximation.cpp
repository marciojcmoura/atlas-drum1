#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <math.h>
#include <vector>
using namespace std;

#include "PriorDistribution.h"
#include "LikelihoodFunction.h"
#include "PosteriorAproximation.h"
#include "ranlib.h"

#define SMALL_VALUE 1e-20
#define ONE 1
#define MIN_POPULATION 3
#define MAX_POPULATION (3 * 3 * 3 * 3 * 3 * 3 * 3 * 3)
#define POPULATION_STEP 3
#define MINIMUM_REJECTION_RATE (2)


PriorDistribution *prior = new PriorDistribution ();
LikelihoodFunction *lkl = new LikelihoodFunction ();

// Antes era aberto direto num arquivo fixo ("mean.xls"). Agora fica sem
// abrir ate que main() (Exec.cpp) decida o caminho de saida via fout.open(...),
// permitindo rodar varios cenarios/ambientes no mesmo processo.
ofstream fout;

double PosteriorAproximation::computeDirichletLogDensity 
			(int number_environments, double u[], double alpha[]){
	double dif;
	double accuracy = 0.000001;
	double beta = prior->getBeta ();
	double sumLog = 0.0;
	for (int i = 1; i <= number_environments + 1; i++){
		dif = u[i - 1] - u[i];
		if (prior->isEqual (dif, 0.0, accuracy)){
			dif = accuracy;
		}
		sumLog += (beta*alpha[i-1] - 1)*log (dif);
	}
	return sumLog;
}

double *PosteriorAproximation::generateCandidatePosterior(int number_environments, double X[], double Y[])
{
	// ANTES: esta funcao atualizava prior->uAux[i] em sequencia dentro do
	// proprio loop (linha "prior->uAux[i] = Y[i];"), de forma que o vizinho
	// "de baixo" (i-1), ao ser lido para i>=2, ja vinha ATUALIZADO com o novo
	// Y[i-1] em vez do X[i-1] original -- uma mistura indevida de valores
	// antigos/novos dentro da MESMA proposta. Isso por si so nao seria
	// necessariamente errado (seria uma varredura de Gibbs valida), MAS
	// computeBetaLogDensityPosterior (usada para calcular a razao de
	// Metropolis-Hastings) assume neighbors vindos EXCLUSIVAMENTE do array
	// antigo (ver correcao abaixo) -- a inconsistencia entre o mecanismo real
	// de geracao da proposta e a densidade usada na razao de aceitacao
	// quebra o balanco detalhado do algoritmo, enviesando a posteriori.
	// Corrigido para ler estritamente de X (nunca mutado) e escrever em Y.
	//
	// Proposta adaptativa: a forma Beta(5,3) ORIGINAL tem moda fixa em
	// 5/8=0.625 do intervalo [d,e], INDEPENDENTE do valor atual X[i] -- ou
	// seja, nao e uma perturbacao local do estado atual, e sim uma proposta
	// de "independencia" que sempre mira no mesmo ponto fixo entre os
	// vizinhos. Isso foi descoberto ao testar o esquema adaptativo: estreitar
	// essa Beta (aumentar _densityPopulation) so torna a proposta mais
	// deterministicamente igual a esse ponto fixo errado -- NAO mais proxima
	// de X[i] -- entao a taxa de aceitacao nao melhora (na pratica so piora)
	// por mais que se aumente a concentracao, confirmado empiricamente
	// (taxa ficou em ~0% mesmo apos varias reducoes de escala).
	//
	// Correcao: recentralizar a Beta na posicao relativa ATUAL de X[i] entre
	// seus vizinhos (v0 = (X[i]-d)/(e-d)), tornando-a uma proposta de passeio
	// aleatorio local de fato -- com _densityPopulation controlando apenas a
	// concentracao (tamanho do passo) ao redor de v0, nao mais o alvo.
	// alpha=1+v0*força, beta=1+(1-v0)*força: quando força->0 (_densityPopulation
	// pequeno) a proposta tende a Uniforme(0,1) (passos largos, muita
	// exploracao); quando força->infinito (_densityPopulation grande) a
	// proposta tende a massa pontual em v0, ou seja, Y[i]->X[i] (passos
	// minimos, aceitacao proxima de 100%). Este e o comportamento correto
	// esperado de um esquema adaptativo: aumentar _densityPopulation (via
	// decreaseVariation(), quando a taxa de aceitacao observada esta baixa
	// demais) deve control-avelmente ELEVAR a taxa de aceitacao aproximando
	// Y de X, o que a parametrizacao anterior nao fazia.
	double value;
	double d, e;
	double strength = _densityPopulation;

	for (int i = 1; i <= number_environments; i++) {
		d = X[i + 1];
		e = X[i - 1];
		double v0 = (X[i] - d) / (e - d);
		double alpha1 = 1.0 + v0 * strength;
		double beta = 1.0 + (1.0 - v0) * strength;

		value = 0;
		while (value < SMALL_VALUE || value >= ONE) {
			value = prior->generateBetaDeviate(alpha1, beta);
		}

		Y[i] = d + (e - d) * value;
	}
	return Y;
}

// Calcula log{q(target | reference)}, a densidade (com correcao de Jacobiano)
// de propor "target[i]" a partir dos vizinhos de "reference" (reference[i-1],
// reference[i+1]), consistente com a transformacao usada em
// generateCandidatePosterior: target[i] = reference[i+1] + (reference[i-1] -
// reference[i+1]) * value, value ~ Beta(5,3).
//
// ANTES esta funcao ignorava por completo o primeiro argumento no calculo
// (a variavel "x" era computada e nunca usada) e normalizava Y usando os
// vizinhos do PROPRIO Y em vez dos vizinhos de X -- ou seja, lnqxy e lnqyx
// acabavam medindo duas quantidades autoreferentes e nao relacionadas entre
// si, em vez da real densidade de proposta condicional necessaria para a
// razao de Metropolis-Hastings (eq. de Metropolis-Hastings padrao).
double PosteriorAproximation::computeBetaLogDensityPosterior(int number_environments, double reference[], double target[])
{
	// Mesma forma (recentralizada em torno da posicao relativa de
	// "reference[i]", com concentracao _densityPopulation) usada para GERAR
	// a proposta em generateCandidatePosterior -- precisa ser identica aqui
	// para a razao de Metropolis-Hastings ficar correta (ver comentario la
	// sobre a recentralizacao vs. a Beta(5,3) de moda fixa original).
	double strength = _densityPopulation;
	double density = 0.0;

	for (int i = 1; i <= number_environments; i++) {
		double d = reference[i + 1];
		double e = reference[i - 1];
		double span = e - d;
		double v0 = (reference[i] - d) / span;
		double alpha1 = 1.0 + v0 * strength;
		double beta = 1.0 + (1.0 - v0) * strength;
		double value = (target[i] - d) / span;

		double lnv = log(value);
		double lnomv;
		if (lnv < -25) {
			lnomv = -value;
		} else {
			lnomv = log(1 - value);
		}
		// densidade Beta(alpha1,beta) de "value" menos o Jacobiano da
		// transformacao linear (d(target[i])/d(value) = span), necessario
		// porque a densidade e avaliada sobre target[i], nao sobre "value"
		// diretamente.
		density += lnv * (alpha1 - 1) + lnomv * (beta - 1) - prior->computeLnBeta(alpha1, beta) - log(span);
	}
	return density;
}

double PosteriorAproximation::generateUniformDeviate()
{
	double result;
	result = ignlgi() / 2147483562.0;
	return result;
}

double PosteriorAproximation::generateUniformSample()
{
	double uniform = 0;
	while (uniform == 0 || uniform == 1) 
		uniform = generateUniformDeviate();
	return uniform;
}

double PosteriorAproximation::computeMoveProbabilityPosterior
(int number_items, int number_steps, int number_environments, int index, double betaWeibull, double c, double failureTimes[], 
	double X[], double Y[], double beta, double alpha[], bool censoredData)
{  
	bool transition;// transition flag
	double d,e,f,g,logAlphaxy,uniform,Alphaxy;
	double logPiY, logPiX, lnqyx, lnqxy;
	try {
		if (censoredData){
			d = lkl->computeLogLikelihoodTypeOneCensoredData (number_items, number_steps,  
				number_environments, betaWeibull, c, Y, failureTimes);
			f = lkl->computeLogLikelihoodTypeOneCensoredData (number_items, number_steps,
				number_environments, betaWeibull, c, X, failureTimes);
		}else {
			d = lkl->computeLogLikelihoodIntervalData (number_items, number_steps,  
				number_environments, betaWeibull, c, Y, failureTimes);
			f = lkl->computeLogLikelihoodIntervalData (number_items, number_steps,
				number_environments, betaWeibull, c, X, failureTimes);
		}

		e = computeDirichletLogDensity (number_environments, Y, alpha);
		g = computeDirichletLogDensity (number_environments, X, alpha);

		logPiY = d + e;
		logPiX = f + g;
			
		lnqyx = computeBetaLogDensityPosterior(number_environments, Y, X);

		lnqxy = computeBetaLogDensityPosterior(number_environments, X, Y);

		logAlphaxy = logPiY - logPiX + lnqyx - lnqxy;

		// ANTES: quando logAlphaxy < 0 (a proposta deveria ser aceita apenas
		// com probabilidade exp(logAlphaxy) < 1), o codigo sorteava um
		// uniforme PROPRIO, usava-o para decidir aceitar/rejeitar (guardado
		// em "transition"), mas NUNCA RETORNAVA essa decisao -- em vez disso
		// retornava Alphaxy = uniform, ou seja, o proprio numero aleatorio
		// sorteado, nao a probabilidade de aceitacao. O chamador (MetropolisSampler)
		// entao compara ESSE uniforme contra OUTRO uniforme independente
		// (w = generateUniformSample(); if (w <= p) aceita), o que equivale a
		// comparar duas variaveis Uniform(0,1) i.i.d. entre si -- uma
		// probabilidade de aceitacao de exatamente 50%, TOTALMENTE
		// INDEPENDENTE do valor real de logAlphaxy, sempre que a razao MH
		// seria menor que 1. Esta e a causa raiz da divergencia observada
		// entre a posteriori calculada por este software e a Tabela 3 do
		// artigo: o algoritmo estava, na pratica, quase sempre aceitando
		// (com 50% de chance) movimentos que deveriam ser fortemente
		// rejeitados quando desfavoraveis a luz da verossimilhanca e da
		// priori, distorcendo a cadeia inteira. Corrigido para retornar a
		// probabilidade de aceitacao correta, min(1, exp(logAlphaxy)); a
		// decisao aceitar/rejeitar (comparacao com um uniforme) permanece,
		// como antes, no chamador.
		if (logAlphaxy >= 0) {
			Alphaxy = 1.0;
		} else {
			Alphaxy = exp(logAlphaxy);
		}
		return Alphaxy;

	} catch (int errCode) {
			transition = false;
			// Antes cataloga o erro mas nao retornava valor (comportamento
			// indefinido em C++ padrao). Como este e um caminho de excecao
			// (RNG nao inicializado), rejeitar o movimento (0.0) e o resultado
			// mais seguro, sem alterar o comportamento em condicoes normais.
			return 0.0;
	}
}

double PosteriorAproximation::generateCandidatePrior(int number_environments, int index, double beta, double alpha[])
{
	double a, b, d, e, xmAux, p;
	for (int i = 1; i <= number_environments; i++){
		a = beta * alpha[i], b = beta * alpha[i - 1];
		d = u[i + 1];
		e = u[i - 1];
		p = (double)rand()/RAND_MAX;
		xmAux = prior->computeBetaQuantile (a, b, p);
		u[i] = d + (e - d) * xmAux;
		prior->counterTotalEnvironmentValue[i] = prior->counterTotalEnvironmentValue[i] + u[i];
	}
	return u[index];
}

/*double PosteriorAproximation::computeLogDensityPrior(int number_environments, double beta, double alpha[])
{
	double x, y;
	double  dx, ex, dy, ey;
	double a, b;

	double density = 0.0;
	int dimension = number_environments;

	for (int i = 1; i <= dimension; i++) {
		dx = X[i + 1];
		ex = X[i - 1];
		x = (double)(X[i] - dx)/(ex - dx);

		a = beta * alpha[i], b = beta * alpha[i - 1];

		dy = Y[i + 1];
		ey = Y[i - 1];
		y = (double)(Y[i] - dy)/(ey - dy);

		double lny = log(y);

		double lnomy;
		if (lny < -25) {
			lnomy = -y;
		} else {
			lnomy = log(1 - y);
		}
		density += lny * (a - 1) + lnomy * (b - 1) - prior->computeLnBeta(a, b);
	}
	return density;
}*/

double PosteriorAproximation::computeMoveProbabilityPrior
(int number_environments, double X[], double Y[], double alphaElicitated[])
{  
	bool transition;// transition flag
	double d,f,logAlphaxy,uniform,Alphaxy;
	double logPiY, logPiX, lnqyx, lnqxy;

	try {
		d = computeDirichletLogDensity (number_environments, Y, alpha);
		logPiY = d;
			
		f = computeDirichletLogDensity (number_environments, X, alpha);
		
		logPiX = f;
			
		lnqyx = computeBetaLogDensityPosterior(number_environments, Y, X);

		lnqxy = computeBetaLogDensityPosterior(number_environments, X, Y);

		logAlphaxy = logPiY - logPiX + lnqyx - lnqxy;

		// Mesma correcao aplicada em computeMoveProbabilityPosterior: retornar
		// a probabilidade de aceitacao min(1, exp(logAlphaxy)), nao um
		// uniforme sorteado internamente (esta funcao nao e chamada pelo
		// pipeline atual, mas corrigida por consistencia/seguranca futura).
		if (logAlphaxy >= 0) {
			Alphaxy = 1.0;
		} else {
			Alphaxy = exp(logAlphaxy);
		}
		return Alphaxy;

	} catch (int errCode) {
			transition = false;
			return 0.0;
	}
}

void PosteriorAproximation::PriorMetropolisSampler (int index, int number_environments, int numberThin, int samplesBurnIn, double beta, double alpha[])
{
	double uPrior = 0;
	int i;
	sizePrior = 0;
	for (int k = 0; k <= number_environments + 1; k++){
		prior->counterTotalEnvironmentValue[k] = 0.0;
	}
	for (i = 0; i < numberThin - samplesBurnIn; i++) {
		uPrior = generateCandidatePrior (number_environments, index, beta, alpha);
		// (antes imprimia uPrior a cada iteracao -- removido: sao dezenas de
		// milhares de linhas por ambiente, sem uso pelo chamador em lote)
		valuesPrior [i][0] = uPrior;
		valuesPrior [i][1] = 1./numberThin;
	}
	sizePrior = i;
}

bool PosteriorAproximation::increaseVariation()
{
	if (_densityPopulation >= MIN_POPULATION * POPULATION_STEP) {
		_densityPopulation /= POPULATION_STEP;
		return true;
	}
	return false;
}

bool PosteriorAproximation::decreaseVariation()
{
	if (_densityPopulation <= MAX_POPULATION / POPULATION_STEP) {
		_densityPopulation *= POPULATION_STEP;
		return true;
	}
	return false;
}

void PosteriorAproximation::setMinimumVariation()
{
	_densityPopulation = MAX_POPULATION;
}

void PosteriorAproximation::setMaximumVariation()
{
	_densityPopulation = MIN_POPULATION;
}


void PosteriorAproximation::MetropolisSampler (int samples, int samplesBurnIn, int intervals, int nSkip, int index, int number_items, int number_steps, int number_environments, double betaWeibull, double timeMission, 
 int indexEnvironmentUse, double reliabilityQuantileEnvironmentUse, double reliabilityPrior[], double failureTimes[], bool censoredData)
{
	int i = 0;//counter
	int t = 0;//index of the current sample
	int count = -samplesBurnIn;//number of descarted samples
	int pNumberThin = (double)samples/nSkip;
	int counterThin = 0;//counter of the thin chain
	int weight = 1;
	double a;//value of the reliability for the environment less severe
	double b;//value of the reliability for the environment more severe
	double w;//random variable
	double p;//probabilty of acceptance
	double beta;//parameter of the beta distribution

	prior->uAux = new double [number_environments + 2];
	prior->counterTotalEnvironmentValue = new double [number_environments + 2];
	uPriorMean = new double [number_environments + 2];//prior mean values
	u = new double [number_environments + 2];//transformed reliability
	X = new double [number_environments + 2];//current array
	Y = new double [number_environments + 2];//proposal array
	
	valuesPrior = new double*[samples - samplesBurnIn];
	for (int i = 0; i < samples - samplesBurnIn; i++){
		valuesPrior[i] = new double[2];
	}

	valuesPosterior = new double*[samples - samplesBurnIn];
	for (int i = 0; i < samples - samplesBurnIn; i++){
		valuesPosterior[i] = new double[2];
	}

	valuesReduced = new double*[intervals];
	for (int i = 0; i < intervals; i++){
		valuesReduced[i] = new double[2];
	}

	sizePosterior = 0;
	_samples = samples;
	_samplesBurnIn = samplesBurnIn;
	_intervals = intervals;
	
	a = reliabilityPrior[0];
	b = reliabilityPrior[number_environments - 1];

	c = prior->computeTransformationFactor (a, b, timeMission);
	u = prior->getTransformedReliability (number_environments, c, timeMission, reliabilityPrior);

	alphaElicitated = prior->getScaleParametersElicitated (number_environments, c, indexEnvironmentUse, reliabilityQuantileEnvironmentUse, timeMission, u);
	alpha = prior->getScaleParameters (number_environments, c, indexEnvironmentUse, reliabilityQuantileEnvironmentUse, timeMission, u);
	beta = prior->getBeta();

	X[0] = u[0];
	prior->uAux[0] = u[0];
	Y[0] = u[0];
	prior->counterTotalEnvironmentValue[0] = u[0];
	uPriorMean[0] = u[0];
	
	X[number_environments + 1] = u[number_environments + 1];
	prior->uAux[number_environments + 1] = u[number_environments + 1];
	Y[number_environments + 1] = u[number_environments + 1];
	uPriorMean[number_environments + 1] = u[number_environments + 1];
	prior->counterTotalEnvironmentValue[number_environments + 1] = u[number_environments + 1];

	srand (time(0)%1000);//inicializa a semente

	PriorMetropolisSampler (index, number_environments, samples, samplesBurnIn, beta, alpha);

	// ANTES dividia por pNumberThin (=samples/nSkip, uma quantidade sem
	// relacao com o numero real de amostras acumuladas). PriorMetropolisSampler
	// acumula counterTotalEnvironmentValue[i] exatamente (samples-samplesBurnIn)
	// vezes (ver seu proprio loop "i < numberThin - samplesBurnIn", chamado
	// aqui com numberThin=samples). Dividir pelo denominador errado inflava
	// uPriorMean em ~samples/(nSkip*(samples-samplesBurnIn)) vezes -- com os
	// valores default (samples=100000, nSkip=2, samplesBurnIn=10000) um fator
	// de 100000/(2*90000) invertido, ou seja, o denominador usado (50000) era
	// MENOR que o correto (90000), inflando uPriorMean em ~1.8x e produzindo
	// valores de u fora do intervalo (0,1) e fora de ordem logo no ponto de
	// partida da cadeia -- a causa raiz da divergencia observada na posteriori.
	int priorSampleCount = samples - samplesBurnIn;
	for (int i = 1; i <= number_environments; i++){
		uPriorMean[i] = (double)prior->counterTotalEnvironmentValue[i]/priorSampleCount;
	}

	_rejected = 0;
	// ANTES: chamava setMinimumVariation(), que apesar do nome apenas
	// AJUSTA o densityPopulation para MAX_POPULATION (a proposta mais
	// ESTREITA/concentrada possivel, i.e., "minima variacao" na proposta).
	// Essa chamada ja existia no codigo original, mas ate esta sessao era
	// inofensiva porque generateCandidatePosterior/computeBetaLogDensityPosterior
	// usavam alpha=5.0/beta=3.0 fixos, ignorando _densityPopulation por
	// completo. Agora que essas duas funcoes usam _densityPopulation (ver
	// comentarios la), iniciar no extremo mais estreito faz a proposta
	// praticamente degenerar (Beta(10935,6561), quase um ponto fixo
	// independente de X) e a taxa de aceitacao cai a ~0% -- e como
	// decreaseVariation() ja esta no teto (MAX_POPULATION), o mecanismo
	// adaptativo de burn-in fica sem margem para corrigir (trava, nunca
	// ajusta). O ponto de partida correto para um esquema adaptativo e o
	// MAIS LARGO (a forma Beta(5,3) original), deixando o monitor de
	// aceitacao ESTREITAR a proposta (decreaseVariation) apenas se/quando a
	// taxa observada for baixa demais.
	setMaximumVariation();

	cerr << "Avaliando a distribuicao a Posteriori: MetropolisHastings (ambiente " << index << ")" << endl;

	X = generateCandidatePosterior (number_environments, uPriorMean, X);

	long acceptedCount = 0;
	const char *dbg = getenv("ALT_DEBUG_TRACE");

	// Proposta adaptativa (ver comentarios em generateCandidatePosterior e
	// computeBetaLogDensityPosterior): monitoramos a taxa de aceitacao numa
	// janela deslizante de ADAPT_INTERVAL iteracoes e apertamos/alargamos a
	// forma da proposta via decreaseVariation()/increaseVariation(), visando
	// uma taxa saudavel (20%-50%). CRITICO: isso so pode acontecer durante o
	// burn-in (count < 0) -- adaptar a proposta depois de comecar a gravar
	// amostras (count >= 0) tornaria a cadeia nao-homogenea no tempo, o que
	// invalidaria a razao de Metropolis-Hastings das amostras gravadas.
	// Congelamos a forma da proposta assim que o burn-in termina.
	const int ADAPT_INTERVAL = 2000;
	const double ADAPT_LOW = 0.20, ADAPT_HIGH = 0.50;
	long windowAccepts = 0;
	int windowCount = 0;
	int adjustments = 0;

	for (t = 0; t < samples; t++) {
		Y = generateCandidatePosterior (number_environments, X, Y);
		w = generateUniformSample();
		p = computeMoveProbabilityPosterior (number_items, number_steps,
				number_environments, index, betaWeibull, c, failureTimes, X, Y, beta, alpha, censoredData);
		bool accepted = (w <= p);
		if (accepted) {
			acceptedCount++;
			// ANTES: X era atualizado para Y e SO DEPOIS X[index] (agora
			// igual a Y[index]!) era gravado com o peso "weight" -- mas
			// "weight" mede quantas iteracoes consecutivas o chain ficou
			// no valor ANTIGO (pre-atualizacao), nao no novo. Isso associava
			// sistematicamente o peso de persistencia de um estado ao VALOR
			// de outro estado (o que acabou de ser aceito), distorcendo a
			// reconstrucao da distribuicao a partir de (valor, peso).
			// Corrigido: capturamos o valor antigo antes de atualizar X.
			double oldXindex = X[index];
			for (int i = 1; i <= number_environments; i++){
				X[i] = Y[i];
			}
			if (count >= 0){
				valuesPosterior [sizePosterior][0] = oldXindex;
				valuesPosterior [sizePosterior][1] = log (weight);
				weight = 1;
				sizePosterior++;
			}
		}else if (count >= 0){
			weight++;
		}

		if (count < 0) {
			windowCount++;
			if (accepted) windowAccepts++;
			if (windowCount >= ADAPT_INTERVAL) {
				double rate = (double)windowAccepts / windowCount;
				bool changed;
				if (rate < ADAPT_LOW) {
					changed = decreaseVariation();
				} else if (rate > ADAPT_HIGH) {
					changed = increaseVariation();
				} else {
					changed = false;
				}
				if (changed) adjustments++;
				if (dbg) {
					cerr << "  [adapt t=" << t << "] taxa_janela=" << rate
						 << " _densityPopulation=" << _densityPopulation << endl;
				}
				windowAccepts = 0;
				windowCount = 0;
			}
		}

		count++;
		if (dbg && (t % 10000 == 0)) {
			cerr << "  [t=" << t << "] X = ";
			for (int i = 1; i <= number_environments; i++) cerr << X[i] << " ";
			cerr << " p=" << p << endl;
		}
   }//end for t
	if (dbg) {
		cerr << "  taxa de aceitacao: " << (double)acceptedCount/samples
			 << "  sizePosterior=" << sizePosterior
			 << "  ajustes_de_proposta=" << adjustments
			 << "  _densityPopulation_final=" << _densityPopulation << endl;
	}
	cout << "Fim" << " MestropolisHastings..." << endl << endl;
}//end MetropolisSampler


// ============================================================================
// Amostragem CONJUNTA -- ver comentario extenso na declaracao destas funcoes
// em PosteriorAproximation.h. Resumo: em vez de rodar MetropolisSampler() uma
// vez POR AMBIENTE (descartando as K-1 coordenadas nao gravadas a cada
// iteracao), rodamos a MESMA cadeia MH uma UNICA vez e gravamos as K
// coordenadas de cada amostra aceita, eliminando o ruido de comparar
// marginais de cadeias/burn-ins independentes entre si.
// ============================================================================

void PosteriorAproximation::PriorMetropolisSamplerJoint (int number_environments, int numberThin, int samplesBurnIn, double beta, double alpha[])
{
	// Identica a PriorMetropolisSampler (acima, inalterada), exceto que
	// grava as K coordenadas de u[] a cada iteracao em vez de apenas
	// u[index]. generateCandidatePrior ja atualiza TODAS as coordenadas de
	// u[1..number_environments] a cada chamada, independente do "index"
	// passado (esse parametro so decide o que a funcao RETORNA, valor que
	// aqui simplesmente ignoramos) -- entao a unica mudanca real e o que
	// escolhemos gravar depois de cada chamada.
	int i;
	_numberEnvironmentsJoint = number_environments;
	for (int k = 0; k <= number_environments + 1; k++){
		prior->counterTotalEnvironmentValue[k] = 0.0;
	}

	int n = numberThin - samplesBurnIn;
	_valuesPriorByEnv = new double**[number_environments + 1];
	for (int e = 1; e <= number_environments; e++){
		_valuesPriorByEnv[e] = new double*[n];
		for (int j = 0; j < n; j++){
			_valuesPriorByEnv[e][j] = new double[2];
		}
	}

	for (i = 0; i < n; i++) {
		// "index"=1 abaixo e um valor arbitrario/sem efeito sobre o que
		// gravamos (ver explicacao acima); mantido so porque a assinatura de
		// generateCandidatePrior exige um "index" valido (1..number_environments)
		// para o calculo interno de a,b (que usa alpha[index],alpha[index-1]).
		// Isso e igual ao que a versao original faz para o "index" escolhido
		// -- so que aqui usamos TODAS as coordenadas resultantes, nao so uma.
		generateCandidatePrior (number_environments, 1, beta, alpha);
		for (int e = 1; e <= number_environments; e++){
			_valuesPriorByEnv[e][i][0] = u[e];
			_valuesPriorByEnv[e][i][1] = 1./numberThin;
		}
	}
	_sizePriorJoint = i;
}

void PosteriorAproximation::MetropolisSamplerJoint (int samples, int samplesBurnIn, int intervals, int nSkip, int number_items, int number_steps, int number_environments, double betaWeibull, double timeMission,
 int indexEnvironmentUse, double reliabilityQuantileEnvironmentUse, double reliabilityPrior[], double failureTimes[], bool censoredData)
{
	// Copia de MetropolisSampler (acima, inalterada) com duas diferencas:
	// (1) chama PriorMetropolisSamplerJoint no lugar de PriorMetropolisSampler;
	// (2) ao aceitar uma proposta, grava as K coordenadas de X (o estado ANTES
	// da atualizacao) em _valuesPosteriorByEnv[1..K], em vez de gravar so
	// X[index] em valuesPosterior. Nao ha parametro "index" porque a decisao
	// de aceitar/rejeitar (computeMoveProbabilityPosterior) sempre foi conjunta
	// sobre o vetor inteiro -- "index" no original so controlava a gravacao.
	int i = 0;
	int t = 0;
	int count = -samplesBurnIn;
	int weight = 1;
	double a, b;
	double w, p;
	double beta;

	prior->uAux = new double [number_environments + 2];
	prior->counterTotalEnvironmentValue = new double [number_environments + 2];
	uPriorMean = new double [number_environments + 2];
	u = new double [number_environments + 2];
	X = new double [number_environments + 2];
	Y = new double [number_environments + 2];

	int n = samples - samplesBurnIn;
	_numberEnvironmentsJoint = number_environments;
	_valuesPosteriorByEnv = new double**[number_environments + 1];
	for (int e = 1; e <= number_environments; e++){
		_valuesPosteriorByEnv[e] = new double*[n];
		for (int j = 0; j < n; j++){
			_valuesPosteriorByEnv[e][j] = new double[2];
		}
	}

	// BUG encontrado nesta refatoracao: valuesReduced (buffer de trabalho de
	// reduce()/reducePrior(), usado por computePriorParameters/
	// computePosteriorParameters -- ORIGINAIS, chamadas via os wrappers
	// *ForEnv) e alocado em MetropolisSampler (a versao original, por-
	// ambiente) mas essa alocacao tinha ficado de fora desta copia,
	// deixando valuesReduced com o ponteiro indefinido default do objeto
	// recem-criado -- reduce()/reducePrior() escreviam nesse ponteiro
	// invalido, causando segmentation fault (confirmado com AddressSanitizer
	// apontando para "valuesReduced[k][0] = mean;" em reducePrior).
	valuesReduced = new double*[intervals];
	for (int i = 0; i < intervals; i++){
		valuesReduced[i] = new double[2];
	}

	_sizePosteriorJoint = 0;
	_samples = samples;
	_samplesBurnIn = samplesBurnIn;
	_intervals = intervals;

	a = reliabilityPrior[0];
	b = reliabilityPrior[number_environments - 1];

	c = prior->computeTransformationFactor (a, b, timeMission);
	u = prior->getTransformedReliability (number_environments, c, timeMission, reliabilityPrior);

	alphaElicitated = prior->getScaleParametersElicitated (number_environments, c, indexEnvironmentUse, reliabilityQuantileEnvironmentUse, timeMission, u);
	alpha = prior->getScaleParameters (number_environments, c, indexEnvironmentUse, reliabilityQuantileEnvironmentUse, timeMission, u);
	beta = prior->getBeta();

	X[0] = u[0];
	prior->uAux[0] = u[0];
	Y[0] = u[0];
	prior->counterTotalEnvironmentValue[0] = u[0];
	uPriorMean[0] = u[0];

	X[number_environments + 1] = u[number_environments + 1];
	prior->uAux[number_environments + 1] = u[number_environments + 1];
	Y[number_environments + 1] = u[number_environments + 1];
	uPriorMean[number_environments + 1] = u[number_environments + 1];
	prior->counterTotalEnvironmentValue[number_environments + 1] = u[number_environments + 1];

	srand (time(0)%1000);

	PriorMetropolisSamplerJoint (number_environments, samples, samplesBurnIn, beta, alpha);

	int priorSampleCount = samples - samplesBurnIn;
	for (int i = 1; i <= number_environments; i++){
		uPriorMean[i] = (double)prior->counterTotalEnvironmentValue[i]/priorSampleCount;
	}

	_rejected = 0;
	setMaximumVariation();

	cerr << "Avaliando a distribuicao a Posteriori: MetropolisHastings (conjunta, " << number_environments << " ambientes)" << endl;

	X = generateCandidatePosterior (number_environments, uPriorMean, X);

	long acceptedCount = 0;
	const char *dbg = getenv("ALT_DEBUG_TRACE");

	const int ADAPT_INTERVAL = 2000;
	const double ADAPT_LOW = 0.20, ADAPT_HIGH = 0.50;
	long windowAccepts = 0;
	int windowCount = 0;
	int adjustments = 0;

	double *oldX = new double[number_environments + 2];

	for (t = 0; t < samples; t++) {
		Y = generateCandidatePosterior (number_environments, X, Y);
		w = generateUniformSample();
		p = computeMoveProbabilityPosterior (number_items, number_steps,
				number_environments, 1, betaWeibull, c, failureTimes, X, Y, beta, alpha, censoredData);
		bool accepted = (w <= p);
		if (accepted) {
			acceptedCount++;
			// Diferenca-chave em relacao a MetropolisSampler original: aqui
			// capturamos o estado ANTIGO de TODAS as K coordenadas (nao so de
			// "index"), pois todas serao gravadas (uma por ambiente) abaixo.
			for (int e = 1; e <= number_environments; e++){
				oldX[e] = X[e];
			}
			for (int i = 1; i <= number_environments; i++){
				X[i] = Y[i];
			}
			if (count >= 0){
				for (int e = 1; e <= number_environments; e++){
					_valuesPosteriorByEnv[e][_sizePosteriorJoint][0] = oldX[e];
					_valuesPosteriorByEnv[e][_sizePosteriorJoint][1] = log (weight);
				}
				weight = 1;
				_sizePosteriorJoint++;
			}
		}else if (count >= 0){
			weight++;
		}

		if (count < 0) {
			windowCount++;
			if (accepted) windowAccepts++;
			if (windowCount >= ADAPT_INTERVAL) {
				double rate = (double)windowAccepts / windowCount;
				bool changed;
				if (rate < ADAPT_LOW) {
					changed = decreaseVariation();
				} else if (rate > ADAPT_HIGH) {
					changed = increaseVariation();
				} else {
					changed = false;
				}
				if (changed) adjustments++;
				if (dbg) {
					cerr << "  [adapt t=" << t << "] taxa_janela=" << rate
						 << " _densityPopulation=" << _densityPopulation << endl;
				}
				windowAccepts = 0;
				windowCount = 0;
			}
		}

		count++;
		if (dbg && (t % 10000 == 0)) {
			cerr << "  [t=" << t << "] X = ";
			for (int i = 1; i <= number_environments; i++) cerr << X[i] << " ";
			cerr << " p=" << p << endl;
		}
   }//end for t

	delete [] oldX;

	if (dbg) {
		cerr << "  taxa de aceitacao: " << (double)acceptedCount/samples
			 << "  sizePosterior=" << _sizePosteriorJoint
			 << "  ajustes_de_proposta=" << adjustments
			 << "  _densityPopulation_final=" << _densityPopulation << endl;
	}
	cout << "Fim" << " MestropolisHastings (conjunta)..." << endl << endl;
}//end MetropolisSamplerJoint

void PosteriorAproximation::computePriorParametersForEnv (int env, int intervals, double t)
{
	// Aponta sizePrior/valuesPrior (usados por computePriorParameters,
	// ORIGINAL e inalterada) para os dados do ambiente "env" coletados por
	// PriorMetropolisSamplerJoint, e delega para ela -- reaproveitando 100%
	// da logica original de reducao/CDF sem duplicar codigo.
	sizePrior = _sizePriorJoint;
	valuesPrior = _valuesPriorByEnv[env];
	computePriorParameters(intervals, t);
}

void PosteriorAproximation::computePosteriorParametersForEnv (int env, int nSkip, int numberThin, int intervals, double t)
{
	sizePosterior = _sizePosteriorJoint;
	valuesPosterior = _valuesPosteriorByEnv[env];
	computePosteriorParameters(nSkip, numberThin, intervals, t);
}


 /* Pre-condition: the array A is a list on [l..r] in some order.
 * Post-condition: the array A on [1..r] is in order.
 * Procedure: recursive QuickSort.  The last value is taken as a pivot,
 *  and its final position in the array is calculated.  Then the two
 *  halves of the array, those before the pivot and those after, are
 *  sorted using the same algorithm.  The base case of the recursion
 *  is when l=r, in which case the array of 1 element is in order as is.
 */	  
void PosteriorAproximation::QuickSort(double A[], int l, int r)
{
	// BUG encontrado nesta refatoracao (nao presente nos testes anteriores
	// porque so aparece com listas grandes o suficiente): o pivo original e
	// SEMPRE o ultimo elemento (A[r]), sem nenhuma aleatorizacao. Para uma
	// entrada ja ordenada ou quase-ordenada (o que acontece com
	// razoavel frequencia aqui, pois valuesPrior/valuesPosterior sao
	// preenchidos por uma cadeia MCMC que tende a produzir sequencias
	// razoavelmente monotonas), esse QuickSort degenera para o pior caso
	// O(n) de PROFUNDIDADE de recursao (em vez de O(log n)) -- com n na
	// casa de centenas de milhares de amostras (ex.: 900 mil, com
	// samples=1.000.000/samplesBurnIn=100.000), isso estoura a pilha de
	// chamadas e causa segmentation fault (confirmado empiricamente ao
	// testar a amostragem conjunta com 1 milhao de iteracoes).
	// Corrigido trocando A[r] por um elemento ALEATORIO de [l,r] antes de
	// definir o pivo -- o restante do algoritmo (particao de Hoare) e
	// mantido 100% identico ao original.
	if (r > l) {
		int pivotIndex = l + rand() % (r - l + 1);
		double tempPivot = A[pivotIndex];
		A[pivotIndex] = A[r];
		A[r] = tempPivot;
	}

	double pivot= A[r];
    int i = l-1;
    int j = r;
    double temp;

    if (r>l)
      {
	do
	  {
	    do i++; while (A[i]<pivot);
	    do j--; while (A[j]>pivot);
	    temp = A[i];
	    A[i] = A[j];
	    A[j] = temp;
	  }
		while (j>i);
		A[j] = A[i];
		A[i] = pivot;
		A[r] = temp;
		QuickSort(A,l,i-1);
		QuickSort(A,i+1,r);
      }
  }

  void PosteriorAproximation::QuickSort(double A[],double P[], int l, int r)
  {
	// Mesma correcao de pivo aleatorio aplicada na outra sobrecarga de
	// QuickSort acima (ver comentario la): evita o estouro de pilha por
	// recursao O(n) em entradas grandes/quase-ordenadas.
	if (r > l) {
		int pivotIndex = l + rand() % (r - l + 1);
		double tA = A[pivotIndex]; A[pivotIndex] = A[r]; A[r] = tA;
		double tP = P[pivotIndex]; P[pivotIndex] = P[r]; P[r] = tP;
	}

	double pivot= A[r];
	double prob = P[r];
    int i = l-1;
    int j = r;
    double temp;
	double tempP;
            
    if (r>l)
      {
	do
	  {
	    do i++; while (A[i]<pivot);
	    do j--; while (A[j]>pivot);
	    temp = A[i];
	    tempP = P[i];
		A[i] = A[j];
		P[i] = P[j];
	    A[j] = temp;
		P[j] = tempP;
	  }
		while (j>i); 
		A[j] = A[i];
		P[j] = P[i];
		A[i] = pivot;
		P[i] = prob;
		A[r] = temp;
		P[r] = tempP;
		QuickSort(A,P,l,i-1);
		QuickSort(A,P,i+1,r);
      }
  }

  
void PosteriorAproximation::computePriorParameters (int intervals, double t)
{
	double u = 0.0, p = 0.0;
	int numberThin = sizePrior;//size of the thin chain
	ratePrior = new double [numberThin];
	relPrior = new double [numberThin];
	probabilityPrior = new double [numberThin];
	
	for (int j = 0; j < sizePrior; j++)
	{	
		u = valuesPrior[j][0];
		p = valuesPrior[j][1];
		ratePrior[j] = -(double)log (u)/c;//prior rate failure	
		relPrior[j] = exp (-ratePrior[j]*t);//prior reliability
		probabilityPrior[j] = p;//move probability of Markov chain
	}//end for

	QuickSort (ratePrior, probabilityPrior, 0, sizePrior - 1);	
	QuickSort (relPrior, probabilityPrior, 0, sizePrior - 1);

	cout << "Reduzindo dados a priori..." << endl << endl;

	fout << "Failure Rate Prior" << "\t" << "PDF" << "\t" << "CDF" << endl;
	reducePrior (intervals, sizePrior, true, ratePrior, probabilityPrior);

	fout << "Reliability Prior" << "\t" << "PDF" << "\t" << "CDF" << endl;
	reducePrior (intervals, sizePrior, true, relPrior, probabilityPrior);
}

void PosteriorAproximation::computePosteriorParameters (int nSkip, int numberThin, int intervals, double t)
{
	double u = 0.0, p = 0.0;
	numberThin = (int)sizePosterior;//size of the thin chain
	ratePosterior = new double [numberThin];
	relPosterior = new double [numberThin];
	probabilityPosterior = new double [numberThin];
	
	for (int j = 0; j < sizePosterior; j++)
	{	
		u = valuesPosterior[j][0];
		p = valuesPosterior[j][1];
		ratePosterior[j] = -(double)log (u)/c;//Taxa de Falha	
		relPosterior[j] = exp (-ratePosterior[j]*t);//Confiabilidade a posteriori
		probabilityPosterior[j] = p;//Probabilidade de deslocamento de cadeia de Markov
	}//end for

	QuickSort (ratePosterior, probabilityPosterior, 0, sizePosterior - 1);	
	QuickSort (relPosterior, probabilityPosterior, 0, sizePosterior - 1);
		
	cout << "Reduzindo " << "dados " << "a" << " posteriori..." << endl << endl;
	fout << "Failure" <<  " Rate" << " Posterior" << "\t" << "PDF" << "\t" << "CDF" << endl;
	reduce (intervals, sizePosterior, true, ratePosterior, probabilityPosterior);

	fout << "Reliability" << " Posterior" << "\t" << "PDF" << "\t" << "CDF" << endl;
	reduce (intervals, sizePosterior, true, relPosterior, probabilityPosterior);
}

void PosteriorAproximation::reduce(int intervals, int size, bool logScale, double varOrder[], double logProbability[])
{
	double value = 0;
	double mean = 0;
	double prob = 0;
	double x = 0;
	double p = 0;
	double F = 0;
	sumOfProbabilities = 0.0;
	normalizer = 0.0;

	int index = 0;
	int oldIndex = 0;
	int prevIndex = 0;

	double minValue = varOrder[0];
	double maxValue = varOrder[size - 1];
	double maxLogWeight;

	maxLogWeight = logProbability[0];
	for (int h = 1; h < size; h++) {
		if (logProbability[h] > maxLogWeight)
			maxLogWeight = logProbability[h];
	}
	int k = 0;
	
	for (int i = 0; i < intervals ; i++) {

		switch (logScale) {

		case true:
			value = minValue * pow(maxValue / minValue, (1. + i) / intervals);
			break;

		case false:
			value = minValue + (maxValue - minValue) * (i + 1.) / intervals;
			break;
		}

		if (i == intervals - 1)
			index = size - 1 ;
		else {
			// move to first point beyond value
			while (varOrder[index] <= value && index < size) {
				index++;
			}
		}

		reduceInterval(oldIndex,index - 1, size, maxLogWeight, mean, prob, varOrder, logProbability);

		if (prob > 0) { 
			// skip intervals that do not have any content
			valuesReduced[k][0] = mean;
			valuesReduced[k][1] = prob;
			k++;
		}
		oldIndex = index;
	}

	// Antes: normalizer = exp(maxLogWeight) + sumOfProbabilities;
	// exp(maxLogWeight) esta em escala absoluta (log do peso bruto), enquanto
	// sumOfProbabilities e a soma de pesos RELATIVOS exp(logWeight_i - maxLogWeight),
	// cada termo <= 1. Somar os dois misturava escalas e fazia o CDF final nao
	// chegar a 1.0 (confirmado empiricamente: ~0.997 em vez de 1.0), distorcendo
	// mediana/media/moda de forma dependente do maior "peso" observado na cadeia.
	// O normalizador correto e apenas a soma dos pesos relativos.
	normalizer = sumOfProbabilities;

	for (int j = 0; j < k; j++){
		x = valuesReduced[j][0];
		p = (double)valuesReduced[j][1]/normalizer;
		F = F + p;
		fout << x << "\t" << p << "\t" << F << endl;
	}
	fout << endl << endl;
}

void PosteriorAproximation::reduceInterval(int start, int end, int size, double maxLogWeight, double & mean, double & prob, double varOrder[], double logWeightValues[])
{
	prob = 0.0;
	mean = 0.0;
	double sum = 0;
	double p = 0.0;

	if (start < 0) start = 0;
	if (end >= size) end = size - 1;

	for (int i = start; i <= end ; i++) {
		p = exp (logWeightValues[i] - maxLogWeight);
		prob += p;
		sum += varOrder[i] * p;
	}

	sumOfProbabilities += prob;

	if (prob > 0) 
		mean = sum / prob;
	else
		mean = 0.0;
}

void PosteriorAproximation::reducePrior(int intervals, int size, bool logScale, double varOrder[], double probability[])
{
	double value = 0;
	double mean = 0;
	double prob = 0;
	double x = 0;
	double p = 0;
	double F = 0;
	normalizer = 0.0;

	int index = 0;
	int oldIndex = 0;
	int prevIndex = 0;

	double minValue = varOrder[0];
	double maxValue = varOrder[size - 1];
	int k = 0;
	
	for (int i = 0; i < intervals ; i++) {

		switch (logScale) {

		case true:
			value = minValue * pow(maxValue / minValue, (1. + i) / intervals);
			break;

		case false:
			value = minValue + (maxValue - minValue) * (i + 1.) / intervals;
			break;
		}

		if (i == intervals - 1)
			index = size - 1 ;
		else {
			// move to first point beyond value
			while (varOrder[index] <= value && index < size) {
				index++;
			}
		}

		reduceIntervalPrior(oldIndex,index - 1, size, mean, prob, varOrder, probability);

		if (prob > 0) { 
			// skip intervals that do not have any content
			valuesReduced[k][0] = mean;
			valuesReduced[k][1] = prob;
			k++;
		}
		oldIndex = index;
	}

	for (int j = 0; j < k; j++){
		x = valuesReduced[j][0];
		p = (double)valuesReduced[j][1]/normalizer;
		F = F + p;
		fout << x << "\t" << p << "\t" << F << endl;
	}
	fout << endl << endl;
}

void PosteriorAproximation::reduceIntervalPrior(int start, int end, int size, double & mean, double & prob, double varOrder[], double probability[])
{
	prob = 0.0;
	mean = 0.0;
	double sum = 0;

	if (start < 0) start = 0;
	if (end >= size) end = size - 1;

	for (int i = start; i <= end ; i++) {
		prob += probability[i];
		sum += varOrder[i] * probability[i];
	}

	normalizer += prob;

	if (prob > 0) 
		mean = sum / prob;
	else
		mean = 0.0;
}

PosteriorAproximation::~PosteriorAproximation (void){
		delete [] prior->counterTotalEnvironmentValue;
		delete []prior->uAux;
		delete []u;
		delete []X;
		delete []Y;
		delete []alpha;
		delete []uPriorMean;
		delete []ratePrior;
		//delete []relPrior;
		delete []ratePosterior;
		delete []relPosterior;
		for (int i = 0; i < _samples - _samplesBurnIn; i++){
			delete *valuesPrior;
			delete *valuesPosterior;
		}
		for (int j = 0; j < _intervals; j++){
			delete *valuesReduced;
		}
		cout << "ainda bem" << endl;	
  }