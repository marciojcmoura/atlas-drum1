#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <math.h>
#include "LikelihoodFunction.h"

/*
	OBS.: O indice '0' corresponde a um ambiente que, na verdade, nao existe
	(placeholder para "nenhum ambiente anterior" no primeiro passo de cada item).

	Os arrays testPattern/tAux/sigma eram originalmente hardcoded para um unico
	cenario fixo (12 itens, 5 passos, o exemplo da Secao 5 do artigo). Agora sao
	alocados dinamicamente em tempo de execucao a partir do arquivo de entrada,
	via setTestPlan(), para suportar qualquer numero de itens/passos.
*/
int **testPattern = 0;
double **tAux = 0;
double **sigma = 0;

void setTestPlan(int number_items, int number_steps,
				  int **env, double **times, double **ramp)
{
	testPattern = new int*[number_items];
	tAux = new double*[number_items];
	sigma = new double*[number_items];
	for (int j = 0; j < number_items; j++) {
		testPattern[j] = new int[number_steps + 1];
		tAux[j] = new double[number_steps + 1];
		sigma[j] = new double[number_steps];
		for (int i = 0; i <= number_steps; i++) {
			testPattern[j][i] = env[j][i];
			tAux[j][i] = times[j][i];
		}
		for (int i = 0; i < number_steps; i++) {
			sigma[j][i] = ramp[j][i];
		}
	}
}

void freeTestPlan(int number_items)
{
	if (!testPattern) return;
	for (int j = 0; j < number_items; j++) {
		delete [] testPattern[j];
		delete [] tAux[j];
		delete [] sigma[j];
	}
	delete [] testPattern;
	delete [] tAux;
	delete [] sigma;
	testPattern = 0; tAux = 0; sigma = 0;
}

/**
* 	
   * @param x - 
   * @param y - 
   * @return the segimal value between x and y
*/
double LikelihoodFunction::getMin(double x, double y)
{
	if (x <= y)
	{
		return x;
	}
	return y;
}

 /*
	* int number_items - number of test items subjected to ALT
	* int number_environments - number of candidate test environments
	* return the environments matrix with the stress patterns which the test items will be subjected
	A = | a1,1 ... a1,K |
		|  ...........  |
		|  ...........  |
		| aN,1 ... aN,K |, where N = number_items, K = number_environments and
						   ai,j specifies the indices of the environments for
						   each test item j in each step i;
*/

double **LikelihoodFunction::getEnvironmentMatrix (int number_items, int number_environments){
	double **MAT_environments = new double*[number_items];
	for (int i = 0; i < number_items; i++){
		MAT_environments[i] = new double[number_environments];
	}
	return MAT_environments;
}

/*
	* int number_items - number of test items subjected to ALT
	* int number_environments - number of candidate test environments
	* return the time matrix with the elapsed times for each test item
	T = | t1,1 ... t1,K |
		|  ...........  |
		|  ...........  |
		| tN,1 ... tN,K |, where N = number_items, K = number_environments and
						   ti,j specifies the elapsed times of each test item j in each step i;
*/

double **LikelihoodFunction::getTimeMatrix (int number_items, int number_environments){
	double **MAT_time = new double*[number_items];
	for (int i = 0; i < number_items; i++){
		MAT_time[i] = new double[number_environments];
	}
	return MAT_time;
}

/**
   * 	
   * @param number_environments - number of test environments
   * @param indexInterval - 
   * @param betaWeibull - parameter of the Weibull distribution
   * @param c - transformation factor
   * @param r -
   * @param u[] -
   * @param t[] -
   * @param sigma[] -
   * @return the failure rate
   */

double LikelihoodFunction::computeFailureRate (int k, int vd, int ve, double betaWeibull, double c,
					   double r, double u[], double t[], double sigma){
	double failureRate = 0.0;
	if (ve == 1){
		if (t[k] <= r && r < t[k + 1]){
			failureRate = -(double)log (u[ve])/c;
		}
	}else if (t[k] <= r && r < t[k] + sigma){
		failureRate = 
			(double)(log(u[vd]) - log(u[ve]))/(c*sigma)*(r - t[k]) - (double)(log(u[vd]))/(c);
	}else if (t[k] + sigma < r && r <= t[k + 1]){
			failureRate = -(double)log (u[ve])/c;
	}
	return failureRate;
}

/**
   * 	
   * @param number_environments - number of test environments
   * @param indexInterval - 
   * @param betaWeibull - parameter of the Weibull distribution
   * @param c - transformation factor
   * @param r -
   * @param u[] -
   * @param t[] -
   * @param sigma[] -
   * @return reliability
   */

double LikelihoodFunction::computeReliability (int k, int vd, int ve, double betaWeibull, double c,
					   double r, double u[], double t[], double sigma){
	double rel = 0.0;
	if (t[k] < r && r <= t[k] + sigma){
		rel =
			pow(u[vd], (double)(-pow(r - t[k], 2) + 2*sigma*(r - t[k]))/(2*c*sigma))
			* pow (u[ve], (double)pow(r - t[k], 2)/(2*c*sigma));
	}else if (t[k] + sigma < r && r <= t[k + 1]){
		rel = pow (u[vd], (double)0.5*pow (sigma,betaWeibull)/c) * pow (u[ve], (double)(pow (r, betaWeibull) - pow (t[k] + 0.5*sigma, betaWeibull))/c);
	}
	return rel;
}

/**
   * 	
   * @param number_items - number of test items
   * @param number_environments - number of test environments
   * @param betaWeibull - parameter of the Weibull distribution
   * @param c - transformation factor
   * @param u[] -
   * @param t[] -
   * @param q[] -
   * @param sigma[] -
   * @return the value of likelihood strategy Interval Data
   */

double LikelihoodFunction::computeLogLikelihoodIntervalData 
	(int number_items, int number_steps, int number_environments, double betaWeibull, double c, 
					double u[], double failureTimes[])
{
		bool isFault;
	int n = 1;//number of times that test item j visits environment Ee
	int k = 0;//indicate the current step
	int vd; //indica o ambiente de teste imediatamente anterior ao corrente
	int ve; //indicate the current test environment
	double sum = 0.0, R, h;
	double *t = new double [number_steps + 1];

	for (int j = 0; j < number_items; j++){
		for (int i = 0; i < number_steps + 1; i++){
			t[i] = tAux[j][i];
		}
		isFault = false;
		k = 0;
		do {
			vd = testPattern[j][k];
			ve = testPattern[j][k + 1];
			if (t[k + 1] + sigma[j][k] <= failureTimes[j]){//the test item j do not failed or censored // k < (int)q[j]
				if (t[k + 1] + sigma[j][k] == failureTimes[j]){
					isFault = true;//censored
				}
				R = computeReliability (k, vd, ve, betaWeibull, c, t[k + 1], u, t, sigma[j][k]);
				sum = sum + log(R);
			}else {//the test item j either failed 
				R = computeReliability (k, vd, ve, betaWeibull, c, failureTimes[j], u, t, sigma[j][k]);
				sum = sum + log(1.0 - R);
				isFault = true;
			}
			k++;
		}while (!isFault);
	}
	delete []t;
	return sum;
}

/**
   * 	
   * @param number_items - number of test items
   * @param number_environments - number of test environments
   * @param betaWeibull - parameter of the Weibull distribution
   * @param c - transformation factor
   * @param u[] -
   * @param t[] -
   * @param q[] -
   * @param sigma[] -
   * @return the value of likelihood strategy Censored Type I Data
   */

double LikelihoodFunction::computeLogLikelihoodTypeOneCensoredData (int number_items, int number_steps, int number_environments, double betaWeibull, double c, 
					double u[], double failureTimes[])
{
	bool isFault;
	int n = 1;//number of times that test item j visits environment Ee
	int k = 0;//indicate the current step
	int vd; //indica o ambiente de teste imediatamente anterior ao corrente
	int ve; //indicate the current test environment
	double sum = 0.0, R = 0.0, h = 0.0, ramp_time;
	double *t = new double [number_steps + 1];

	for (int j = 0; j < number_items; j++){
		for (int i = 0; i < number_steps + 1; i++){
			t[i] = tAux[j][i];
		}
		isFault = false;
		k = 0;

		ramp_time = 0.0;

		do {
			vd = testPattern[j][k];
			ve = testPattern[j][k + 1];
			ramp_time = sigma[j][k];
			if (t[k + 1] <= failureTimes[j]){//the test item j do not failed or censored // k < (int)q[j]
				if (t[k + 1] == failureTimes[j]){
					isFault = true;//censored
				}
				R = computeReliability (k, vd, ve, betaWeibull, c, t[k + 1], u, t, ramp_time);
				sum = sum + log(R);
			}else {//the test item j either failed 
				R = computeReliability (k, vd, ve, betaWeibull, c, failureTimes[j], u, t, ramp_time);
				h = computeFailureRate (k, vd, ve, betaWeibull, c, failureTimes[j], u, t, ramp_time);
				sum = sum + log(h) + log(R);
				isFault = true;
			}
			k++;
		}while (!isFault);
	}
	delete []t;
	return sum;
}