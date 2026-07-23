// $Id: LikelihoodFunctionSignatures.h,v 1.4 2005/03/01 12:28:28 marcio Exp $

#pragma once

#include <iostream>

// Test-plan arrays, sized [number_items][number_steps+1] (testPattern, tAux)
// and [number_items][number_steps] (sigma). Populated at runtime from the
// input configuration file by setTestPlan() -- these used to be hardcoded
// arrays baked into the binary for a single fixed scenario (12 items, 5
// steps, matching the paper's Section 5 example). See setTestPlan().
extern int **testPattern;
extern double **tAux;
extern double **sigma;

// Allocates and fills testPattern/tAux/sigma from parsed input data.
// env[j] has number_steps+1 entries (env[j][0] is the unused dummy
// "environment 0" slot, matching the original convention).
// times[j] has number_steps+1 entries (interval boundaries, t_0=0).
// ramp[j] has number_steps entries (ramp time rho for each step).
void setTestPlan(int number_items, int number_steps,
				  int **env, double **times, double **ramp);
void freeTestPlan(int number_items);

class LikelihoodFunction
{
public:
	double *t;
	double getMin(double x, double y);
	double **getEnvironmentMatrix (int number_items, int number_environments);
	double **getTimeMatrix (int number_items, int number_environments);
	double computeFailureRate (int k, int vd, int ve, double betaWeibull, double c,
					   double r, double u[], double t[], double sigma);
	double computeReliability (int k, int vd, int ve, double betaWeibull, double c,
					   double r, double u[], double t[], double sigma);
	double computeLogLikelihoodIntervalData (int number_items, int number_steps, int number_environments, double betaWeibull, double c, 
					double u[], double failureTimes[]);
	double computeLogLikelihoodTypeOneCensoredData (int number_items, int number_steps, int number_environments, double betaWeibull, double c, double u[], double failureTimes[]);
};