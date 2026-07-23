// $Id: PriorDistributionSignatures.h,v 1.4 2005/03/01 10:37:28 marcio Exp $

#pragma once

#include <iostream>

class PriorDistribution
{
public:
	double alphaEnvironmentUse, beta;
	double *uAux;
	double *counterTotalEnvironmentValue;
	double *alphaElicitated;
			
	double generateNormalDeviate(double mu, double sigma);
	double generateBetaDeviate(double alpha, double beta);
	double generateGammaDeviate(double scale, double shape);
	double computeLnBeta(double a, double b);
	double computeBetaQuantile(double alpha, double beta, double p);
	double computeBetaCDF(double alpha, double beta, double x);
	double computeDeltaBetaCDF(double a, double b, double lo, double hi);
	double computeBetaPDF(double alpha, double beta, double x);
	bool isEqual (double x, double y, double precision);
	double computeTransformationFactor(double a, double b, double t);
	double *getTransformedReliability (int number_environments, double c, double t, double reliabilityPrior[]);
	double getBissectOne (double alpha, double beta, double q);
	double getBissectTwo (double xu, double beta, double qu);
	double getBissectThree (double xl, double xu, double ql, double qu);
	double *getProposalValue (int number_environments, 
											int index, double c, double timeMission, double beta, 
											double u[], double array[], double alpha[]);
	double getAlphaEnvironmentUse (void);
	double *getScaleParametersElicitated 
	(int number_environments, double c, int indexEnvironmentUse, double reliabilityQuantileEnvironmentUse, double timeMission, double u[]);
	double *getScaleParameters (int number_environments, 
											   double c, int indexEnvironmentUse, double reliabilityQuantileEnvironmentUse, double timeMission, double u[]);
	double getBeta (void);
};