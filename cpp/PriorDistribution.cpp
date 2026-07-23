#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <math.h>
#include "PriorDistribution.h"
#include "cdflib.h"
#include "ranlib.h"
using namespace std;

double PriorDistribution::generateNormalDeviate(double mu, double sigma)
{
	double result;
	result = gennor(mu,sigma);
	return result;
}

double PriorDistribution::generateBetaDeviate(double alpha, double beta)
{
	double result;
	result = genbet(alpha,beta);
	return result;
}

double PriorDistribution::generateGammaDeviate(double scale, double shape)
{
	// shape has interpretation of arrival rate
	double result;
	result = gengam(scale,shape);
	return result;
}

double PriorDistribution::computeBetaCDF(double alpha, double beta, double x)
{
	int which = 1;
	double p;
	double q;
	double y = 1 - x;
	int status;
	double bound;
	cdfbet(&which,&p,&q,&x,&y,&alpha,&beta,&status,&bound);
	if (status != 0) return 0;
	return p;
}


double PriorDistribution::computeBetaPDF(double alpha, double beta, double x)
{
	double lnbetafunc = computeLnBeta(alpha,beta);
	double lnx = log(x);
	double lnomx;

	if (lnx < -25)
		lnomx = -x;
	else
		lnomx = log(1 - x);

	return exp( (alpha - 1) * lnx + (beta - 1) * lnomx - lnbetafunc );
}

double PriorDistribution::computeBetaQuantile(double alpha, double beta, double p)
{
	int which = 2;
	double q = 1 - p;
	int status;
	double x;
	double y;
	double bound;
	cdfbet(&which,&p,&q,&x,&y,&alpha,&beta,&status,&bound);
	if (status != 0) return 0;
	return x;
}


double PriorDistribution::computeLnBeta(double a, double b)
{
	double result = 0;
	double aplusb = a + b;
	result = gamln(&a) + gamln(&b) - gamln(&aplusb);
	return result;
}

/**
 * Generalized incomplete beta function.
 */
double PriorDistribution::computeDeltaBetaCDF(double a, double b, double lo, double hi)
{
	double lox = lo;
	double loy = 1 - lo;
	double hix = hi;
	double hiy = 1 - hi;
	double locum,loccum,hicum,hiccum;

	cumbet(&lox,&loy,&a,&b,&locum,&loccum);
	cumbet(&hix,&hiy,&a,&b,&hicum,&hiccum);

	if (locum <= 0.5)
		return hicum - locum;
	else
		return loccum - hiccum;
}

/**
   * 	
   * @param a - reliability level for nominal use conditions
   * @param b - reliability level for more severe conditions
   * @return the transformation factor 'c'. This method uses the algorithm Newton-Raphson
   */


bool PriorDistribution::isEqual (double x, double y, double precision){
	bool isEqual = false;
	if (fabs(x - y) < precision){
		isEqual = true;
	}
	return isEqual;	
}

/**
   * 	
   * @param a - reliability level for nominal use conditions
   * @param b - reliability level for more severe conditions
   * @return the transformation factor 'c'. This method uses the algorithm Newton-Raphson
   */

double PriorDistribution::computeTransformationFactor(double a, double b, double t){
	double fx,fdx,xo,tol,x,d,e;
	tol = 0.00001;
	d = 0;
	e = 10000;
	xo=(d + e)/2;
	fx = pow (a, xo/t) + pow (b, xo/t) - 1.0;
	fdx = pow(a, xo/t)*log(a)*(1.0/t) + pow(b, xo/t)*log(b)*(1.0/t);
	while (fabs(fx)>tol)
	{
		x = xo - (fx/fdx);
		if (fabs(fx) > tol){
			xo=x;
		}
		fx = pow (a, xo/t) + pow (b, xo/t) - 1.0;
		fdx = pow(a, xo/t)*log(a)*(1.0/t) + pow(b, xo/t)*log(b)*(1.0/t);
	}
	return x;
}

/**
   * 	
   * @param c - transformation factor computed by method getTransformationFactor ();
   * @return an array of values of the variable 'u' between 0 e 1
   */

double *PriorDistribution::getTransformedReliability (int number_environments, double c, double t, double reliabilityPrior[]){
	double *u = new double [number_environments + 2];
	u[0] = 1.0;
	for (int i = 1; i <= number_environments; i++){
		u[i] = pow (reliabilityPrior[i - 1], c/t);
	}
	u[number_environments + 1] = 0.0;
	return u;
}

/**
   * 
   * @param alpha - parameter alpha of the distribution Beta
   * @param beta - parameter beta of the distribution Beta
   * @param q - quantile of probability
   * @return the value of a variable 'x' Beta
   */

double PriorDistribution::getBissectOne (double alpha, double beta, double q){
	double a = beta * alpha, b = beta * (1 - alpha);
	double d = 0.0, e = 1.0, xm, qm, gamma = 0.000001;
	do {
		if (!isEqual(d, e, gamma)){
			xm = (double)(d + e)/2;
			qm = computeBetaCDF (a, b, xm);
			if (qm <= q){
				d = xm;
			}else {
				e = xm;
			}
		}else {
			break;
		}
	}while (fabs (qm - q) > gamma);
	return xm;
}

/**
   * 
   * @param xu - upper limit of a variable Beta
   * @param beta - parameter beta of the distribution Beta
   * @param qu - upper quantile of probability
   * @return the value of the parameter alpha of the distribution Beta
   */

double PriorDistribution::getBissectTwo (double xu, double beta, double qu){
	double a, b;
	double d = 0.0, e = 1.0, alpha, qn, gamma = 0.000001;
	do {
		if (!isEqual(d, e, gamma)){
			alpha = (double)(d + e)/2;
			a = beta * alpha, b = beta * (1 - alpha);
			qn = computeBetaCDF (a, b, xu);
			if (qn <= qu){
				e = alpha;
			}else {
				d = alpha;
			}
		}else {
			break;
		}
	}while (fabs (qn - qu) > gamma);
	return alpha;
}	

/**
   * 
   * @param xl - lower limit of a variable Beta
   * @param xu - upper limit of a variable Beta
   * @param ql - lower quantile of probability
   * @param qu - upper quantile of probability
   * @return the value of the parameter beta of the distribution Beta
   */

double PriorDistribution::getBissectThree (double xl, double xu, double ql, double qu){
	bool isFound = false;
	int k = 0;
	double alpha, beta = 1.0, a = 0.0,b = 1.0, x, gamma = 0.000001;
    
	do {	
		if (!isFound){
			alpha = getBissectTwo (xu, beta, qu);
			x = getBissectOne (alpha, beta, ql);
			if (x < xl){
				beta = 2*beta;
				continue;
			}else {
				b = beta;
				isFound = true;
			}
		}else {
			if (!isEqual(a, b, gamma)){
				beta = (a + b)/2;
				alpha = getBissectTwo (xu, beta, qu);
				x = getBissectOne (alpha, beta, ql);
				if (x < xl){
					a = beta;
				}else {
					b = beta;
				}
			}else {
				break;
			}
		}
	}while (fabs (x - xl) > gamma);
    alphaEnvironmentUse = alpha;
	return beta;
}

/**
	@return the value of the parameter alpha of the distribution Beta for the nominal environment
*/

double PriorDistribution::getAlphaEnvironmentUse (void){
	return alphaEnvironmentUse;
}

/**
   * 
   * @param c - transformation factor
   * @param reliabilityQuantileEnvironmentUse - reliability for the nominal environment for a quantile of probability q = 5%
   * @param timeMission - time of mission for assessment reliability
   * @return an array of values of the scale parameters elicitated
   */

double *PriorDistribution::getScaleParametersElicitated 
	(int number_environments, double c, int indexEnvironmentUse, double reliabilityQuantileEnvironmentUse, double timeMission, double u[]){
	double u_environmentUse;
	double *alphaElicitated = new double [number_environments + 1];
	u_environmentUse = pow (reliabilityQuantileEnvironmentUse, c/timeMission);
	beta = getBissectThree (u_environmentUse, u[indexEnvironmentUse], 0.05, 0.5);
	alphaElicitated[indexEnvironmentUse - 1] = 1.0 - getAlphaEnvironmentUse();
	for (int i = 0; i < number_environments; i++){
		if (i != indexEnvironmentUse - 1){
			alphaElicitated [i] = 1.0 - getBissectTwo (u[i + 1], beta, 0.5);	
		}
	}
	alphaElicitated [number_environments] = 1.0;
  	return alphaElicitated;
} 

/**
	@return the value of the parameter Beta
*/

double PriorDistribution::getBeta (void){
	return beta;		
}

/**
	@return scale parameters for each test environment;
*/

/**
	@return scale parameters for each test environment;
*/

double *PriorDistribution::getScaleParameters (int number_environments, 
											   double c, int indexEnvironmentUse, double reliabilityQuantileEnvironmentUse, double timeMission, double u[]){
	double *alpha = new double [number_environments + 1];
	double a, b;
	alphaElicitated = getScaleParametersElicitated (number_environments, 
		c, indexEnvironmentUse, reliabilityQuantileEnvironmentUse, timeMission, u);
	alpha [0] = alphaElicitated [0];
	for (int j = 1; j < number_environments; j++){
		alpha [j] = alphaElicitated [j] - alphaElicitated [j - 1];
	}
	alpha [number_environments] = 1 - alphaElicitated [number_environments - 1];
	cout << "Ambiente "	<< "\t" << "ve" << "\t" << "n" << endl;
	for (int i = 0; i <= number_environments; i++)
	{
		a = alpha[i], b = getBeta();
		cout << i + 1 << "\t" << a << "\t" << b << endl;
	}
	cout << endl;
	return alpha;
}