// DistributionPoint.h: interface for the DistributionPoint class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_DISTRIBUTIONPOINT_H__08663494_4F9C_11D4_9894_002018557056__INCLUDED_)
#define AFX_DISTRIBUTIONPOINT_H__08663494_4F9C_11D4_9894_002018557056__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class DistributionPoint  
{
public:
	DistributionPoint();
	DistributionPoint(double value, double probability);
	DistributionPoint(double rate, double reliability, double probability);
	virtual ~DistributionPoint();
	double getValue() const;
	double getProbability() const;
	bool operator< (const DistributionPoint & point) const;
	bool operator<= (const DistributionPoint & point) const;
	bool operator> (const DistributionPoint & point) const;
	bool operator>= (const DistributionPoint & point) const;
	bool operator== (const DistributionPoint & point) const;
	bool operator!= (const DistributionPoint & point) const;
private:
	double _value;
	double _rate;
	double _reliability;
	double _probability;

};

#endif // !defined(AFX_DISTRIBUTIONPOINT_H__08663494_4F9C_11D4_9894_002018557056__INCLUDED_)
