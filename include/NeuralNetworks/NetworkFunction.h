/*
 * NetworkFunction.h
 *
 *  Created on: Sep 25, 2013
 *      Author: cheryuri
 */

#ifndef NETWORKFUNCTION_H_
#define NETWORKFUNCTION_H_

#include <math.h>

class NetworkFunction {
public:
NetworkFunction(){};
virtual ~NetworkFunction(){};
virtual double 	Process( double inParam ) = 0;
virtual double 	Derivative( double inParam ) = 0;
};

class Linear : public NetworkFunction {
public:
Linear(){};
virtual ~Linear(){};
virtual double 	Process( double inParam ){ return inParam; };
virtual double 	Derivative( double inParam ){ return 0; };
};


class Sigmoid : public NetworkFunction {
public:
Sigmoid(){};
virtual ~Sigmoid(){};
virtual double 	Process( double inParam ){ return ( 1 / ( 1 + exp( -inParam ) ) ); };
virtual double 	Derivative( double inParam ){ return ( this->Process(inParam)*(1 - this->Process(inParam)) );};
};

class BipolarSigmoid : public NetworkFunction {
public:
BipolarSigmoid(){};
virtual ~BipolarSigmoid(){};
virtual double 	Process( double inParam ){ return ( 2 / ( 1 + exp( -inParam ) ) - 1 ) ;};
virtual double 	Derivative( double inParam ){ return ( 0.5 * ( 1 + this->Process( inParam ) ) * ( 1 - this->Process( inParam ) ) ); };
};

#endif /* NETWORKFUNCTION_H_ */


