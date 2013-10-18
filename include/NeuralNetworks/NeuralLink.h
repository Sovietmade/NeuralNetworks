/*
 * NeuralLink.h
 *
 *  Created on: Sep 19, 2013
 *      Author: cheryuri
 */

#ifndef NEURALLINK_H_
#define NEURALLINK_H_

template <typename T>
class Neuron;



template <typename T>
class NeuralLink
{
public:
					NeuralLink( ) : mWeightToNeuron( 0.0 ),  mNeuronLinkedTo( 0 ), mWeightCorrectionTerm( 0 ), mErrorInformationTerm( 0 ), mLastTranslatedSignal( 0 ){ };
					NeuralLink( Neuron<T> * inNeuronLinkedTo, double inWeightToNeuron = 0.0 ) :  mWeightToNeuron( inWeightToNeuron ), mNeuronLinkedTo( inNeuronLinkedTo ), mWeightCorrectionTerm( 0 ), mErrorInformationTerm( 0 ), mLastTranslatedSignal( 0 ){ };

	void			SetWeight( const double& inWeight ){ mWeightToNeuron = inWeight; };
	const double& 	GetWeight( ){ return mWeightToNeuron; };

	void			SetNeuronLinkedTo( Neuron<T> * inNeuronLinkedTo ){ mNeuronLinkedTo = inNeuronLinkedTo; };
	Neuron<T> *		GetNeuronLinkedTo( ){ return mNeuronLinkedTo; };

	void			SetWeightCorrectionTerm( double inWeightCorrectionTerm ){ mWeightCorrectionTerm = inWeightCorrectionTerm; };
	double			GetWeightCorrectionTerm( ){ return mWeightCorrectionTerm; };

	void			UpdateWeight( ){ mWeightToNeuron = mWeightToNeuron + mWeightCorrectionTerm; };

	double			GetErrorInFormationTerm( ){ return mErrorInformationTerm; };
	void			SetErrorInFormationTerm( double inEITerm ){ mErrorInformationTerm = inEITerm; };

	void			SetLastTranslatedSignal( double inLastTranslatedSignal ){ mLastTranslatedSignal = inLastTranslatedSignal; };
	double			GetLastTranslatedSignal( ){ return mLastTranslatedSignal; };
protected:
	double 			mWeightToNeuron;
	Neuron<T> * 	mNeuronLinkedTo;
	double			mWeightCorrectionTerm;
	double			mErrorInformationTerm;
	double			mLastTranslatedSignal;
};


#endif /* NEURALLINK_H_ */
