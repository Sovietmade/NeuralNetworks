/*
 * Neuron.h
 *
 *  Created on: Sep 19, 2013
 *      Author: cheryuri
 */

#ifndef NEURON_H_
#define NEURON_H_

#include "NeuralLink.h"
#include "NetworkFunction.h"
#include <vector>
#include <iostream>

const double LearningRate = 0.01;


/**
 *	Neuron base class.
 *	Represents a basic element of neural network, node in the net's graph.
 *	There are several possibilities for creation an object of type Neuron, different constructors suites for
 *	different situations.
*/

template <typename T>
class Neuron
{
public:

	 /**
	 * 		A default Neuron constructor.
	 * 		- Description:		Creates a Neuron; general purposes.
	 * 		- Purpose:			Creates a Neuron, linked to nothing, with a Linear network function.
	 * 		- Prerequisites:	None.
	 */

									Neuron( ) : mNetFunc( new Linear ), mSumOfCharges( 0.0 )					{ };

	 /**
	 * 		A Neuron constructor based on NetworkFunction.
	 * 		- Description:		Creates a Neuron; mostly designed to create an output kind of neurons.
	 * 			@param inNetFunc - a network function which is producing neuron's output signal;
	 * 		- Purpose:			Creates a Neuron, linked to nothing, with a specific network function.
	 * 		- Prerequisites:	The existence of NetworkFunction object.
	 */

									Neuron( NetworkFunction * inNetFunc ) : mNetFunc( inNetFunc ), mSumOfCharges( 0.0 )		{ };

									Neuron( std::vector<NeuralLink<T > *>& inLinksToNeurons, NetworkFunction * inNetFunc ) :
										mNetFunc( inNetFunc ),
										mLinksToNeurons(inLinksToNeurons),
										mSumOfCharges(0.0) 									{ };

	 /**
	 * 		A Neuron constructor based on layer of Neurons.
	 * 		- Description:		Creates a Neuron; mostly designed to create an input and hidden kinds of neurons.
	 * 			@param inNeuronsLinkTo - a vector of pointers to Neurons which is representing a layer;
	 * 			@param inNetFunc - a network function which is producing neuron's output signal;
	 * 		- Purpose:			Creates a Neuron, linked to every Neuron in provided layer.
	 * 		- Prerequisites:	The existence of std::vector<Neuron *> and NetworkFunction.
	 */

									Neuron( std::vector<Neuron *>& inNeuronsLinkTo, NetworkFunction * inNetFunc );

	virtual					~Neuron( );

	virtual	std::vector<NeuralLink<T > *>&	GetLinksToNeurons( )					{ return mLinksToNeurons; };
	virtual	NeuralLink<T> *			at( const int& inIndexOfNeuralLink )			{ return mLinksToNeurons[ inIndexOfNeuralLink ]; };

	virtual	void				SetLinkToNeuron( NeuralLink<T> * inNeuralLink )		{ mLinksToNeurons.push_back( inNeuralLink ); };

	virtual	void				Input( double inInputData )				{ mSumOfCharges += inInputData; };
	virtual	double				Fire( );
	virtual	int				GetNumOfLinks( )					{ return mLinksToNeurons.size( ); };
	virtual	double				GetSumOfCharges( );
	virtual	void				ResetSumOfCharges( )					{ mSumOfCharges = 0.0; };
	virtual	double				Process( )						{ return mNetFunc->Process( mSumOfCharges ); };
	virtual	double				Process( double inArg )					{ return mNetFunc->Process( inArg ); };
	virtual	double				Derivative( )						{ return mNetFunc->Derivative( mSumOfCharges ); };

	virtual	void				SetInputLink( NeuralLink<T> * inLink )			{ mInputLinks.push_back( inLink ); };
	virtual	std::vector<NeuralLink<T > *>&	GetInputLink( )						{ return mInputLinks; };



	virtual	double							PerformTrainingProcess( double inTarget );
	virtual	void							PerformWeightsUpdating( );

	virtual	void							ShowNeuronState( );
protected:
	NetworkFunction *						mNetFunc;
	std::vector<NeuralLink<T > *>					mInputLinks;
	std::vector<NeuralLink<T > *>					mLinksToNeurons;

	double								mSumOfCharges;
};

template <typename T>
class OutputLayerNeuronDecorator : public Neuron<T>
{
public:
									OutputLayerNeuronDecorator( Neuron<T> * inNeuron )		{ mOutputCharge = 0; mNeuron = inNeuron; };
	virtual								~OutputLayerNeuronDecorator( );

	virtual std::vector<NeuralLink<T > *>&				GetLinksToNeurons( )						{ return mNeuron->GetLinksToNeurons( ) ;};
	virtual NeuralLink<T> *						at( const int& inIndexOfNeuralLink )				{ return ( mNeuron->at( inIndexOfNeuralLink ) ) ;};
	virtual void							SetLinkToNeuron( NeuralLink<T> * inNeuralLink )			{ mNeuron->SetLinkToNeuron( inNeuralLink ); };
	virtual double							GetSumOfCharges( )						{ return mNeuron->GetSumOfCharges( ); };

	virtual void							ResetSumOfCharges( )						{ mNeuron->ResetSumOfCharges( ); };
	virtual void							Input( double inInputData )					{ mNeuron->Input( inInputData ); };
	virtual double							Fire( );
	virtual int							GetNumOfLinks( )						{ return mNeuron->GetNumOfLinks( ); };


	virtual	double							Process( )							{ return mNeuron->Process( ); };
	virtual	double							Process( double inArg )						{ return mNeuron->Process( inArg ); };

	virtual	double							Derivative( )							{ return mNeuron->Derivative( ); };

	virtual void							SetInputLink( NeuralLink<T> * inLink )				{ mNeuron->SetInputLink( inLink ); };
	virtual std::vector<NeuralLink<T > *>&				GetInputLink( )							{ return mNeuron->GetInputLink( ); };

	virtual	double							PerformTrainingProcess( double inTarget );
	virtual	void							PerformWeightsUpdating( );
	virtual void							ShowNeuronState( )						{ mNeuron->ShowNeuronState( ); };
protected:
	double								mOutputCharge;
	Neuron<T> *							mNeuron;

};

template <typename T>
class HiddenLayerNeuronDecorator : public Neuron<T>
{
public:
									HiddenLayerNeuronDecorator( Neuron<T> * inNeuron )		{ mNeuron = inNeuron; };
	virtual								~HiddenLayerNeuronDecorator( );

	virtual std::vector<NeuralLink<T > *>&				GetLinksToNeurons( )						{ return mNeuron->GetLinksToNeurons( ); };
	virtual void							SetLinkToNeuron( NeuralLink<T> * inNeuralLink )			{ mNeuron->SetLinkToNeuron( inNeuralLink ); };
	virtual double							GetSumOfCharges( )						{ return mNeuron->GetSumOfCharges( ) ;};

	virtual void							ResetSumOfCharges( )						{mNeuron->ResetSumOfCharges( ); };
	virtual void							Input( double inInputData )					{ mNeuron->Input( inInputData ); };
	virtual double							Fire( );
	virtual int							GetNumOfLinks( )						{ return mNeuron->GetNumOfLinks( ); };
	virtual NeuralLink<T> *						at( const int& inIndexOfNeuralLink )				{ return ( mNeuron->at( inIndexOfNeuralLink) ); };

	virtual	double							Process( )							{ return mNeuron->Process( ); };
	virtual	double							Process( double inArg )						{ return mNeuron->Process( inArg ); };

	virtual	double							Derivative( )							{ return mNeuron->Derivative( ); };

	virtual void							SetInputLink( NeuralLink<T> * inLink )				{ mNeuron->SetInputLink( inLink ); };
	virtual std::vector<NeuralLink<T > *>&				GetInputLink( )							{ return mNeuron->GetInputLink( ); };

	virtual	double							PerformTrainingProcess( double inTarget );
	virtual	void							PerformWeightsUpdating( );

	virtual void							ShowNeuronState( )						{ mNeuron->ShowNeuronState( ); };
protected:


	Neuron<T> *							mNeuron;

};


#endif /* NEURON_H_ */
