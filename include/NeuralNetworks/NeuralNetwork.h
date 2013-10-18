/*
 * NeuralNetwork.h
 *
 *  Created on: Sep 20, 2013
 *      Author: cheryuri
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_



#include "NeuronFactory.h"
#include "trainAlgorithm.h"
#include <string.h>
#include <iostream>

template <typename T>
class TrainAlgorithm;

/**
 *	Neural network class.
 *	An object of that type represents a neural network of several types:
 *	- Single layer perceptron;
 *	- Multiple layers perceptron.
 *
 * 	There are several training algorithms available as well:
 * 	- Perceptron;
 * 	- Backpropagation.
 *
 * 	How to use this class:
 * 	To be able to use neural network , you have to create an instance of that class, specifying
 * 	a number of input neurons, output neurons, number of hidden layers and amount of neurons in hidden layers.
 * 	You can also specify a type of neural network, by passing a string with a name of neural network, otherwise
 * 	MultiLayerPerceptron will be used. ( A training algorithm can be changed via public calls);
 *
 * 	Once the neural network was created, all u have to do is to set the biggest MSE required to achieve during
 * 	the training phase ( or u can skip this step, then mMinMSE will be set to 0.01 ),
 * 	train the network by providing a training data with target results.
 * 	Afterwards u can obtain the net response by feeding the net with data;
 *
*/

template <typename T>
class NeuralNetwork
{

public:

	 /**
	 * 		A Neural Network constructor.
	 * 		- Description:		A template constructor. T is a data type, all the nodes will operate with. Create a neural network by providing it with:
	 * 							@param inInputs - an integer argument - number of input neurons of newly created neural network;
	 * 							@param inOutputs- an integer argument - number of output neurons of newly created neural network;
	 * 							@param inNumOfHiddenLayers - an integer argument - number of hidden layers of newly created neural network, default is 0;
	 * 							@param inNumOfNeuronsInHiddenLayers - an integer argument - number of neurons in hidden layers of newly created neural network ( note that every hidden layer has the same amount of neurons), default is 0;
	 * 							@param inTypeOfNeuralNetwork - a const char * argument - a type of neural network, we are going to create. The values may be:
	 * 							<UL>
	 * 								<LI>MultiLayerPerceptron;</LI>
	 * 								<LI>Default is MultiLayerPerceptron.</LI>
	 *							</UL>
	 * 		- Purpose:			Creates a neural network for solving some interesting problems.
	 * 		- Prerequisites:	The template parameter has to be picked based on your input data.
	 *
	 */
					NeuralNetwork( const int& inInputs,
						const int& inOutputs,
						const int& inNumOfHiddenLayers = 0,
						const int& inNumOfNeuronsInHiddenLayers = 0,
						const char * inTypeOfNeuralNetwork = "MultiLayerPerceptron"
					);

					~NeuralNetwork( );

	 /**
	 * 		Public method Train.
	 *		- Description:		Method for training the network.
	 *		- Purpose:			Trains a network, so the weights on the links adjusted in the way to be able to solve problem.
	 *		- Prerequisites:
	 *			@param inData 	- a vector of vectors with data to train with;
	 *			@param inTarget - a vector of vectors with target data;
	 *					  		- the number of data samples and target samples has to be equal;
	 *					  		- the data and targets has to be in the appropriate order u want the network to learn.
	 */

	bool				Train( const std::vector<std::vector<T > >& inData,
						const std::vector<std::vector<T > >& inTarget );

	 /**
	 * 		Public method GetNetResponse.
	 *		- Description:		Method for actually get response from net by feeding it with data.
	 *		- Purpose:			By calling this method u make the network evaluate the response for u.
	 *		- Prerequisites:
	 *			@param inData 	- a vector data to feed with.
	 */

	std::vector<int>		GetNetResponse( const std::vector<T>& inData );

	 /**
	 * 		Public method SetAlgorithm.
	 *		- Description:		Setter for algorithm of training the net.
	 *		- Purpose:			Can be used for dynamic change of training algorithm.
	 *		- Prerequisites:
	 *			@param inTrainingAlgorithm 	- an existence of already created object  of type TrainAlgorithm.
	 */

	void				SetAlgorithm( TrainAlgorithm<T> * inTrainingAlgorithm )		{ mTrainingAlgoritm = inTrainingAlgorithm; };

	 /**
	 * 		Public method SetNeuronFactory.
	 *		- Description:		Setter for the factory, which is making neurons for the net.
	 *		- Purpose:			Can be used for dynamic change of neuron factory.
	 *		- Prerequisites:
	 *			@param inNeuronFactory 	- an existence of already created object  of type NeuronFactory.
	 */

	void				SetNeuronFactory( NeuronFactory<T> * inNeuronFactory )		{ mNeuronFactory = inNeuronFactory; };

	 /**
	 * 		Public method ShowNetworkState.
	 *		- Description:		Prints current state to the standard output: weight of every link.
	 *		- Purpose:			Can be used for monitoring the weights change during training of the net.
	 *		- Prerequisites:	None.
	 */

	void				ShowNetworkState( );

	 /**
	 * 		Public method GetMinMSE.
	 *		- Description:		Returns the biggest MSE required to achieve during the training phase.
	 *		- Purpose:			Can be used for getting the biggest MSE required to achieve during the training phase.
	 *		- Prerequisites:	None.
	 */

	const double&			GetMinMSE( )							{ return mMinMSE; };

	 /**
	 * 		Public method SetMinMSE.
	 *		- Description:		Setter for the biggest MSE required to achieve during the training phase.
	 *		- Purpose:			Can be used for setting the biggest MSE required to achieve during the training phase.
	 *		- Prerequisites:
	 *			@param inMinMse 	- double value, the biggest MSE required to achieve during the training phase.
	 */

	void				SetMinMSE( const double& inMinMse )				{ mMinMSE = inMinMse; };

	/**
	* 		Friend class.
	*/

	friend class 			Hebb<T>;

	/**
	* 		Friend class.
	*/

	friend class 			Backpropagation<T>;

protected:

	 /**
	 * 		Protected method GetLayer.
	 *		- Description:		Getter for the layer by index of that layer.
	 *		- Purpose:			Can be used by inner implementation for getting access to neural network's layers.
	 *		- Prerequisites:
	 *			@param inInd 	-  an integer index of layer.
	 */

	std::vector<Neuron<T > *>&	GetLayer( const int& inInd ) 					{ return mLayers[inInd]; };

	/**
	 * 		Protected method size.
	 *		- Description:		Returns the number of layers in the network.
	 *		- Purpose:			Can be used by inner implementation for getting number of layers in the network.
	 *		- Prerequisites:	None.
	 */

	unsigned int			size( ) 							{ return mLayers.size( ); };

	/**
	 * 		Protected method GetNumOfOutputs.
	 *		- Description:		Returns the number of units in the output layer.
	 *		- Purpose:			Can be used by inner implementation for getting number of units in the output layer.
	 *		- Prerequisites:	None.
	 */

	std::vector<Neuron<T > *>&	GetOutputLayer( ) 						{ return mLayers[mLayers.size( )-1]; };

	/**
	 * 		Protected method GetInputLayer.
	 *		- Description:		Returns the input layer.
	 *		- Purpose:			Can be used by inner implementation for getting the input layer.
	 *		- Prerequisites:	None.
	 */

	std::vector<Neuron<T > *>&	GetInputLayer( ) 						{ return mLayers[0]; };

	/**
	 * 		Protected method GetBiasLayer.
	 *		- Description:		Returns the vector of Biases.
	 *		- Purpose:			Can be used by inner implementation for getting vector of Biases.
	 *		- Prerequisites:	None.
	 */

	std::vector<Neuron<T > *>& 	GetBiasLayer( )							{ return mBiasLayer; };

	/**
	 * 		Protected method UpdateWeights.
	 *		- Description:		Updates the weights of every link between the neurons.
	 *		- Purpose:			Can be used by inner implementation for updating the weights of links between the neurons.
	 *		- Prerequisites:	None, but only makes sense, when its called during the training phase.
	 */

	void				UpdateWeights( );

	/**
	 * 		Protected method ResetCharges.
	 *		- Description:		Resets the neuron's data received during iteration of net training.
	 *		- Purpose:			Can be used by inner implementation for reset the neuron's data between iterations.
	 *		- Prerequisites:	None, but only makes sense, when its called during the training phase.
	 */

	void				ResetCharges( );

	/**
	 * 		Protected method AddMSE.
	 *		- Description:		Changes MSE during the training phase.
	 *		- Purpose:			Can be used by inner implementation for changing MSE during the training phase.
	 *		- Prerequisites:
	 *			@param inInd 	-  a double amount of MSE to be add.
	 */

	void				AddMSE( double inPortion )					{ mMeanSquaredError += inPortion; };

	/**
	 * 		Protected method GetMSE.
	 *		- Description:		Getter for MSE value.
	 *		- Purpose:			Can be used by inner implementation for getting access to the MSE value.
	 *		- Prerequisites:	None.
	 */

	double				GetMSE( )							{ return mMeanSquaredError; };

	/**
	 * 		Protected method ResetMSE.
	 *		- Description:		Resets MSE value.
	 *		- Purpose:			Can be used by inner implementation for resetting MSE value.
	 *		- Prerequisites:	None.
	 */

	void				ResetMSE( )							{ mMeanSquaredError = 0; };


	NeuronFactory<T> *		mNeuronFactory;							/*!< Member, which is responsible for creating neurons @see SetNeuronFactory */
	TrainAlgorithm<T> *				mTrainingAlgoritm;				/*!< Member, which is responsible for the way the network will trained @see SetAlgorithm */
	std::vector<std::vector<Neuron<T > *> > 	mLayers;					/*!< Inner representation of neural networks */
	std::vector<Neuron<T > *> 			mBiasLayer;					/*!< Container for biases */
	unsigned int					mInputs, mOutputs, mHidden;			/*!< Number of inputs, outputs and hidden units */
	double						mMeanSquaredError;				/*!< Mean Squared Error which is changing every iteration of the training*/
	double						mMinMSE;					/*!< The biggest Mean Squared Error required for training to stop*/
};




#endif /* NEURALNETWORK_H_ */
