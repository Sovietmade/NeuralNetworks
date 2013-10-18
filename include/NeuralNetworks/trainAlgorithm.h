/*
 * trainAlgorithm.h
 *
 *  Created on: Sep 24, 2013
 *      Author: cheryuri
 */

#ifndef TRAINALGORITHM_H_
#define TRAINALGORITHM_H_

#include <vector>

template <typename T>
class NeuralNetwork;

template <typename T>
class TrainAlgorithm
{
public:
		virtual 			~TrainAlgorithm(){};
		virtual double 			Train(const std::vector<T>& inData, const std::vector<T>& inTarget) = 0;
		virtual void			WeightsInitialization() = 0;
protected:
};

template <typename T>
class Hebb : public TrainAlgorithm<T>
{
public:
					Hebb(NeuralNetwork<T> * inNeuralNetwork) : mNeuralNetwork(inNeuralNetwork){};
	virtual				~Hebb(){};
	virtual double 			Train(const std::vector<T>& inData, const std::vector<T>& inTarget);
	virtual void			WeightsInitialization();
protected:
	NeuralNetwork<T> * 		mNeuralNetwork;
};

template <typename T>
class Backpropagation : public TrainAlgorithm<T>
{
public:
					Backpropagation(NeuralNetwork<T> * inNeuralNetwork);
	virtual				~Backpropagation(){};
	virtual double 			Train(const std::vector<T>& inData, const std::vector<T>& inTarget);
	virtual void			WeightsInitialization();
protected:
	void				NguyenWidrowWeightsInitialization();
	void				CommonInitialization();
	NeuralNetwork<T> * 		mNeuralNetwork;
};




#endif /* TRAINALGORITHM_H_ */
