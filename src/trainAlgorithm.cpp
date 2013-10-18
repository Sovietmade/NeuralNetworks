/*
 * trainAlgorithm.cpp
 *
 *  Created on: Sep 25, 2013
 *      Author: cheryuri
 */


#include "NeuralNetworks/trainAlgorithm.h"
#include "NeuralNetworks/NeuralNetwork.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

template <typename T>
double Hebb<T>::Train(const std::vector<T>& inData, const std::vector<T>& inTarget)
{
	if(inData.size() != mNeuralNetwork->GetInputs() || inTarget.size() != mNeuralNetwork->GetOutputs()){
			std::cout << "Input data dimensions are wrong, expected: " << mNeuralNetwork->GetInputs() << " elements\n";
			return false;
		}
	else{
		for(unsigned int indexOfData = 0; indexOfData < mNeuralNetwork->GetInputs(); indexOfData++){
			mNeuralNetwork->GetInputLayer().at(indexOfData)->Input(inData[indexOfData]);
		}

		for(unsigned int numOfLayers = 0; numOfLayers < mNeuralNetwork->size() - 1; numOfLayers++){
			mNeuralNetwork->GetBiasLayer().at(numOfLayers)->Input(1.0);

			for(unsigned int indexOfData = 0; indexOfData < mNeuralNetwork->GetLayer(numOfLayers).size(); indexOfData++){
				mNeuralNetwork->GetLayer(numOfLayers).at(indexOfData)->Fire();
			}

			mNeuralNetwork->GetBiasLayer().at(numOfLayers)->Fire();
		}



		//output inTarget vector
		std::cout << "InTarget vector: { ";
		for(unsigned int i = 0; i < inTarget.size(); i++){
			 std::cout <<  inTarget[i] << " ";
		}
		std::cout << " }\n";


		std::vector<int> netResponse;
		for(unsigned int indexOfOutputElements = 0; indexOfOutputElements < mNeuralNetwork->GetNumOfOutputs(); indexOfOutputElements++){

			/*
			 * 		For every neuron in output layer, make it fire its sum of charges;
			 */

			double res = mNeuralNetwork->GetOutputLayer().at(indexOfOutputElements)->Fire();
			if(res > 0.0){
				netResponse.push_back(1);
			}
			else if( res == 0.0 ){
				netResponse.push_back(0);
			}
			else{
				netResponse.push_back(-1);
			}

		}

		//output netResponse vector
		std::cout << "netResponse vector: { ";
		for(unsigned int i = 0; i < netResponse.size(); i++){
			 std::cout <<  netResponse[i] << " ";
		}
		std::cout << " }\n";

		for(unsigned int indexOfOutputElements = 0; indexOfOutputElements < mNeuralNetwork->GetNumOfOutputs(); indexOfOutputElements++){
			if ( netResponse[indexOfOutputElements] != inTarget[indexOfOutputElements] ){




				for(unsigned int numOfLayers = 0; numOfLayers < mNeuralNetwork->mLayers.size() - 1; numOfLayers++){

					for(unsigned int NumOfNeurons = 0; NumOfNeurons < mNeuralNetwork->GetLayers().at(numOfLayers).size(); NumOfNeurons++){
						Neuron<T> * currentInputNeuron = (mNeuralNetwork->GetLayers().at(numOfLayers))[NumOfNeurons];

						for(int NumOfWeights = 0; NumOfWeights < currentInputNeuron->GetNumOfLinks(); NumOfWeights++){

							//std::cout << "x" << NumOfNeurons << ": " << inData[NumOfNeurons] << ", x" << NumOfNeurons << " * " << inTarget[NumOfWeights] << " + " << currentInputNeuron->at(NumOfWeights)->GetWeight() << " = " << currentInputNeuron->at(NumOfWeights)->GetWeight() + inTarget[NumOfWeights]*inData[NumOfNeurons] << std::endl;

							currentInputNeuron->at(NumOfWeights)->SetWeight(currentInputNeuron->at(NumOfWeights)->GetWeight() + inTarget[NumOfWeights]*inData[NumOfNeurons]);
						}

					}

					for(int NumOfWeights = 0; NumOfWeights < mNeuralNetwork->GetBiasLayer().at(numOfLayers)->GetNumOfLinks(); NumOfWeights++){

						//std::cout << "Bias"  << ": " << 1 << ", Bias "  << " * " << 1 << " + " << mNeuralNetwork->GetBiasLayer().at(numOfLayers)->at(NumOfWeights)->GetWeight() << " = " <<mNeuralNetwork->GetBiasLayer().at(numOfLayers)->at(NumOfWeights)->GetWeight() + inTarget[NumOfWeights]*1 << std::endl;

						mNeuralNetwork->GetBiasLayer().at(numOfLayers)->at(NumOfWeights)->SetWeight(mNeuralNetwork->GetBiasLayer().at(numOfLayers)->at(NumOfWeights)->GetWeight() + inTarget[NumOfWeights]*1);

					}

				}
				mNeuralNetwork->ResetCharges();
				return false;
			}
		}

		mNeuralNetwork->ResetCharges();
		return true;
	}

}


template <typename T>
void Backpropagation<T>::NguyenWidrowWeightsInitialization()
{
	/*
	 * 		Step 0. Initialize weights ( Set to small values )
	*/
	srand((unsigned)time(0));


	/*
	 *		For every layer, for every neuron and bias in that layer,  for every link in that neuron, set the weight
	 *		to random number from 0 to 1;
	 *
	*/

	double dNumOfInputs = mNeuralNetwork->mInputs;
	double dNumOfHiddens = mNeuralNetwork->mHidden;
	double degree = 1.0 / dNumOfInputs ;
	double dScaleFactor = 0.7*( pow( dNumOfHiddens , degree ) );

	for(unsigned int layerInd = 0; layerInd < mNeuralNetwork->size(); layerInd++){
		for(unsigned int neuronInd = 0; neuronInd < mNeuralNetwork->GetLayer(layerInd).size(); neuronInd++){
			Neuron<T> * currentNeuron = mNeuralNetwork->GetLayer(layerInd).at(neuronInd);
			for(int linkInd = 0; linkInd < currentNeuron->GetNumOfLinks(); linkInd++){
				NeuralLink<T> * currentNeuralLink = currentNeuron->at(linkInd);
				float pseudoRandWeight = -0.5 + (float)rand()/((float)RAND_MAX/(0.5 + 0.5));
				//float pseudoRandWeight = 0;
				currentNeuralLink->SetWeight(pseudoRandWeight);

				//std::cout << "layerInd: " << layerInd << ", neuronInd: " << neuronInd << ", linkInd: " << linkInd << ", Weight: " << currentNeuralLink->GetWeight() << std::endl;

			}
		}
	}


	for(unsigned int neuronHiddenInd = 0; neuronHiddenInd < mNeuralNetwork->GetLayer(1).size(); neuronHiddenInd++){
		//Neuron * currentHiddenNeuron = mNeuralNetwork->GetLayer(1).at(neuronHiddenInd);

		double dSquaredNorm = 0;

		for(unsigned int neuronInputInd = 0; neuronInputInd < mNeuralNetwork->GetLayer(0).size(); neuronInputInd++){
			Neuron<T> * currentInputNeuron = mNeuralNetwork->GetLayer(0).at(neuronInputInd);

			NeuralLink<T> * currentNeuralLink = currentInputNeuron->at(neuronHiddenInd);

			dSquaredNorm += pow(currentNeuralLink->GetWeight(),2.0);
		}

		double dNorm = sqrt(dSquaredNorm);

		for(unsigned int neuronInputInd = 0; neuronInputInd < mNeuralNetwork->GetLayer(0).size(); neuronInputInd++){
			Neuron<T> * currentInputNeuron = mNeuralNetwork->GetLayer(0).at(neuronInputInd);

			NeuralLink<T> * currentNeuralLink = currentInputNeuron->at(neuronHiddenInd);

			double dNewWeight = ( dScaleFactor * ( currentNeuralLink->GetWeight() ) ) / dNorm;
			currentNeuralLink->SetWeight(dNewWeight);
		}

	}

	for(unsigned int layerInd = 0; layerInd < mNeuralNetwork->size() - 1; layerInd++){

		Neuron<T> * Bias = mNeuralNetwork->GetBiasLayer().at(layerInd);
		for(int linkInd = 0; linkInd < Bias->GetNumOfLinks(); linkInd++){
			NeuralLink<T> * currentNeuralLink = Bias->at(linkInd);
			float pseudoRandWeight = -dScaleFactor + (float)rand()/((float)RAND_MAX/(dScaleFactor + dScaleFactor));
			//float pseudoRandWeight = 0;
			currentNeuralLink->SetWeight(pseudoRandWeight);
			//std::cout << "layerInd Bias: " << layerInd  << ", linkInd: " << linkInd << ", Weight: " << currentNeuralLink->GetWeight() << std::endl;
		}
	}
}

template <typename T>
void Backpropagation<T>::CommonInitialization()
{
	/*
	 * 		Step 0. Initialize weights ( Set to small values )
	*/

	srand((unsigned)time(0));


	/*
	 *		For every layer, for every neuron and bias in that layer,  for every link in that neuron, set the weight
	 *		to random number from 0 to 1;
	 *
	*/

	for(unsigned int layerInd = 0; layerInd < mNeuralNetwork->size(); layerInd++){
		for(unsigned int neuronInd = 0; neuronInd < mNeuralNetwork->GetLayer(layerInd).size(); neuronInd++){
			Neuron<T> * currentNeuron = mNeuralNetwork->GetLayer(layerInd).at(neuronInd);
			for(int linkInd = 0; linkInd < currentNeuron->GetNumOfLinks(); linkInd++){
				NeuralLink<T> * currentNeuralLink = currentNeuron->at(linkInd);
				float pseudoRandWeight = -0.5 + (float)rand()/((float)RAND_MAX/(0.5 + 0.5));
				//float pseudoRandWeight = 0;
				currentNeuralLink->SetWeight(pseudoRandWeight);

				//std::cout << "layerInd: " << layerInd << ", neuronInd: " << neuronInd << ", linkInd: " << linkInd << ", Weight: " << currentNeuralLink->GetWeight() << std::endl;

			}
		}
	}
	for(unsigned int layerInd = 0; layerInd < mNeuralNetwork->size() - 1; layerInd++){

		Neuron<T> * Bias = mNeuralNetwork->GetBiasLayer().at(layerInd);
		for(int linkInd = 0; linkInd < Bias->GetNumOfLinks(); linkInd++){
			NeuralLink<T> * currentNeuralLink = Bias->at(linkInd);
			float pseudoRandWeight = -0.5 + (float)rand()/((float)RAND_MAX/(0.5 + 0.5));
			//float pseudoRandWeight = 0;
			currentNeuralLink->SetWeight(pseudoRandWeight);

			//std::cout << "layerInd Bias: " << layerInd  << ", linkInd: " << linkInd << ", Weight: " << currentNeuralLink->GetWeight() << std::endl;

		}
	}
}

template <typename T>
void Backpropagation<T>::WeightsInitialization()
{
	this->NguyenWidrowWeightsInitialization();

}

template <typename T>
Backpropagation<T>::Backpropagation(NeuralNetwork<T> * inNeuralNetwork)
{
	mNeuralNetwork = inNeuralNetwork;
	//this->WeightsInitialization();


}


/*template <typename T>
Backpropagation<T>::Backpropagation(NeuralNetwork<T> * inNeuralNetwork)
{
	mNeuralNetwork = inNeuralNetwork;


	 * 		Step 0. Initialize weights ( Set to small values )

	srand((unsigned)time(0));



	 *		For every layer, for every neuron and bias in that layer,  for every link in that neuron, set the weight
	 *		to random number from 0 to 1;
	 *


	for(unsigned int layerInd = 0; layerInd < mNeuralNetwork->size(); layerInd++){
		for(unsigned int neuronInd = 0; neuronInd < mNeuralNetwork->GetLayer(layerInd).size(); neuronInd++){
			Neuron<T> * currentNeuron = mNeuralNetwork->GetLayer(layerInd).at(neuronInd);
			for(int linkInd = 0; linkInd < currentNeuron->GetNumOfLinks(); linkInd++){
				NeuralLink<T> * currentNeuralLink = currentNeuron->at(linkInd);
				float pseudoRandWeight = -0.5 + (float)rand()/((float)RAND_MAX/(0.5 + 0.5));
				//float pseudoRandWeight = 0;
				currentNeuralLink->SetWeight(pseudoRandWeight);

				//std::cout << "layerInd: " << layerInd << ", neuronInd: " << neuronInd << ", linkInd: " << linkInd << ", Weight: " << currentNeuralLink->GetWeight() << std::endl;

			}
		}
	}
	for(unsigned int layerInd = 0; layerInd < mNeuralNetwork->size() - 1; layerInd++){

		Neuron<T> * Bias = mNeuralNetwork->GetBiasLayer().at(layerInd);
		for(int linkInd = 0; linkInd < Bias->GetNumOfLinks(); linkInd++){
			NeuralLink<T> * currentNeuralLink = Bias->at(linkInd);
			float pseudoRandWeight = -0.5 + (float)rand()/((float)RAND_MAX/(0.5 + 0.5));
			//float pseudoRandWeight = 0;
			currentNeuralLink->SetWeight(pseudoRandWeight);

			//std::cout << "layerInd Bias: " << layerInd  << ", linkInd: " << linkInd << ", Weight: " << currentNeuralLink->GetWeight() << std::endl;

		}
	}
}*/


template <typename T>
double Backpropagation<T>::Train(const std::vector<T>& inData, const std::vector<T>& inTarget)
{
	/*
	 * 		Check incoming data
	*/

	double result = 0;
	if( inData.size() != mNeuralNetwork->mInputs || inTarget.size() != mNeuralNetwork->mOutputs ){
		std::cout << "Input data dimensions are wrong, expected: " << mNeuralNetwork->mInputs << " elements\n";

		return -1;
	}
	else{

		/*
		 * 		Step 3. Feedforward: Each input unit receives input signal and
		 * 		broadcast this signal to all units in the layer above (the hidden units)
		*/

		for(unsigned int indexOfData = 0; indexOfData < mNeuralNetwork->mInputs; indexOfData++){
			//std::cout << "input" << indexOfData << ": " << inData[indexOfData] << std::endl;
			mNeuralNetwork->GetInputLayer().at(indexOfData)->Input(inData[indexOfData]);
		}


		for(unsigned int numOfLayer = 0; numOfLayer < mNeuralNetwork->size() - 1; numOfLayer++){
			mNeuralNetwork->GetBiasLayer().at(numOfLayer)->Input(1.0);
			//std::cout << "BiasInput"   << std::endl;
			//std::cout << "Layer: " << numOfLayer << std::endl;
			for(unsigned int indexOfNeuronInLayer = 0; indexOfNeuronInLayer < mNeuralNetwork->GetLayer(numOfLayer).size(); indexOfNeuronInLayer++){
				//std::cout << "IndexOfNeuron: " << indexOfNeuronInLayer << std::endl;
				mNeuralNetwork->GetLayer(numOfLayer).at(indexOfNeuronInLayer)->Fire();
			}
			//std::cout << "Bias: " << numOfLayer << std::endl;
			mNeuralNetwork->GetBiasLayer().at(numOfLayer)->Fire();
			for(int i = 0; i < mNeuralNetwork->GetBiasLayer().at(numOfLayer)->GetNumOfLinks(); i++){
				mNeuralNetwork->GetBiasLayer().at(numOfLayer)->GetLinksToNeurons().at(i)->SetLastTranslatedSignal(1);
			}
		}

		/*
		 * 		Step 5. Each output unit applies its activation function to compute its output
		 * 		signal.
		*/


		std::vector<double> netResponseYk;
		for(unsigned int indexOfOutputElements = 0; indexOfOutputElements < mNeuralNetwork->mOutputs; indexOfOutputElements++){

			double Yk = mNeuralNetwork->GetOutputLayer().at(indexOfOutputElements)->Fire();
			netResponseYk.push_back(Yk);

		}

		/*
		 * 		Step 6. Backpropagation of error
		 *		Computing error information for each output unit.
		*/

		for(unsigned int indexOfData = 0; indexOfData < mNeuralNetwork->mOutputs; indexOfData++){
			result = mNeuralNetwork->GetOutputLayer().at(indexOfData)->PerformTrainingProcess(inTarget[indexOfData]);
			mNeuralNetwork->AddMSE(result);
		}


		/*
		 *		FIXME: Net should perform training process not only for last layer and layer before last, but also for any
		 *		layers except input one, so fix it DUDE!
		*/

		for(unsigned int iIndOfLayer = mNeuralNetwork->size() - 2; iIndOfLayer > 0 ; iIndOfLayer--){
			for(unsigned int indexOfNeuron = 0; indexOfNeuron < mNeuralNetwork->GetLayer(iIndOfLayer).size(); indexOfNeuron++){
				mNeuralNetwork->GetLayer(iIndOfLayer).at(indexOfNeuron)->PerformTrainingProcess(0);
			}
		}
		mNeuralNetwork->UpdateWeights();


		mNeuralNetwork->ResetCharges();
		return result;
	}
}


template class Backpropagation<double>;
template class Backpropagation<float>;
template class Backpropagation<int>;
