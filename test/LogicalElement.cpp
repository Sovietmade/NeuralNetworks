/*
 * main.cpp
 *
 *  Created on: Sep 27, 2013
 *      Author: cheryuri
 */



#include "NeuralNetworks/NeuralNetwork.h"
#include "NeuralNetworks/trainAlgorithm.h"
#include "stdlib.h"
#include <iostream>





int main()
{

  std::vector<std::vector<double> > DataToFeedNN;
  std::vector<double>  Data1;
  Data1.push_back(1.0);
  Data1.push_back(1.0);
  
  DataToFeedNN.push_back(Data1);
  
  std::vector<double>  Data2;
  Data2.push_back(1.0);
  Data2.push_back(-1.0);
  
  DataToFeedNN.push_back(Data2);
  
  std::vector<double>  Data3;
  Data3.push_back(-1.0);
  Data3.push_back(1.0);
  
  DataToFeedNN.push_back(Data3);
  
  std::vector<double>  Data4;
  Data4.push_back(-1.0);
  Data4.push_back(-1.0);
  
  DataToFeedNN.push_back(Data4);
  
  std::vector<double>  Data5;
  Data5.push_back(0.5);
  Data5.push_back(-1.0);
  //DataToFeedNN.push_back(Data5);
  
  std::vector<std::vector<double> > trainingSample;
  std::vector<double> ts1;
  ts1.push_back(1);
  ts1.push_back(-1);
  std::vector<double> ts2;
  ts2.push_back(-1);
  ts2.push_back(-1);
  std::vector<double> ts3;
  ts3.push_back(-1);
  ts3.push_back(-1);
  std::vector<double> ts4;
  ts4.push_back(-1);
  ts4.push_back(-1);
  std::vector<double> ts5;
  ts5.push_back(0);
  ts5.push_back(1);
  trainingSample.push_back(ts1);
  trainingSample.push_back(ts2);
  trainingSample.push_back(ts3);
  trainingSample.push_back(ts4);
  //trainingSample.push_back(ts5);
  
  NeuralNetwork<double> * NN = new NeuralNetwork<double>(2,2,1,4);
  NN->SetMinMSE(0.01);
  NN->Train(DataToFeedNN,trainingSample);
  
  
  std::cout << std::endl;
  std::cout << "Input data: { 1, 1 }\n";
  NN->GetNetResponse(DataToFeedNN[0]);
  
  std::cout << std::endl;
  std::cout << "Input data: { 1, 0 }\n";
  NN->GetNetResponse(DataToFeedNN[1]);
  
  std::cout << std::endl;
  std::cout << "Input data: { 0, 1 }\n";
  NN->GetNetResponse(DataToFeedNN[2]);
  
  std::cout << std::endl;
  std::cout << "Input data: { 0, 0 }\n";
  NN->GetNetResponse(DataToFeedNN[3]);
  
  
  std::cout << std::endl;
  std::cout << "Input data: { test }\n";
  NN->GetNetResponse(Data5);
  
  delete NN;
  
  return 0;
}



