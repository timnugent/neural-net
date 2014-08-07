#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <sstream>
#include "neuralnet.h"

using namespace std;

neuralnet::neuralnet(int n_i, int n_h, int n_o) : inputs(n_i), hidden(n_h), outputs(n_o){

	// Allocate storage for neurons inc. bias
	input_neurons = new double[inputs+1];
	hidden_neurons = new double[hidden+1];
	output_neurons = new double[outputs];

	// Allocate storage for weights inc. bias
	ih_weights = new double*[inputs+1];
	for (unsigned int i = 0; i <= inputs; i++){
		ih_weights[i] = new double[hidden];
	}
	ho_weights = new double*[hidden+1];
	for (unsigned int i = 0; i <= hidden; i++){
		ho_weights[i] = new double[outputs];
	}
	initializeweights();	

}

neuralnet::~neuralnet(){

	delete [] input_neurons;
	delete [] hidden_neurons;
	delete [] output_neurons;

	for (unsigned int i =0; i <= inputs; i++){
		delete [] ih_weights[i];
	}
	delete [] ih_weights;

	for (unsigned int i =0; i <= hidden; i++){
		delete [] ho_weights[i];
	}
	delete [] ho_weights;

}

void neuralnet::initializeweights(double range){

	// Setup weights
	// http://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
	// http://stackoverflow.com/questions/20027598/why-should-weights-of-neural-networks-be-initialized-to-random-numbers
	// http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> rand_weight(-range,range);

	for(unsigned int i = 0; i <= inputs; i++){		
		for(unsigned int j = 0; j < hidden; j++){
			ih_weights[i][j] = rand_weight(gen);	
			//printf("ih: %i --> %i = %1.3f\n",i,j,ih_weights[i][j]);		
		}
	}
	for(unsigned int i = 0; i <= hidden; i++){		
		for(unsigned int j = 0; j < outputs; j++){
			ho_weights[i][j] = rand_weight(gen);
			//printf("ho: %i --> %i = %1.3f\n",i,j,ho_weights[i][j]);					
		}
	}

}

void neuralnet::feedforward(double* features){

	// Setup input neurons
	for(unsigned int i = 0; i < inputs; i++){
		input_neurons[i] = features[i];
	}
	// Calculate hidden neuron values
	for(unsigned int j = 0; j < hidden; j++){
		hidden_neurons[j] = 0.0;			
		// Include bias	
		for(unsigned int i = 0; i <= inputs; i++){
			hidden_neurons[j] += input_neurons[i] * ih_weights[i][j];
		}
		// Activate
		hidden_neurons[j] = activation(hidden_neurons[j]);	
	}	
	for(unsigned int k = 0; k < outputs; k++){
		output_neurons[k] = 0.0;				
		// Include bias
		for(unsigned int j = 0; j <= hidden; j++){
			output_neurons[k] += hidden_neurons[j] * ho_weights[j][k];
		}		
		// Activate
		output_neurons[k] = activation(output_neurons[k]);	
	}
}

double neuralnet::activation(double x){
	return 1.0/(1.0+exp(-x));
}	

int neuralnet::writeweights(const char* filename){

	ofstream outfile;
	outfile.open(filename, ios::out);

	// Write weights to file
	if(outfile.is_open()){
		outfile.precision(6);		
		for (unsigned int i = 0; i <= inputs; i++){
			for(unsigned int j = 0; j < hidden; j++){
				outfile << ih_weights[i][j] << endl;				
			}
		}		
		for (unsigned int i =0; i <= hidden; i++){		
			for (unsigned int j = 0; j < outputs; j++){
				outfile << ho_weights[i][j] << endl;					
			}
		}
		outfile.close();		
		return(1);
	}else{
		return(0);
	}
}

int neuralnet::readweights(const char* filename){

	ifstream infile;
	infile.open(filename, ios::in);

	// Read weights from file
	if(infile.is_open()){
		vector<double> weights;
		string line;
		while(getline(infile, line)){
			double w = stod(line);
			weights.push_back(w);
		}
		infile.close();

		if(weights.size() != ((inputs+1)*hidden + (hidden+1)*outputs) ){
			cout << "Weight count mismatch!" << endl;
			return(0);
		}

		unsigned int pos = 0;
		for(unsigned int i = 0; i <= inputs; i++){
			for (unsigned int j = 0; j < hidden; j++){
				ih_weights[i][j] = weights[pos++];					
			}
		}		
		for(unsigned int i = 0; i <= hidden; i++){		
			for (unsigned int j = 0; j < outputs; j++){
				ho_weights[i][j] = weights[pos++];						
			}
		}
		return(1);
	}else{
		return(0);
	}
	
}	

double neuralnet::clampoutput(double x){

	if(x < 0.1){
		return(0.0);
	}else if(x > 0.9){
		return(1.0);
	}
	return(-1.0);

}

int neuralnet::test(const char* file){

	ifstream infile;
	infile.open(file, ios::in);

	double mse = 0.0;
	int correct_count = 0, total = 0;

	// Read training data in CSV format from file
	if(infile.is_open()){
		string line,token;
		while(getline(infile, line)){
			stringstream tmp(line);
			unsigned int pos = 0;
			double* t = new double[outputs]();
			double* features = new double[inputs]();
			while(getline(tmp, token, ',')){
				if(pos < inputs){
					features[pos] = stod(token);	
				}else{
					t[pos-inputs] = stod(token);	
				}
				pos++;
			}

			feedforward(features);
			bool correct = true;
			for(unsigned int k = 0; k < outputs; k++){	
				mse += pow((output_neurons[k] - t[k]), 2);
				if(clampoutput(output_neurons[k]) != t[k]){
					correct = false;
				}
			}
			delete [] t;
			delete [] features;

			if(correct){
				correct_count++;
			}
			total++;
		}

		mse /= outputs * total;
		double accuracy = 100*(double)correct_count/total;
		printf("MSE = %2.4f Accuracy = %2.3f %%\n",mse,accuracy);

		return(1);
	}else{
		return(0);
	}

}