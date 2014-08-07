#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <math.h>
#include "trainer.h"
#include "neuralnet.h"

using namespace std;

trainer::trainer(neuralnet* n) : nn(n), epochs(0), max_epochs(5000), lr(0.001), batch(false), momentum(0.9) {

	ih_delta = new double*[nn->get_inputs()+1];
	for (unsigned int i = 0; i <= nn->get_inputs(); i++){
		ih_delta[i] = new double[nn->get_hidden()];
	}

	ho_delta = new double*[nn->get_hidden()+1];
	for (unsigned int i = 0; i <= nn->get_hidden(); i++){
		ho_delta[i] = new double[nn->get_outputs()];			
	}

	hidden_gradients = new double[nn->get_hidden()+1];
	output_gradients = new double[nn->get_outputs()+1];

}

trainer::~trainer(){

	for(unsigned int i = 0; i <= nn->get_inputs(); i++){
		delete [] ih_delta[i];
	}
	delete [] ih_delta;

	for(unsigned int i = 0; i <= nn->get_hidden(); i++){
		delete [] ho_delta[i];			
	}
	delete [] ho_delta;

	delete [] hidden_gradients;
	delete [] output_gradients;

	for(unsigned int i = 0; i < training_data.size(); i++){
		delete training_data[i];
	}
}

int trainer::load_training_data(const char* file){

	ifstream infile;
	infile.open(file, ios::in);

	// Read training data in CSV format from file
	if(infile.is_open()){
		string line,token;
		while(getline(infile, line)){

			stringstream tmp(line);
			unsigned int pos = 0;

			double* targets = new double[nn->get_outputs()]();
			double* features = new double[nn->get_inputs()]();

			for(unsigned int i = 0; i < nn->get_outputs(); i++){
				targets[i]=0;
			}
			for(unsigned int i = 0; i < nn->get_inputs(); i++){
				features[i]=0;
			}

			while(getline(tmp, token, ',')){
				if(pos < nn->get_inputs()){
					features[pos] = stod(token);	
				}else{
					targets[pos-nn->get_inputs()] = stod(token);	
				}
				pos++;
			}
			training_data.push_back(new example(targets,features));	

			/*
			for(auto i : t){
				cout << i << ",";
			}
			cout << endl;
			for(auto i : features){
				cout << i << ",";
			}
			cout << endl;
			*/

		}
		return(1);
	}else{
		return(0);
	}

}

double trainer::clampoutput(double x){

	if(x < 0.1){
		return(0.0);
	}else if(x > 0.9){
		return(1.0);
	}
	return(-1.0);

}

int trainer::train(){

	if(!training_data.size()){
		return(0);
	}

	while(epochs < max_epochs){

		if(!batch) random_shuffle (training_data.begin(), training_data.end());

		double mse = 0.0;
		int correct_count = 0;
		for(unsigned int i = 0; i < training_data.size(); i++){
			
			nn->feedforward(training_data[i]->features);
			backpropagate(training_data[i]->target);	
			
			// Calculate MSE and accuracy			
			bool correct = true;
			for(unsigned int k = 0; k < nn->get_outputs(); k++){	
				mse += pow((nn->get_output_value(k) - training_data[i]->target[k]), 2);
				if(clampoutput(nn->get_output_value(k)) != training_data[i]->target[k]){
					correct = false;
				}
			}
			//cout << endl;

			if(correct){
				correct_count++;
			}
		}
		if(batch){
			update_weights();
		}
		if(++epochs && !(epochs % 100)){
			mse /= nn->get_outputs() * training_data.size();
			double accuracy = 100*(double)correct_count/training_data.size();
			printf("%i epochs : MSE = %2.4f Accuracy = %2.3f %%\n",epochs,mse,accuracy);
		}
	}

	return(1);

}


double trainer::get_output_gradient(double target, double output){
	
	// Gradient for sigmoidal activation function
	// http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
	return output * (1 - output) * (target - output);
}

double trainer::get_hidden_gradient(int j){
		
	double weighted_sum = 0.0;
	for(unsigned int k = 0; k < nn->get_outputs(); k++){
		weighted_sum += nn->get_ho_weight(j,k) * output_gradients[k];
	}
	// Gradient for sigmoidal activation function	
	// http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
	return nn->get_hidden_value(j) * (1 - nn->get_hidden_value(j)) * weighted_sum;
}


void trainer::backpropagate(double* target){

	// Calculate deltas between hidden and output layers
	for (unsigned int k = 0; k < nn->get_outputs(); k++){
		
		// Calculate gradient for output nodes
		output_gradients[k] = get_output_gradient(target[k], nn->get_output_value(k));

		// Calculate weight changes for hidden layer -> output neruons
		for (unsigned int j = 0; j <= nn->get_outputs(); j++){			
			if (!batch){
				ho_delta[j][k] = lr * nn->get_hidden_value(j) * output_gradients[k] + (momentum * ho_delta[j][k]);
			}else{
				ho_delta[j][k] += lr * nn->get_hidden_value(j) * output_gradients[k];
			}
		}
	}

	// Calculate deltas between input and hidden layers
	for (unsigned int j = 0; j < nn->get_hidden(); j++){

		//get error gradient for every hidden node
		hidden_gradients[j] = get_hidden_gradient(j);

		// Calculate weight changes for input layer -> hidden neruons
		for (unsigned int i = 0; i <= nn->get_inputs(); i++){
			//calculate change in weight 
			if(!batch){
				ih_delta[i][j] = lr * nn->get_input_value(i) * hidden_gradients[j] + (momentum * ih_delta[i][j]);
			}else{
				ih_delta[i][j] += lr * nn->get_input_value(i) * hidden_gradients[j]; 
			}
		}
	}
	
	//if using stochastic learning update the weights immediately
	if (!batch){
		update_weights();
	}
}	

void trainer::update_weights(){

	//cout << "updating weights" << endl;
	// Update input -> hidden weights
	for (unsigned int i = 0; i <= nn->get_inputs(); i++){
		for (unsigned int j = 0; j < nn->get_hidden(); j++){
			nn->update_ih_weight(i,j,ih_delta[i][j]);	
			if (batch) ih_delta[i][j] = 0.0;
		}
	}
	// Update hidden -> output weights
	for (unsigned int j = 0; j <= nn->get_hidden(); j++){
		for (unsigned int k = 0; k < nn->get_outputs(); k++){	
			nn->update_ho_weight(j,k,ho_delta[j][k]);	
			if (batch) ho_delta[j][k] = 0.0;
		}
	}

}
