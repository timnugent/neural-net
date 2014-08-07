#ifndef NNET
#define NNET

#include <vector>
#include <cmath>

class neuralnet{

public:

	neuralnet(int,int,int);
	~neuralnet();
	void feedforward(double*);
	void initializeweights(double range = 0.5);	
	int writeweights(const char*);
	int readweights(const char*);
	int test(const char*);
	unsigned int get_inputs(){
		return inputs;
	}
	unsigned int get_hidden(){
		return hidden;
	}
	unsigned int get_outputs(){
		return outputs;
	}
	double get_input_value(int i){
		return input_neurons[i];
	}
	double get_hidden_value(int i){
		return hidden_neurons[i];
	}
	double get_output_value(int i){
		return output_neurons[i];
	}
	double get_ho_weight(int i, int j){
		return ho_weights[i][j];
	}
	void update_ih_weight(int i, int j, double w){
		if(!isnan(w)) ih_weights[i][j] += w;
	}
	void update_ho_weight(int i, int j, double w){
		if(!isnan(w)) ho_weights[i][j] += w;
	}	

private:
	
	double activation(double);
	double clampoutput(double);
	unsigned int inputs, hidden, outputs;
	double *input_neurons, *hidden_neurons, *output_neurons;
	double **ih_weights, **ho_weights;

};	

#endif
