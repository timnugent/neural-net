#include <iostream>
#include "neuralnet.h"
#include "trainer.h"

using namespace std;

int main(){

	// Initialize network
	neuralnet* N = new neuralnet(4,9,3);

	// Setup trainer
	trainer* T = new trainer(N);
	T->set_learning_rate(0.001);
	T->set_batch(false);
	T->set_momentum(0.9);
	T->set_max_epochs(100000);
	T->load_training_data("data/iris.scale.train.csv");

	// Train using some data
	cout << "Training:" << endl;
	if(T->train()){
		cout << "Training complete." << endl;
	}else{
		cout << "Training failed." << endl;
	}
	// Clean up trainer
	delete T;

	// Test some data
	cout << "Testing:" << endl;
	if(N->test("data/iris.scale.test.csv")){
		cout << "Testing complete." << endl;
	}else{
		cout << "Testing failed." << endl;
	}

	// Write weights to file
	if(N->writeweights("data/iris_weights.out")){
		cout << "Wrote data/iris_weights.out" << endl;
	}else{
		cout << "Failed to write data/iris_weights.out" << endl;
	}

	// Clean up network
	delete N;


	/*
	neuralnet* NN = new neuralnet(4,9,3);
	if(NN->readweights("data/iris_weights.out")){
		cout << "Read data/iris_weights.out" << endl;
	}else{
		cout << "Failed to read data/iris_weights.out" << endl;
	}
	// Test some data
	if(NN->test("data/iris.scale.test.csv")){
		cout << "Testing complete." << endl;
	}else{
		cout << "Testing failed." << endl;
	}
	// Clean up network
	delete NN;
	*/

	return(0);
}