#include <iostream>
#include "neuralnet.h"
#include "trainer.h"

using namespace std;

int main(){

	// Initialize network
	neuralnet* N = new neuralnet(784,300,10);

	// Setup trainer
	trainer* T = new trainer(N);
	T->set_learning_rate(0.001);
	T->set_batch(false);
	T->set_momentum(0.9);
	T->set_max_epochs(5000);
	T->load_training_data("minst/mnist_train_nn_10K.csv");

	// Train using some data
	if(T->train()){
		cout << "Training complete." << endl;
	}else{
		cout << "Training failed." << endl;
	}
	// Clean up trainer
	delete T;


	// Test some data
	if(N->test("minst/mnist_test_nn_5K.csv")){
		cout << "Testing complete." << endl;
	}else{
		cout << "Testing failed." << endl;
	}

	// Write weights to file
	if(N->writeweights("mnist_weights.out")){
		cout << "Wrote mnist_weights.out" << endl;
	}else{
		cout << "Failed to write mnist_weights.out" << endl;
	}

	// Clean up network
	delete N;

	return(0);
}