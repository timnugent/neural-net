#ifndef TRAINER
#define TRAINER

#include <vector>
#include "neuralnet.h"

struct example{	
	example(double* t, double* f) : target(t), features(f) {}
	~example(){
		delete [] target;
		delete [] features;
	}
	double* target;
	double* features;
};

class trainer{

public:

	explicit trainer(neuralnet*);
	~trainer();
	int train();
	int load_training_data(const char*);
	void set_learning_rate(double x){lr = x;};
	void set_batch(bool x){batch = x;};
	void set_momentum(double x){momentum = x;};
	void set_max_epochs(int x){max_epochs = x;};
	double get_learning_rate(){return lr;}
	bool get_gradient_learning(){return batch;};
	double get_momentum(){return momentum;};

private:

	void backpropagate(double*);
	double get_output_gradient(double, double);
	double get_hidden_gradient(int);
	double clampoutput(double);
	void update_weights();
	
	neuralnet* nn;
	int epochs, max_epochs;
	double lr;
	bool batch;
	double momentum;
	double **ih_delta, **ho_delta;
	double *hidden_gradients, *output_gradients;
	std::vector<example*> training_data;

};

#endif