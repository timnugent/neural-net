all:
	g++ -O3 --std=c++11 -Wall -Wextra src/main.cpp src/neuralnet.cpp src/trainer.cpp -o bin/nn
test:
	bin/nn
clean:
	rm bin/nn
