
#include <cstdio>
#include <stdlib.h>

#include <iostream>

#include "ModelTfLite.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "testmodel <tflite model>\n");
        return 1;
    }
    const char* filename = argv[1];

    ModelTfLite m(filename);

    //Create features/targets buffers
    std::vector<float> train_inputs, train_targets;
    for (int i = 0; i < 100; i++) {
        train_inputs.push_back(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
        train_targets.push_back(2 * train_inputs.back() + 5);

        std::cout << "x: " << train_inputs[i] << " y: " << train_targets[i] << '\n';
    }

    m.Predict(0.5);

    m.Train(train_inputs, train_targets);

    m.Predict(0.5);

    //m.PrintInterpreterState();

    return 0;
}