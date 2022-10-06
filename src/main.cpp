
#include <cstdio>
#include <stdlib.h>

#include <iostream>
#include <fstream>

#include "ModelTfLite.hpp"

static bool isFile(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "testmodel <tflite model>\n");
        return 1;
    }
    const char* filename = argv[1];

    ModelTfLite m(filename);

    std::cout << "Number of signatures: " << m.GetNumSignatures() << '\n';
    m.PrintSignatures();

    //Create features/targets buffers
    std::vector<float> train_inputs, train_targets;
    for (int i = 0; i < 100; i++) {
        train_inputs.push_back(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
        train_targets.push_back(2 * train_inputs.back() + 5);

        std::cout << "x: " << train_inputs[i] << " y: " << train_targets[i] << '\n';
    }

    std::string ckpt = "checkpoint.ckpt";

    if (isFile(ckpt)) {
        std::cout << "Checkpoint file found restore & infer \n";

        m.Restore("checkpoint.ckpt");
        m.Predict(0.5);
        m.GetAccuracy(train_inputs, train_targets);
        return 0;
    }

    m.Predict(0.5);
    m.Train(train_inputs, train_targets);
    m.Predict(0.5);
    m.GetAccuracy(train_inputs, train_targets);

    m.Save("checkpoint.ckpt");

    //m.PrintInterpreterState();

    return 0;
}