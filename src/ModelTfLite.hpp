#pragma once

#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

class ModelTfLite
{
public:
    ModelTfLite(const std::string& path);
    ~ModelTfLite();

    void PrintInterpreterState();

    bool Predict(std::vector<float> inputs);
    bool Train(std::vector<float> features, std::vector<float> targets);
    float GetAccuracy(std::vector<float> features, std::vector<float> targets);
    bool Save(const std::string& checkpointPath);
    bool Restore(const std::string& checkpointPath);
    size_t GetNumSignatures();
    void PrintSignatures();

    bool Invoke();
    void PrintInfoTensor();

private:
    bool SaveOrRestore(const std::string& checkpointPath, bool save);

    std::unique_ptr<tflite::FlatBufferModel> model_;
    tflite::ops::builtin::BuiltinOpResolver resolver_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
};