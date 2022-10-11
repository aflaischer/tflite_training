#include "ModelTfLite.hpp"

#include <cassert>
#include <iostream>
#include <cstring>

#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

constexpr unsigned int BATCH_SIZE=100;
constexpr unsigned int NB_EPOCHES=100;

static int GetTensorSize(const TfLiteTensor* tensor)
{
    int size = 1;

    if (tensor->dims->size == 0) {
        return 0;
    }

    for (int i = 0; i < tensor->dims->size; i++) {
        size *= tensor->dims->data[i];
    }
    return size;
}

static void PrintTensorInfo(const TfLiteTensor* tensor)
{
    std::cout << "Tensor " << tensor->name << " size: " << GetTensorSize(tensor) << '\n';
    std::cout << "Tensor " << tensor->name << " bytes: " << tensor->bytes << '\n';
    std::cout << "Tensor " << tensor->name << " type: " << TfLiteTypeGetName(tensor->type) << '\n';
}

ModelTfLite::ModelTfLite(const std::string& path)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    assert(model_ != nullptr);

    tflite::InterpreterBuilder builder(*model_, resolver_);
    builder(&interpreter_);
    assert(interpreter_ != nullptr);
}

ModelTfLite::~ModelTfLite()
{
}

bool ModelTfLite::Predict(std::vector<float> inputs)
{
    TfLiteStatus status;
    tflite::SignatureRunner* infer_runner = interpreter_->GetSignatureRunner("infer");
    assert(infer_runner != nullptr);

    infer_runner->ResizeInputTensor("x", {static_cast<int>(inputs.size()), 1});
    status = infer_runner->AllocateTensors();
    if(status != kTfLiteOk) {
        std::cout << "Failed to allocate inference signature tensors \n";
        return false;
    }

    TfLiteTensor* input_tensor = infer_runner->input_tensor("x");
    assert(input_tensor != nullptr);
    PrintTensorInfo(input_tensor);

    auto tensor_inputs = input_tensor->data.f;
    for (int i = 0; i < GetTensorSize(input_tensor); i++)
    {
        tensor_inputs[i] = inputs[i];
    }

    status = infer_runner->Invoke();
    if(status != kTfLiteOk) {
        std::cout << "Failed to run training signature \n";
        return false;
    }

    const TfLiteTensor* output_tensor = infer_runner->output_tensor("output");
    assert(output_tensor != nullptr);
    PrintTensorInfo(output_tensor);

    auto tensor_outputs = output_tensor->data.f;
    std::cout << "Output are: ";
    for (int i = 0; i < GetTensorSize(output_tensor); i++)
    {
        std::cout << tensor_outputs[i] << ',';
    }
    std::cout << "\n";

    return true;
}

bool ModelTfLite::Train(std::vector<float> features, std::vector<float> targets)
{
    TfLiteStatus status;

    assert(features.size() == targets.size());

    tflite::SignatureRunner* train_runner = interpreter_->GetSignatureRunner("train");
    assert(train_runner != nullptr);

    train_runner->ResizeInputTensor("x", {BATCH_SIZE, 1});
    train_runner->ResizeInputTensor("y", {BATCH_SIZE, 1});
    status = train_runner->AllocateTensors();
    if(status != kTfLiteOk) {
        std::cout << "Failed to allocate training signature tensors \n";
        return false;
    }

    TfLiteTensor* input_tensor_features = train_runner->input_tensor("x");
    TfLiteTensor* input_tensor_targets = train_runner->input_tensor("y");
    const TfLiteTensor* output_tensor = train_runner->output_tensor("loss");

    assert(input_tensor_features != nullptr);
    assert(input_tensor_targets != nullptr);
    assert(output_tensor != nullptr);

    PrintTensorInfo(input_tensor_features);
    PrintTensorInfo(input_tensor_targets);
    PrintTensorInfo(output_tensor);

    auto input_features = input_tensor_features->data.f;
    auto input_targets = input_tensor_targets->data.f;

    for (int i = 0; i < NB_EPOCHES; i++)
    {
        for (int b = 0; b < features.size(); b += BATCH_SIZE)
        {
            for (int i = 0; i < GetTensorSize(input_tensor_features); i++)
            {
                input_features[i] = features[i + b];
                input_targets[i] = targets[i + b];
            }

            status = train_runner->Invoke();
            if(status != kTfLiteOk) {
                std::cout << "Failed to run training signature \n";
                return false;
            }
        }

        // Get Loss from last batch
        float* output = output_tensor->data.f;
        std::cout << "epoch " << i << " Loss is: " << *output << '\n';
    }

    return true;
}

float ModelTfLite::GetAccuracy(std::vector<float> features, std::vector<float> targets)
{
    TfLiteStatus status;

    assert(features.size() == targets.size());

    tflite::SignatureRunner* train_runner = interpreter_->GetSignatureRunner("accuracy");
    assert(train_runner != nullptr);

    train_runner->ResizeInputTensor("x", {BATCH_SIZE, 1});
    train_runner->ResizeInputTensor("y", {BATCH_SIZE, 1});
    status = train_runner->AllocateTensors();
    if(status != kTfLiteOk) {
        std::cout << "Failed to allocate accuracy signature tensors \n";
        return -1;
    }

    TfLiteTensor* input_tensor_features = train_runner->input_tensor("x");
    TfLiteTensor* input_tensor_targets = train_runner->input_tensor("y");
    const TfLiteTensor* mae_tensor = train_runner->output_tensor("mean_absolute_error");
    const TfLiteTensor* mae_percent_tensor = train_runner->output_tensor("mean_absolute_percentage_error");

    assert(input_tensor_features != nullptr);
    assert(input_tensor_targets != nullptr);
    assert(mae_tensor != nullptr);
    assert(mae_percent_tensor != nullptr);

    PrintTensorInfo(input_tensor_features);
    PrintTensorInfo(input_tensor_targets);
    PrintTensorInfo(mae_tensor);
    PrintTensorInfo(mae_percent_tensor);

    auto input_features = input_tensor_features->data.f;
    auto input_targets = input_tensor_targets->data.f;

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        input_features[i] = features[i];
        input_targets[i] = targets[i];
    }

    status = train_runner->Invoke();
    if(status != kTfLiteOk) {
        std::cout << "Failed to run accuracy signature \n";
        return -1;
    }

    float* output = mae_tensor->data.f;
    std::cout << "MAE Accuracy is: " << *output << '\n';
    std::cout << "MAE \% Accuracy is: " << *mae_percent_tensor->data.f << '\n';

    return *output;
}

bool ModelTfLite::Save(const std::string& checkpointPath)
{
    return SaveOrRestore(checkpointPath, true);
}

bool ModelTfLite::Restore(const std::string& checkpointPath)
{
    return SaveOrRestore(checkpointPath, false);
}

bool ModelTfLite::SaveOrRestore(const std::string& checkpointPath, bool save)
{
    TfLiteStatus status;

    std::string runner_name = (save) ? "save" : "restore";

    tflite::SignatureRunner* runner = interpreter_->GetSignatureRunner(runner_name.c_str());
    assert(runner != nullptr);

    status = runner->AllocateTensors();
    if(status != kTfLiteOk) {
        std::cout << "Failed to allocate inference signature tensors \n";
        return false;
    }

    TfLiteTensor* input_tensor = runner->input_tensor("checkpoint_path");
    assert(input_tensor != nullptr);

    PrintTensorInfo(input_tensor);

    tflite::DynamicBuffer buf;
    buf.AddString(checkpointPath.c_str(), checkpointPath.size());
    buf.WriteToTensor(input_tensor, /*new_shape=*/TfLiteIntArrayCreate(0));

    status = runner->Invoke();
    if(status != kTfLiteOk) {
        std::cout << "Failed to run training signature \n";
        return false;
    }

    return true;
}

void ModelTfLite::PrintInterpreterState()
{
    tflite::PrintInterpreterState(interpreter_.get());
}

size_t ModelTfLite::GetNumSignatures()
{
    return interpreter_->subgraphs_size();
}

void ModelTfLite::PrintSignatures()
{
    int size = GetNumSignatures();

    std::cout << "Signature names: ";
    for (int i = 0; i < size; i++)
    {
        std::cout << interpreter_->subgraph(i)->GetName() << ", ";
    }
    std::cout << "\n";
}