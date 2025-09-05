#!/usr/bin/env python3

import os
import tensorflow as tf
import keras
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

INPUT_NB_ELEMENTS = 1
OUTPUT_NB_ELEMENTS = 1

# Custom metric for accuracy
def mean_absolute_percentage_accuracy(y_true, y_pred):
    return 100 - keras.losses.mean_absolute_percentage_error(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
class MeanAbsolutePercentageAccuracy(keras.metrics.MeanMetricWrapper):
    def __init__(self, name="mean_absolute_percentage_accuracy", dtype=None):
        super().__init__(mean_absolute_percentage_accuracy, name, dtype=dtype)

class LinearModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential([
            keras.layers.InputLayer(shape=(INPUT_NB_ELEMENTS,)),
            keras.layers.Dense(10, activation='relu', kernel_initializer='glorot_normal', name='dense_1'),
            keras.layers.Dense(OUTPUT_NB_ELEMENTS, name='output')
        ])

        self.model.compile(
            optimizer='sgd',
            loss=keras.losses.MeanSquaredError(),
            metrics=[
                MeanAbsolutePercentageAccuracy(name='accuracy')
                ]
            )

        self.signatures = [
            'train',
            'infer',
            'save',
            'restore',
            'accuracy'
        ]

    @tf.function(input_signature=[tf.TensorSpec([None, INPUT_NB_ELEMENTS], tf.float32, name="inputs")])
    def __call__(self, x):
        return self.model(x)

    @tf.function(input_signature=[
        tf.TensorSpec([None, INPUT_NB_ELEMENTS], tf.float32),
        tf.TensorSpec([None, OUTPUT_NB_ELEMENTS], tf.float32),
    ])
    def train(self, features, targets):
        with tf.GradientTape() as tape:
            prediction = self.model(features)
            loss = self.model.loss(targets, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        result = {"loss": loss}
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([None, INPUT_NB_ELEMENTS], tf.float32),
    ])
    def infer(self, features):
        output = self.model(features)
        return {
            "output": output,
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.value.name for weight in self.model.weights]
        tensors_to_save = [weight.value.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.value.name, dt=var.value.dtype,
                name='restore')
            var.value.assign(restored)
            restored_tensors[var.value.name] = restored
        return restored_tensors

    @tf.function(input_signature=[
        tf.TensorSpec([None, INPUT_NB_ELEMENTS], tf.float32),
        tf.TensorSpec([None, OUTPUT_NB_ELEMENTS], tf.float32),
    ])
    def accuracy(self, features, targets):
        prediction = self.model(features)
        return_metrics = {}
        for metric in self.model.metrics:
            if metric.name == "compile_metrics":
                metric.update_state(targets, prediction)
                result = metric.result()
                if isinstance(result, dict):
                    return_metrics.update(result)
                else:
                    return_metrics[metric.name] = result
        return return_metrics

    def save_model(self, model_path):
        # Need to have a first call to save it correctly in archive.
        for signature_name in self.signatures:
            signature = getattr(self, signature_name)
            signature.get_concrete_function()

        export_archive = keras.export.ExportArchive()
        export_archive.track(self.model)

        for signature_name in self.signatures:
            signature = getattr(self, signature_name)
            export_archive.add_endpoint(
                name=signature_name,
                fn=signature,
            )

        export_archive.write_out(model_path)

    def convert_to_tflite(self, model_path, tflite_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(
            model_path,
            signature_keys=self.signatures)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]

        tflite_model = converter.convert()
        converter.experimental_enable_resource_variables = True
        open(tflite_path, "wb").write(tflite_model)

def main():
    model = LinearModel()

    # Save Model
    model_path = "./linear_model"
    model.save_model(model_path)

    # Training in python
    NUM_EPOCHS = 100

    #x = np.array([[i] for i in range(100)])
    x = np.random.rand(10000, 1)
    y = x*2 + 5

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    #print("x: ", x)
    #print("y: ", y)

    print("predict before training : ", model.infer([[0.5]]))
    for _ in range(NUM_EPOCHS):
        result = model.train(x, y)
        print("loss: ", result['loss'])
    print("predict after training: ", model.infer([[0.5]]))

    acc = model.accuracy(x, y)
    print("Accuracy is: ", acc)

    # Convert to tflite
    tflite_path = model_path + ".tflite"
    model.convert_to_tflite(model_path, tflite_path)

    tf.lite.experimental.Analyzer.analyze(model_path=tflite_path)

if __name__ == "__main__" :
    main()