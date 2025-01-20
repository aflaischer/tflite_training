#!/usr/bin/env python3

import os
import tensorflow as tf
import keras

os.environ["KERAS_BACKEND"] = "tensorflow"

INPUT_SIZE = 40
OUTPUT_SIZE = 1

class DNNModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential([
            keras.layers.InputLayer(shape=(INPUT_SIZE,)),
            keras.layers.Dense(100, activation='sigmoid', kernel_initializer='glorot_normal', name='dense_1'),
            keras.layers.Dense(100, activation='sigmoid', kernel_initializer='glorot_normal', name='dense_2'),
            keras.layers.Dense(100, activation='sigmoid', kernel_initializer='glorot_normal', name='dense_3'),
            keras.layers.Dense(OUTPUT_SIZE, name='output')
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.MeanSquaredError())

        self.signatures = [
            'train',
            'infer',
            'save',
            'restore'
        ]

    @tf.function(input_signature=[tf.TensorSpec([None, INPUT_SIZE], tf.float32, name="inputs")])
    def __call__(self, x):
        return self.model(x)

    @tf.function(input_signature=[
        tf.TensorSpec([None, INPUT_SIZE], tf.float32),
        tf.TensorSpec([None, OUTPUT_SIZE], tf.float32),
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
        tf.TensorSpec([None, INPUT_SIZE], tf.float32),
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

    def convert_to_tflite(self, model_path, tflite_path, infer_only=False):
        signature_keys = self.signatures if not infer_only else ['infer']

        converter = tf.lite.TFLiteConverter.from_saved_model(
            model_path,
            signature_keys=signature_keys)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]

        tflite_model = converter.convert()
        converter.experimental_enable_resource_variables = True
        open(tflite_path, "wb").write(tflite_model)


def main():
    model = DNNModel()

    # Save Model
    model_path = "./dnn_model"
    model.save_model(model_path)

    # Convert to tflite trainable model
    tflite_path = "dnn_model_trainable.tflite"
    model.convert_to_tflite(model_path, tflite_path)

    # Convert to tflite inference only model
    tflite_path = "dnn_model_infer.tflite"
    model.convert_to_tflite(model_path, tflite_path, infer_only=True)

if __name__ == "__main__" :
    main()