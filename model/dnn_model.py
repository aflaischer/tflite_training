import tensorflow as tf
import numpy as np
from pathlib import Path

INPUT_SIZE = 40
OUTPUT_SIZE = 1

class DNNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(INPUT_SIZE,)),
            tf.keras.layers.Dense(100, activation='sigmoid', kernel_initializer='glorot_normal', name='dense_1'),
            tf.keras.layers.Dense(100, activation='sigmoid', kernel_initializer='glorot_normal', name='dense_2'),
            tf.keras.layers.Dense(100, activation='sigmoid', kernel_initializer='glorot_normal', name='dense_3'),
            tf.keras.layers.Dense(OUTPUT_SIZE, name='output')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError())

    @tf.function(input_signature=[tf.TensorSpec([None, INPUT_SIZE], tf.float32, name="inputs")])
    def __call__(self, features):
        return self.model(features)

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
            "output": output
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
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
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors

model = DNNModel()

model_path = "./output"
tf.saved_model.save(
    model,
    model_path,
    signatures={
        'train' :
            model.train.get_concrete_function(),
        'infer' :
            model.infer.get_concrete_function(),
        'save' :
            model.save.get_concrete_function(),
        'restore' :
            model.restore.get_concrete_function()
        })

# Convert to tflite trainable model

converter = tf.lite.TFLiteConverter.from_saved_model(
    model_path,
    signature_keys=[
        'train',
        'infer',
        'save',
        'restore'
        ])
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
converter.experimental_enable_resource_variables = True
open("dnn_model_trainable.tflite", "wb").write(tflite_model)

# Convert to tflite inference only model

model_path = "./output"
tf.saved_model.save(
    model,
    model_path,
    signatures=model.infer.get_concrete_function()
    )

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
]
tflite_model = converter.convert()
open("dnn_model_infer.tflite", "wb").write(tflite_model)