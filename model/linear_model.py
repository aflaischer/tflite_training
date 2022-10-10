import tensorflow as tf
import numpy as np
from pathlib import Path

INPUT_SIZE = 1
OUTPUT_SIZE = 1

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(INPUT_SIZE,)),
            tf.keras.layers.Dense(10, activation='relu', kernel_initializer='glorot_normal', name='dense_1'),
            tf.keras.layers.Dense(OUTPUT_SIZE, name='output')
        ])

        self.model.compile(
            optimizer='sgd', #tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError()
                ]
            )

    @tf.function(input_signature=[tf.TensorSpec([None, INPUT_SIZE], tf.float32, name="inputs")])
    def __call__(self, x):
        return self.model(x)

    @tf.function(input_signature=[
        tf.TensorSpec([None, INPUT_SIZE], tf.float32),
        tf.TensorSpec([None, OUTPUT_SIZE], tf.float32),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x)
            loss = self.model.loss(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss": loss}
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([None, INPUT_SIZE], tf.float32),
    ])
    def infer(self, x):
        output = self.model(x)
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

    @tf.function(input_signature=[
        tf.TensorSpec([None, INPUT_SIZE], tf.float32),
        tf.TensorSpec([None, OUTPUT_SIZE], tf.float32),
    ])
    def accuracy(self, x, y):
        prediction = self.model(x)
        self.model.compiled_metrics.update_state(y, prediction)
        return_metrics = {}
        for metric in self.model.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

model = MyModel()

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
            model.restore.get_concrete_function(),
        'accuracy' :
            model.accuracy.get_concrete_function()
        })

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

converter = tf.lite.TFLiteConverter.from_saved_model(
    model_path,
    signature_keys=[
        'train',
        'infer',
        'save',
        'restore',
        'accuracy'
        ])
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
converter.experimental_enable_resource_variables = True
open("linear_model.tflite", "wb").write(tflite_model)