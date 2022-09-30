# tflite_training

## Compile tflite libraries:

### For C++:

clone tensorflow github then in directory:

bazel build //tensorflow/lite:libtensorflowlite.so
bazel build -c opt --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex

### For Python:

clone tensorflow github then in directory:

bazel build -c opt --define=tflite_convert_with_select_tf_ops=true //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tf_pip
cd /tmp/tf_pip
pip install tensorflow-2.10.0-cp38-cp38-linux_x86_64.whl

## Compile example:

mkdir build
cd build; cmake ..
make

## Generate model:

cd model; python linear_model.py

## Run testApp:

./testmodel <path_to_model>