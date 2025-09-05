# tflite_training

## Compile tflite libraries:

Clone tensorflow github, e.g.:

    it clone --depth 1 --branch v2.19.0  https://github.com/tensorflow/tensorflow.git

### For C++:

In tensorflow src repository:

    ./configure
    bazel build -c opt --cxxopt='--std=c++17' --copt=-Wno-gnu-offsetof-extensions //tensorflow/lite:libtensorflowlite.so
    bazel build -c opt --cxxopt='--std=c++17' --copt=-Wno-gnu-offsetof-extensions --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex

### For Python:

In tensorflow src repository:

    ./configure
    bazel build -c opt --define=tflite_convert_with_select_tf_ops=true --copt=-Wno-gnu-offsetof-extensions //tensorflow/tools/pip_package:wheel --repo_env=USE_PYWRAP_RULES=1 --repo_env=WHEEL_NAME=tensorflow_cpu
    cd bazel-bin/tensorflow/tools/pip_package/wheel_house/; pip install tensorflow_cpu-2.19.0-cp312-cp312-linux_x86_64.whl


## Compile example:

    mkdir build
    cd build; cmake ..
    make

## Generate model:

    cd model; python linear_model.py

## Run testApp:

    ./testmodel <path_to_model>