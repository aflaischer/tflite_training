# tflite_training

## Compile tflite libraries:

Clone tensorflow github, e.g.:

    git clone --depth 1 --branch v2.16.1 https://github.com/tensorflow/tensorflow.git

### For C++:

In tensorflow src repository:

    ./configure
    bazel build -c opt --cxxopt='--std=c++17' --copt=-Wno-gnu-offsetof-extensions //tensorflow/lite:libtensorflowlite.so
    bazel build -c opt --cxxopt='--std=c++17' --copt=-Wno-gnu-offsetof-extensions --config=monolithic tensorflow/lite/delegates/flex:tensorflowlite_flex

### For Python:

In tensorflow src repository:

    ./configure
    bazel build -c opt --define=tflite_convert_with_select_tf_ops=true //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tf_pip
    cd /tmp/tf_pip; pip install tensorflow-2.16.1-cp311-cp311-linux_x86_64.whl


## Compile example:

    mkdir build
    cd build; cmake ..
    make

## Generate model:

    cd model; python linear_model.py

## Run testApp:

    ./testmodel <path_to_model>