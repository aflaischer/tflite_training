file(GLOB ${PROJECT_NAME}_EXE_SRC
    src/*.cpp
)

set(TENSORFLOW_SOURCE_DIR "/tmp/tensorflow" CACHE PATH
  "Directory that contains the TensorFlow project"
)

find_library(TF_LIB tensorflowlite HINTS "${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow/lite/")
find_library(TF_LIB_FLEX tensorflowlite_flex HINTS "${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow/lite/delegates/flex/")

include_directories(
    "${TENSORFLOW_SOURCE_DIR}/"
    "${TENSORFLOW_SOURCE_DIR}/bazel-bin/external/flatbuffers/src/_virtual_includes/flatbuffers/"
)

add_executable(testmodel ${${PROJECT_NAME}_EXE_SRC})
target_link_libraries(testmodel
-Wl,--no-as-needed # Need --no-as-needed to link tensorflowlite_flex
${TF_LIB}
${TF_LIB_FLEX}
)
