if(NOT TENSORRT_ROOT)
    message(FATAL_ERROR
        "TENSORRT_ROOT is empty. "
        "Configure with -DTENSORRT_ROOT=/path/to/TensorRT"
    )
endif()

find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    HINTS
        "${TENSORRT_ROOT}/include"
    REQUIRED
)

find_library(
    TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer
    HINTS
        "${TENSORRT_ROOT}/lib"
        "${TENSORRT_ROOT}/lib64"
    REQUIRED
)

find_library(
    TENSORRT_ONNXPARSER_LIBRARY
    NAMES nvonnxparser
    HINTS
        "${TENSORRT_ROOT}/lib"
        "${TENSORRT_ROOT}/lib64"
    REQUIRED
)

add_library(tensorrt_nvinfer UNKNOWN IMPORTED GLOBAL)

set_target_properties(
    tensorrt_nvinfer
    PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
)

add_library(TensorRT::nvinfer ALIAS tensorrt_nvinfer)

add_library(tensorrt_onnxparser UNKNOWN IMPORTED GLOBAL)

set_target_properties(
    tensorrt_onnxparser
    PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_ONNXPARSER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
)

add_library(TensorRT::onnxparser ALIAS tensorrt_onnxparser)