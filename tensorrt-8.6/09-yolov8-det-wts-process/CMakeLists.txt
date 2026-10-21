get_filename_component(THIS_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(SOURCES
    main.cpp
    src/model.cpp
    src/block.cpp
    src/preprocess.cu
    src/postprocess.cu
    plugin/pluginLayer.cu
)

add_executable(${THIS_PROJECT_NAME} ${SOURCES})

target_link_libraries(${THIS_PROJECT_NAME}
    PRIVATE
        Learning::CompilerOptions
        Learning::TensorRTRuntime
)

target_include_directories(${THIS_PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin
)

set_target_properties(${THIS_PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "yolov8_det"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${THIS_PROJECT_NAME}"
)
