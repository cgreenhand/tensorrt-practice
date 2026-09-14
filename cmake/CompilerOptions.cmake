add_library(learning_compiler_options INTERFACE)

target_compile_features(
    learning_compiler_options
    INTERFACE
        cxx_std_17
)

if(MSVC)
    target_compile_options(
        learning_compiler_options
        INTERFACE
            /W4
    )
else()
    target_compile_options(
        learning_compiler_options
        INTERFACE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

add_library(
    Learning::CompilerOptions
    ALIAS learning_compiler_options
)