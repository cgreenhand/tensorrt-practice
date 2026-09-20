#pragma once

#ifdef _WIN32
#  ifdef ADD_CONSTANT_EXPORTS
#    define ADD_CONSTANT_API __declspec(dllexport)
#  else
#    define ADD_CONSTANT_API __declspec(dllimport)
#  endif
#else
#  define ADD_CONSTANT_API
#endif

// Both builder and runtime must call this before using the plugin.
extern "C" ADD_CONSTANT_API bool registerAddConstantPlugin() noexcept;
