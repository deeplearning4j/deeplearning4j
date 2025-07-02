# GCC flags for C++ template duplicate instantiation issues
cmake_minimum_required(VERSION 3.15)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # For C++ template duplicate instantiation errors
   if(!SD_CUDA)
    add_compile_options(-fpermissive)
    # Alternative: Use external template instantiation model
    # add_compile_options(-fno-implicit-templates)

    # For older C++ standards that are more lenient

    # Reduce template instantiation depth (may help with complex templates)
    add_compile_options(-ftemplate-depth=1024)

    # Allow duplicate weak symbols (helps with templates)
    add_compile_options(-fno-gnu-unique)

    message(STATUS "Added -fpermissive: Allows duplicate template instantiations")
    message(STATUS "Added template-related flags for C++ duplicate handling")
endif()

endif()

# The real fix is usually in the code:
# 1. Use 'extern template' declarations in headers
# 2. Only instantiate templates in one .cpp file
# 3. Use proper include guards
# 4. Check for BUILD_TRIPLE_TEMPLATE macro being called multiple times