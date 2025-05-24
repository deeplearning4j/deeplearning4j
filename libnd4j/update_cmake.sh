#!/bin/bash

# Find and update the template locations section in CMakeLists.txt
sed -i '/# Define template locations/,/set(GEN_COMPILATION_TEMPLATES/c\
# Define template locations\
set(INSTANTIATION_TEMPLATES_3\
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in"\
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_template_3.cpp.in"\
)\
\
set(INSTANTIATION_TEMPLATES_2\
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in"\
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_template_2.cpp.in"\
)\
\
set(GEN_COMPILATION_TEMPLATES\
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/compilation_units/pairwise_instantiation_template.cpp.in"' /home/agibsonccc/Documents/GitHub/deeplearning4j/libnd4j/CMakeLists.txt
