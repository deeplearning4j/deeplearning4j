cmake_minimum_required(VERSION 2.8.2)

project(flatbuffers-download NONE)

include(ExternalProject)
ExternalProject_Add(flatbuffers
  GIT_REPOSITORY    https://github.com/KonduitAI/flatbuffers.git
  GIT_TAG           main
  SOURCE_DIR        "/home/agibsonccc/Documents/GitHub/deeplearning4j/build/flatbuffers-src"
  BINARY_DIR        "/home/agibsonccc/Documents/GitHub/deeplearning4j/build/flatbuffers-build"
  CONFIGURE_COMMAND ""
  CMAKE_ARGS        "-DFLATBUFFERS_BUILD_FLATC=OFF"
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
