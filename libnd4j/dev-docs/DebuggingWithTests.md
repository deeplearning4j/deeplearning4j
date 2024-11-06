# Debugging Practices with C++ Tests in libnd4j Library

In this document, we will walk through some general debugging practices while working with C++ tests in the libnd4j library. This involves the use of Google Test framework (gtest), building with debug flags, and employing memory checking tools like Valgrind.

## Prerequisites

Ensure that you have the following installed on your system:

- Google Test framework (gtest)
- CMake (for building the project)
- Valgrind (for memory leak checking)
- A modern C++ compiler that supports C++11 or later

## Building with Debug Flags

In order to debug effectively, you will need to compile your project with debugging information. This can be accomplished by adding debug flags during the build process. 

If you're using CMake, you can do this by setting the `CMAKE_BUILD_TYPE` variable to `Debug`. This can be done through the command line as follows:

```
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

When you're building your project with debug flags, the compiler will include extra information in the resulting executable that can be used to help find bugs.

## Debugging with Google Test (gtest)

Google Test is a powerful framework for writing C++ tests. It provides a simple and flexible way to write reliable tests, including features like test suites and assertions that make it easy to structure your tests and verify your code.

You can run a specific test using the `--gtest_filter` flag. The value passed to `--gtest_filter` is a colon-separated list of wildcard patterns (they may include `*` as a wildcard). Only the tests whose full names match one of the patterns will be executed.

For example, to run the `BasicTest_Scatter_1` test of the `ListOperationsTests` test case, you can use:

```
./blasbuild/cpu/tests_cpu/layers_tests/runtests --gtest_filter=ListOperationsTests.BasicTest_Scatter_1
```

## Debugging with Valgrind

Valgrind is an open-source tool that can be used for memory debugging, memory leak detection, and profiling. If your program is crashing or behaving unexpectedly, Valgrind can help identify memory leaks, uninitialized memory, and other related problems.

To run your tests under Valgrind, use the following command:

```
valgrind --track-origins=yes -v ./blasbuild/cpu/tests_cpu/layers_tests/runtests --gtest_filter=ListOperationsTests.BasicTest_Scatter_1
```

Here's what the flags do:

- `--track-origins=yes`: This tells Valgrind to track the origin of uninitialized values. This can be useful for finding the source of uninitialized value errors.
- `-v`: This increases the verbosity of Valgrind's output, giving you more information about what's happening.
- `--gtest_filter=ListOperationsTests.BasicTest_Scatter_1`: This is telling gtest to run only the `BasicTest_Scatter_1` test from the `ListOperationsTests` test suite.

## Conclusion

This should give you a solid foundation for debugging C++ tests in the libnd4j library. Remember, the key to effective debugging is to take a systematic approach, carefully observing the behavior of your program and making hypotheses about what could be causing any issues.
