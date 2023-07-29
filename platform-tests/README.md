Platform tests
---------------------------


This module is used to run tests for different backends. It is used to test the following backends:

1. CUDA
2. CPU


    
## Running Tests with Maven

Run the project tests using Maven:

```bash
mvn test
```

Override default properties when running tests. For example:

```bash
mvn test -Dtest.heap.size=8g -Dsurefire.forks=2
```

Below are the properties you can override:

- `test.heap.size`: Heap size per test. 
  This is used for dynamic memory allocation.
  Default: 6g.

- `test.offheap.size`: Off-heap memory size per test. 
  This is useful for manual memory management. 
  Default: 6g.

- `surefire.forks`: Number of JVM instances (forks) created for tests.
  Default: 1.

- `surefire.threads`: Number of threads to use per fork.
  Default: 1.

- `test.nogc`: Toggles garbage collector during tests. 
  Disabling can speed up tests but might cause memory management issues.
  Default: true.

- `test.asan.options`: Options for AddressSanitizer, a memory error detector.
  Default: various options.

- `tests`: Defines the test tags to run. 
  This allows for selective testing.
  Default: various tags.

- `excludedTests`: Defines the test tags to exclude. 
  This is useful for excluding long or unstable tests. 
  Default: various tags.

- `preload`: Sets LD_PRELOAD for loading a shared library before others.
  Default: path to jemalloc.

- `jemalloc.path`: Path to the jemalloc library, a malloc implementation.
  Default: path to jemalloc.




## Additional Details About Running Tests 

This testing module comprehensively covers all tests in the code base. Tests are neatly organized and 
categorized using JUnit 5 tags, allowing for selective running and exclusion based on your specific needs.

### Test Tags
Test tags enable selective testing. The default tags for tests to run include:

- `samediff`
- `rng`
- `java-only`
- `dl4j-old-api`
- `ndarray-indexing`
- `compression`
- `loss-functions`
- `keras`
- `python`
- `tensorflow`
- `onnx`
  
  Tests to exclude by default are:

- `large-resources`
- `downloads`
- `long-running-test`
  
  ### Special Considerations

  Please be aware that some tests may take a considerable amount of time to complete 
  or may require downloads. This is particularly true for `large-resources` tests, 
  which may require significant computational resources or time.

  ### dl4j-test-resources

  To run many of the tests, the `dl4j-test-resources` Maven artifact must be installed. 
  You can obtain this artifact from [here](https://github.com/KonduitAI/dl4j-test-resources/archive/refs/heads/master.zip). 
  After downloading, run `mvn install` in the root directory of the extracted project. 
  Note that this JAR file is large but contains all necessary models and resources needed for tests.

  ### JUnit 5 Extensions

  The test architecture utilizes two JUnit 5 extensions:

- [BackendCheckerExtension](https://github.com/deeplearning4j/deeplearning4j/blob/98b85b1e7ba2f7f2effe92796f0d00cdde5e08e1/platform-tests/src/test/java/org/eclipse/deeplearning4j/tests/extensions/BackendCheckerExtension.java#L37): 
This extension disables larger tests with specific tags when tests are run on a GPU. 
This helps manage resource use during GPU testing.

- [DeallocationExtension](https://github.com/deeplearning4j/deeplearning4j/blob/98b85b1e7ba2f7f2effe92796f0d00cdde5e08e1/platform-tests/src/test/java/org/eclipse/deeplearning4j/tests/extensions/DeallocationExtension.java#L11): 
This extension manages deallocating off-heap memory after each test. It hooks into the deallocator service 
and intercepts potential deallocation calls, thereby preventing race conditions between tests ending and deallocation.
  
