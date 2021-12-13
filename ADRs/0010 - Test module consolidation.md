# Testing

## Status
**Proposed**

Proposed by: Adam Gibson (13th December 2021)


## Context

Testing historically on a large code base like deeplearning4j often involves
platform specific code with several categories as documented in
[the Test Architectures ADR](./0006 - Test architecture.md)


There are multiple levels of testing which test ever larger chunks of the application.

**Unit tests** typically test code at the smallest possible unit. In Java this usually encompasses just a single class.

**Component tests** are meant to test a component consisting of multiple units working together. A logical component
usually consists of a few classes at most and don't cross the boundary between two components.

**Integration tests** are meant to test how components integrate with each other. Their most important job is to ensure
that those components properly interface with each other.

**End-to-End tests**, often also called system tests, are meant to test the entire system. They interface with the
application through the same UI as a regular user does.

**Regression tests** are meant to mimic a specific behavior or usage that results in a bug. They are created before
the bug fix and need to reproduce the bug but expect the correct behavior, i.e. they should fail at first. Once the bug
is fixed, they should pass without any change in the test definition. These test cases accumulate as bug reports come in
and guard us from recreating that particular bug in that particular situation.


The Eclipse Deeplearning4j project has mostly what we would call End-To-End tests.
We want to run a set of tests on different classifiers (eg: cuda version + cudnn, cuda version + non cudnn, cpu, arm32,arm64,..)
in order to verify platform specific behavior works as intended.

When testing, we generally have a few things we test the behavior of:
1. Compatibility across backends
2. Performance 
3. Regressions in behavior (gradient checks failing, ops providing wrong results)
4. Different runtime tests: standalone, spark


Verifying behavior across these different backends even at release time
is time-consuming and error-prone taking hours to run with some tests
being inconsistent (oftentimes spark and multi threading clashing with OMP math threads causing crahses/slowdowns)


## Proposal

We put anything that is considered an end-to-end test requiring platform
specific behavior in to its own module.
These tests would already be tagged. We would have an accompanying pom.xml
that accommodated downloading snapshots to allow us to run specified tests
on different classifiers.

The goal would be to allow specifying the following parameters from the command line:
1. Classifiers to run
2. Groups of tests to run
3. Version to run (defaults to the latest snapshots)


Future work may extend this behavior to add performance tests as well.

The intended workflow would be to allow the following steps:
1. Clone the code base
2. Cd in to the test module
3. Specify the combination of tests you want to run on which platform

This allows easy configuration on CI and creation of different scripts for validation
along the lines of behavior we want to run. Examples include:
1. run model import tests (keras, tensorflow, onnx)
2. Run spark tests
3. Run basic dl4j tests


These distinctions would be achieved through a mix of test tags and test
name filters.



## Consequences

### Advantages
* Tests become more accessible
* It becomes much easier to set up test suites to be run on different classifiers on CI
as a recurring job
* Release testing/validation on specific platforms like embedded pis, nanos don't require you to build the
binaries, but instead you can just download them and run binaries cross compiled on CI to verify behavior
* Allows specifying older versions of library as necessary 

### Disadvantages
* Lose old behavior with tests breaking old assumptions causing contributors
to learn a specific way of running tests
* Requires discipline when tagging tests
* A fairly complex pom.xml will be required for flexibly running tests