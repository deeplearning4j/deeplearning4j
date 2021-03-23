# Junit 5 tag usage

## Status
Proposed

Proposed by: Adam Gibson (21-03-2021)

Discussed with: N/A

## Context
DL4J was a junit 4 based code based for testing.
It's now based on junit 5's jupiter API, which has support for [Tags](https://junit.org/junit5/docs/5.0.1/api/org/junit/jupiter/api/Tag.html).

DL4j's code base has a number of different kinds of tests that fall in to several categories:
1. Long and flaky involving distributed systems (spark, parameter-server)
2. Code that requires large downloads, but runs quickly
3. Quick tests that test basic functionality
4. Comprehensive integration tests that test several parts of a code  base

Due to the variety of behaviors across different tests, it's hard to tell what's actually needed
for running and validating whether changes work against such a complex test base.

Much of the time, most of the tests aren't related to a given change.
Often times, quick sanity checks are all that's needed in order to make sure a change works.

A common set of tags is used to filter which tests are needed to run when.
This allows us to retain complex integration tests and run them on a set schedule
to catch regressions while allowing a defined subset of tests to run for a quick feedback loop.




## Decision

A few kinds of tags exist:
1. Time based: long-time,short-time
2. Network based: has-download
3. Distributed systems: spark, multi-threaded
4. Functional cross-cutting concerns: multi module tests, similar functionality (excludes time based)
5. Platform specific tests that can vary on different hardware: cpu, gpu
6. JVM crash: (jvm-crash) Tests with native code can crash the JVM for tests. It's useful to be able to turn those off when debugging.: jvm-crash
7. RNG: (rng) for RNG related tests
8. Samediff:(samediff) samediff related tests
9. Training related functionality


## Consequences
### Advantages
* Ability to sort through and filter tests based on different running environments

* Ability to reason about test suites as a whole dynamically across modules

* Avoid the need to define test suites

* Ability to define groups of tags based in profiles 

* Ability to dynamically filter tests from the maven command line


### Disadvantages

* Documentation and maintenance burden needing to know what tags do what

* Test maintenance for newcomers who may not know how to tag tests


