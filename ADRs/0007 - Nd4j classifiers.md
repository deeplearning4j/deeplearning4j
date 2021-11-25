# Nd4j Classifiers

## Status
Accepted

Proposed by: Adam Gibson (5-5-2021)

Discussed with: saudet  (Samuel Audet)

## Context
Nd4j relies upon the c++ library [libnd4j](../libnd4j) for native math execution.
It uses [javacpp](https://github.com/bytedeco/javacpp) to link against 
libnd4j. Libnd4j is capable of being compiled a myriad of different ways allowing different trade offs to be made
in terms of performance and dependencies. This presents complexity in exposing this flexibility to the end user.



## Decision

In order to allow users to pick which configuration they would like to use, while avoiding adding a lot of different artifact
ids to the project, the following javacpp platform extensions are used:
compiled type (avx etc or blank if normal) - software linked against (cudnn, onednn, armcompute) - version 


An example for the one dnn platform extension could be:
dnnl-2.2
avx256-dnnl-2.2

This presents 2 examples where a special compilation is enabled and one where it's not
both linking against dnnl/onednn 2.2.


## Discussion

Saudet: Javacpp's extensions can actually support optional
inclusion. It plays well with other platform declarations.
This means you can use platform like:
```bash
mvn -Djavacpp.platform.extension=-avx512  -Djavacpp.platform=windows-x86_64 clean ...
```

to enable certain extensions.

## Consequences
### Advantages
* Adds more extensions than the previous release. This allows nd4j-native/cuda to play nice
with the standard javacpp.platform scheme when [reducing the number of dependencies](https://github.com/bytedeco/javacpp-presets/wiki/Reducing-the-Number-of-Dependencies)
  while also enabling the new accelerated extensions, but in an optional manner.
  
* Allows users to pick how they want libnd4j to be included in their build

* Maintains 2 artifact ids people have to know without too much extensive knowledge.

* Allows us to keep a sane default for people with optimizations being optional


### Disadvantages

* Could be deprecated in the future depending on how libnd4j evolves

* Complexity for the user with the number of new extensions to be used.


