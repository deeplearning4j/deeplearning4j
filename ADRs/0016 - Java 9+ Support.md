# Java 9+ Support

## Status
**Discussion**

Proposed by: Adam Gibson (7th Feb 2022)


## Context

With the next LTS upcoming (java 17)  forcing module metadata to be present. This means supporting java 9+ modules
with mdoule-info.java being present.

This ADR addresses the changes made to support java 9 modules. Many of the changes are related to adding
[module-info.java](https://www.oracle.com/corporate/features/understanding-java-9-modules.html)

## Proposal

1. [moditect plugin](https://github.com/moditect/moditect) metadata for every module
   a. Each module has a module.name with 1 inherited declaration in the root pom for generating and adding 
       module metadata.
   b. Github Workflow upgrades for generating module metadata before publishing
   c. Moditect plugin declarations are optional  by default and selectively invoked

2. Add build steps invoking the plugin ensuring proper metadata gets added

3. Each maven jar plugin invocation needs a unique module name for jars that have classifiers such as
the nd4j backends this avoids duplicate name errors when trying to invoke metadata

4. Proper package split package declarations enable proper module exposure. This means some classes have been moved
around. Mainly internal classes are affected such as the preset names.

## Consequences

### Advantages

* Allows easier integration in to various module systems including jigsaw and OSGI
* Allows proper integration in to newer java versions enabling users to produce modularized applications
* Allows better support for newer java versions
* Our plugin approach enables backwards compatibility with java 8


### Disadvantages
* Breaking backwards compatibility
* May introduce new bugs due to renaming