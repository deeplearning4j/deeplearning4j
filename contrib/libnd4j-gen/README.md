# Libnd4j Op Descriptor Generator

This module contains a few files for generating op descriptors for libnd4j.
An op descriptor contains its number of inputs and outputs as well as the names
in the code of those attributes when they are assigned.

The main class parses a root directory specified by the user containing the libnd4j code base.
In the libnd4j code base, it will scan for op signatures and certain macros found in https://github.com/eclipse/deeplearning4j/blob/master/libnd4j/include/system/op_boilerplate.h

From that, it will automatically generate a set of op descriptors including counts of types of ops as well as names.
The up to date OpDescriptor class can be found [here](src/main/java/org/nd4j/descriptor/OpDescriptor.java)
The main class can be found [here](src/main/java/org/nd4j/descriptor/ParseGen.java)




