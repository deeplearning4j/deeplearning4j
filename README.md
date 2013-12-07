SDA - JBlas
=========================


An implementation of Stacked Denoising Auto Encoders with jblas in mind.

Classes in the *.berkeley package are imports from 

berkeley's utility package.

Basic idea: use binomial noise to corrupt examples

and come up with an appoximate distribution using 

a probability approximation based on reconstructing corrupted input

from noise.

A few pointers:

Use high noise for low example datasets.


You typically don't need more than 8-10 layers for most problems.

Usually the more data the more neurons is better if possible.

You have the option to switch out the sigmoid implementation in there

for tanh as well. I'm going to make this a bit more modular soon.

More to come!


Apache 2 licensed
