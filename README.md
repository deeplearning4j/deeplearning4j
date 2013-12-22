Deep Learning for Java
=====================================

Leverages matrix operations built on top of 

the BLAS linear algebra libraries for faster 

performance over your standard java libraries.

Includes the following algorithms:

DBN - Deep belief networks; restricted boltzmann machines stacked as layers
CDBN - Continuous Deep Belief Networks; contiuous layer at the front
RBM - Restricted Boltzmann Machines
CRBM - Continuous Restricted Boltzmann Machines
SdA- Stacked Denoising AutoEncoders
DenoisingAutoEncoders



Typically building a network will look something like this.



        BaseMultiLayerNetwork matrix = new BaseMultiLayerNetwork.Builder<>()
                                .numberOfInputs(conf.getInt(N_IN)).numberOfOutPuts(conf.getInt(OUT)).withClazz(conf.getClazz(CLASS))
                                .hiddenLayerSizes(conf.getIntsWithSeparator(LAYER_SIZES, ",")).withRng(rng)
                                .build();


Configuration is based on the constants specified in DeepLearningConfigurable.

Maven central and other support coming soon.



Apache 2 Licensed
