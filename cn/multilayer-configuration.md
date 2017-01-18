---
title: MultiLayerConfiguration class
layout: default
---

# MultiLayerConfiguration class:
*DL4J MultiLayer Neural Net Builder Basics*

For creating a deep learning network in Deeplearning4j, the foundation is the MultiLayerConfiguration constructor. Below are the parameters for this configuration and the default settings.

A multilayer network will accept the same kinds of inputs as a single-layer network. The multilayer network parameters are also typically the same as their single-layer network counterparts.

How to start constructing the class in Java for a multilayer network:

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

Append parameters onto this class by linking them up as follows:

    new NeuralNetConfiguration.Builder().iterations(100).layer(new RBM())
        .nIn(784).nOut(10).list(4).hiddenLayerSizes(new int[]{500, 250, 200})
        .override(new ClassifierOverride(3))
        .build();

Parameters:

While the following parameters are specific to multilayer networks, the same inputs for a single-layer neural network
will also work for a multilayer network.

- **hiddenLayerSizes**: *int[]*, number of nodes for the feed forward layer
   - two layers format = new int[]{50} = initiate int array with 50 nodes
   - five layers format = new int[]{32,20,40,32} = layer 1 is 32 nodes, layer 2 is 20 nodes, etc
- **list**: *int*, number of layers; this function replicates your configuration n times and builds a layerwise configuration
    - do not include input in the layer count
- **override**: (*int*, *class*) {layer number, data processor class}, override with specific layer configuration
    - When you're building a multilayer network, you won't necessarily want the same configuration for all layers.
    - In that case, you should use the override method to modify any configurations that are necessary.
    - Use override(), and specify the layer you want to modify, as well as the builder to override values for.
    - Example for setting up a convolutional layer as the first layer in a deep network:

            .override(0, new ConfOverride() {
                        public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                            builder.layer(new ConvolutionLayer());
                            builder.convolutionType(ConvolutionLayer.ConvolutionType.MAX);
                            builder.featureMapSize(2, 2);
                        }
                    })

    - Example for overriding the second/last layer of a neural network to be a classifier:

            .override(new ClassifierOverride(1))

- **useDropConnect**: *boolean*, a generalization of dropout; a randomly selected subset of weights within the neural network set to zero.
    - default = false

For more information on this class, checkout [Javadocs](http://deeplearning4j.org/doc/).
