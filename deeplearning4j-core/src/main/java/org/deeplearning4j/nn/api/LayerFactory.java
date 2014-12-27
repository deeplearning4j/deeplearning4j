package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 *
 * Common interface for creating neural network layers.
 *
 *
 * @author Adam Gibson
 */
public interface LayerFactory {




    /**
     *
     * Create a layer based on the based in configuration
     * and an added context.
     * @param conf the configuration to create the layer based on
     * @param index the index of the layer
     * @param numLayers the number of total layers in the net work
     * @return the created layer
     */
    Layer create(NeuralNetConfiguration conf,int index,int numLayers);

    /**
     *
     * Create a layer based on the based in configuration
     * @param conf the configuration to create the layer based on
     * @return the created layer
     */
    Layer create(NeuralNetConfiguration conf);


    /**
     * Get the param initializer used for initializing layers
     * @return the param initializer
     */
    ParamInitializer getInitializer();


}
