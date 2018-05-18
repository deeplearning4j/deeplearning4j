package org.nd4j.weightinit;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Defines weight initialization for neural networks.
 *
 * Use {@link BaseWeightInitScheme}
 * to create a new {@link WeightInitScheme}
 * This is needed to  handle things like the parameters view.
 *
 * @author Adam Gibson
 */
public interface WeightInitScheme {

    /**
     * Create the array
     * @param shape the shape of the array
     * @param paramsView the parameters view
     * @return the created array
     */
    INDArray create(long[] shape,INDArray paramsView);



    /**
     * Create the array
     * @param shape the shape of the array
     * @return the created array
     */
    INDArray create(long... shape);


    /**
     * The order of the weight init
     * @return
     */
    char order();

    /**
     * The type of the weight init
     * @return
     */
    WeightInit type();

}
