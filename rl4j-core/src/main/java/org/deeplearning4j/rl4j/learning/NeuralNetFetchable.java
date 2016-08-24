package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.network.NeuralNet;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/19/16.
 *
 * Common interface for all training that contain a neural net
 * fetchable for evaluation
 */
public interface NeuralNetFetchable<NN extends NeuralNet> {

    NN getNeuralNet();
}
