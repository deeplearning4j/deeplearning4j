package org.deeplearning4j.gym.space;

/**
 * @author rubenfiszel on 7/13/16.
 *
 * Generalize every Observation being encoded on a few dimension.
 * If the observation is used as input of a DQN, you can assume that
 * it is sufficient to not reduce its dimension unlike high-dimensional
 * observation like pixels that require to be "pre-processed" by convolutional layers.
 */
public interface LowDimensional {

    /**
     * $
     * encodes all the information of an Observation in an array double and can be used as input of a DQN directly
     *
     * @return the encoded informations
     */
    double[] toArray();

}
