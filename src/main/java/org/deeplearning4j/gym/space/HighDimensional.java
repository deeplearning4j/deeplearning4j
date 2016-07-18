package org.deeplearning4j.gym.space;

/**
 * @author rubenfiszel on 7/13/16.
 *
 * Generalize every Observation being encoded on a high dimensions (like pixels of a screen).
 * If the observation is used as input of a DQN, you should use a convolutional layer to reduce its dimension.
 */
public interface HighDimensional {

    double[][] to2DArray();

}
