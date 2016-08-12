package org.deeplearning4j.rl4j.network.dqn;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/27/16.
 */
public interface DQNFactory {

    IDQN buildDQN(int shapeInputs[], int numOutputs);

}
