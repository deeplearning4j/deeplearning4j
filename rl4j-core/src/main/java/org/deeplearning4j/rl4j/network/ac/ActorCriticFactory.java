package org.deeplearning4j.rl4j.network.ac;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 */
public interface ActorCriticFactory {

    IActorCritic buildActorCritic(int shapeInputs[], int numOutputs);

}
