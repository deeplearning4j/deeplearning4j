package org.deeplearning4j.rl4j.network.ac;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 *
 * A factory for Actor Critic. Extend this to implement and provide your own
 * Actor Critic!
 */
public interface ActorCriticFactoryCompGraph {

    IActorCritic buildActorCritic(int shapeInputs[], int numOutputs);

}
