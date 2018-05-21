package org.deeplearning4j.rl4j.mdp;


import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 * An interface that ensure an environment is expressible as a
 * Markov Decsision Process. This implementation follow the gym model.
 * It works based on side effects which is perfect for imperative simulation.
 *
 * A bit sad that it doesn't use efficiently stateful mdp that could be rolled back
 * in a "functionnal manner" if step return a mdp
 *
 */
public interface MDP<O, A, AS extends ActionSpace<A>> {

    ObservationSpace<O> getObservationSpace();

    AS getActionSpace();

    O reset();

    void close();

    StepReply<O> step(A action);

    boolean isDone();

    MDP<O, A, AS> newInstance();

}
