package org.deeplearning4j.rl4j.space;

/**
 * @param <A> the type of Action
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *         <p>
 *         Should contain contextual information about the Action space, which is the space of all the actions that could be available.
 *         Also must know how to return a randomly uniformly sampled action.
 */
public interface ActionSpace<A> {

    /**
     * @return A randomly uniformly sampled action,
     */
    A randomAction();

    void setSeed(int seed);

    Object encode(A action);

    int getSize();

    A noOp();


}
