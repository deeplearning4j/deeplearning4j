package org.deeplearning4j.rl4j.learning.sync;

import java.util.ArrayList;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 *
 * Common Interface for Experience replays
 *
 * A prioritized Exp Replay could be implemented by changing the interface
 * and integrating the TD-error in the transition for ranking
 * Not a high priority feature right now
 *
 * The memory is optimised by using array of INDArray in the transitions
 * such that two same INDArrays are not allocated twice
 */
public interface IExpReplay<A> {

    /**
     * @return a batch of uniformly sampled transitions
     */
    ArrayList<Transition<A>> getBatch();

    /**
     *
     * @param transition a new transition to store
     */
    void store(Transition<A> transition);

}
