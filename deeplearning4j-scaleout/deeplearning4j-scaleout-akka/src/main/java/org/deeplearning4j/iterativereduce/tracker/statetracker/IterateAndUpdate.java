package org.deeplearning4j.iterativereduce.tracker.statetracker;

import com.google.common.base.Function;

/**
 * Iterates and updates over the possible updates.
 * This is meant for use by the {@link org.deeplearning4j.iterativereduce.tracker.statetracker.UpdateSaver}
 * to handle iterating over the ids and doing something with the updates. This will usually be via the:
 * {@link org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator}
 * to handle collapsing updates avoiding having them all in memory at once
 * @author Adam Gibson
 */
public interface IterateAndUpdate<E>  {


    /**
     * The accumulated result
     * @return the accumulated result
     */
    public E accumulated();
    /**
     * Accumulates the updates in to a result
     * by iterating over each possible worker
     * and obtaining the mini batch updates for each.
     */
    void accumulate() throws Exception;


}
