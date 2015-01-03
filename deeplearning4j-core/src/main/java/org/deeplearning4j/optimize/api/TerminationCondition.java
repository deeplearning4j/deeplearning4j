package org.deeplearning4j.optimize.api;

/**
 * Created by agibsonccc on 12/24/14.
 */
public interface TerminationCondition {

    /**
     * Whether to terminate based on the given metadata
     * @param cost the new cost
     * @param oldCost the old cost
     * @param otherParams
     * @return
     */
    boolean terminate(double cost,double oldCost,Object[] otherParams);

}
