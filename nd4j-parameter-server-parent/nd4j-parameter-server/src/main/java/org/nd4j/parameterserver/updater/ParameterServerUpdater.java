package org.nd4j.parameterserver.updater;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * A parameter server updater
 * for applying updates on the parameter server
 *
 * @author Adam Gibson
 */
public interface ParameterServerUpdater {


    /**
     * Returns the current status of this parameter server
     * updater
     * @return
     */
    Map<String,Number> status();

    /**
     * Serialize this updater as json
     * @return
     */
    String toJson();

    /**
     * Reset internal counters
     * such as number of updates accumulated.
     */
    void reset();

    /**
     * Returns true if
     * the updater has accumulated enough ndarrays to
     * replicate to the workers
     * @return true if replication should happen,false otherwise
     */
    boolean shouldReplicate();

    /**
     * Updates result
     * based on arr
     * @param arr the array to update
     * @param result the result ndarray to update
     */
    void update(INDArray arr,INDArray result);
}
