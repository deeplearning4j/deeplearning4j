package org.nd4j.parameterserver.updater;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 12/1/16.
 */
public class SoftSyncParameterUpdater implements ParameterServerUpdater  {
    //s is the number of updates
    private int s;
    private int currentVersion;
    private int accumulatedUpdates = 0;
    private double scalingFactor;


    /**
     * Reset internal counters
     * such as number of updates accumulated.
     */
    @Override
    public void reset() {
        currentVersion++;
    }

    /**
     * Returns true if
     * the updater has accumulated enough ndarrays to
     * replicate to the workers
     *
     * @return true if replication should happen,false otherwise
     */
    @Override
    public boolean shouldReplicate() {
        return accumulatedUpdates == s;
    }

    /**
     * Updates result
     * based on arr
     *
     * @param arr    the array to update
     * @param result the result ndarray to update
     */
    @Override
    public void update(INDArray arr, INDArray result) {

    }
}
