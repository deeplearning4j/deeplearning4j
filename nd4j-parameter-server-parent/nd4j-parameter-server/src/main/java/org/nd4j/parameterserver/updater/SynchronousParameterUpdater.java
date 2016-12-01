package org.nd4j.parameterserver.updater;

import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Adds the 2 arrays together,
 * synchronizing when
 * all updates have been collected.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class SynchronousParameterUpdater implements ParameterServerUpdater {

    private int workers = Runtime.getRuntime().availableProcessors();
    private int accumulatedUpdates;

    public SynchronousParameterUpdater(int workers) {
        this.workers = workers;
    }

    /**
     * Reset internal counters
     * such as number of updates accumulated.
     */
    @Override
    public void reset() {
        accumulatedUpdates = 0;
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
        return accumulatedUpdates == workers;
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
        result.addi(arr);
        accumulatedUpdates++;
    }
}
