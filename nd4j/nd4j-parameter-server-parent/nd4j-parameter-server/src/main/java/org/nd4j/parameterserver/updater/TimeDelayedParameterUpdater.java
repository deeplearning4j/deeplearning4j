package org.nd4j.parameterserver.updater;

import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Created by agibsonccc on 12/1/16.
 */
public class TimeDelayedParameterUpdater extends BaseParameterUpdater {
    private long syncTime;
    private long lastSynced;


    /**
     * Returns the number of required
     * updates for a new pass
     *
     * @return the number of required updates for a new pass
     */
    @Override
    public int requiredUpdatesForPass() {
        return 0;
    }

    /**
     * Returns the current status of this parameter server
     * updater
     *
     * @return
     */
    @Override
    public Map<String, Number> status() {
        return null;
    }

    /**
     * Serialize this updater as json
     *
     * @return
     */
    @Override
    public String toJson() {
        return null;
    }

    /**
     * Reset internal counters
     * such as number of updates accumulated.
     */
    @Override
    public void reset() {

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
        long now = System.currentTimeMillis();
        long diff = Math.abs(now - lastSynced);
        return diff > syncTime;
    }

    /**
     * Do an update based on the ndarray message.
     *
     * @param message
     */
    @Override
    public void update(NDArrayMessage message) {

    }

    /**
     * Updates result
     * based on arr along a particular
     * {@link INDArray#tensorAlongDimension(int, int...)}
     *
     * @param arr        the array to update
     * @param result     the result ndarray to update
     * @param idx        the index to update
     * @param dimensions the dimensions to update
     */
    @Override
    public void partialUpdate(INDArray arr, INDArray result, long idx, int... dimensions) {

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
