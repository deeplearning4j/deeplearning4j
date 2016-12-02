package org.nd4j.parameterserver.updater;

import org.nd4j.aeron.ipc.NDArrayMessage;
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
     * Do an update based on the ndarray message.
     * @param message
     */
    void update(NDArrayMessage message);

    /**
     * Updates result
     * based on arr along a particular
     * {@link INDArray#tensorAlongDimension(int, int...)}
     * @param arr the array to update
     * @param result the result ndarray to update
     * @param idx the index to update
     * @param dimensions the dimensions to update
     */
    void partialUpdate(INDArray arr,INDArray result, long idx, int...dimensions);

    /**
     * Updates result
     * based on arr
     * @param arr the array to update
     * @param result the result ndarray to update
     */
    void update(INDArray arr,INDArray result);
}
