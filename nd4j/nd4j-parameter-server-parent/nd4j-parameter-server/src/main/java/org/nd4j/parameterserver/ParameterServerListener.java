package org.nd4j.parameterserver;

import lombok.Data;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.aeron.ndarrayholder.InMemoryNDArrayHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.updater.ParameterServerUpdater;
import org.nd4j.parameterserver.updater.SynchronousParameterUpdater;
import org.nd4j.parameterserver.updater.storage.NoUpdateStorage;


/**
 * Parameter server
 * listener. This holds an
 * {@link INDArray} in memory
 * and maintains the "master copy"
 *
 * of the ndarray.
 * @author Adam Gibson
 */
@Data
public class ParameterServerListener implements NDArrayCallback {
    private ParameterServerUpdater updater;
    private boolean master;
    private int[] shape;

    /**
     * Shape of the ndarray
     * @param shape the shape of the array
     * @param updatesPerEpoch  the number of updates per epoch
     *                         for synchronization
     */
    public ParameterServerListener(int[] shape, int updatesPerEpoch) {
        updater = new SynchronousParameterUpdater(new NoUpdateStorage(), new InMemoryNDArrayHolder(shape),
                        updatesPerEpoch);
    }

    /**
     * Shape of the ndarray
     * @param shape the shape of the array
     */
    public ParameterServerListener(int[] shape) {
        this(shape, Runtime.getRuntime().availableProcessors());
    }


    /**
     *
     * @param shape the shape of the array
     * @param updater the updater to use for this server
     */
    public ParameterServerListener(int[] shape, ParameterServerUpdater updater) {
        this.updater = updater;
        this.shape = shape;

    }

    /**
     * A listener for ndarray message
     *
     * @param message the message for the callback
     */
    @Override
    public void onNDArrayMessage(NDArrayMessage message) {
        updater.update(message);
    }

    /**
     * Used for partial updates using tensor along
     * dimension
     *  @param arr        the array to count as an update
     * @param idx        the index for the tensor along dimension
     * @param dimensions the dimensions to act on for the tensor along dimension
     */
    @Override
    public synchronized void onNDArrayPartial(INDArray arr, long idx, int... dimensions) {
        updater.partialUpdate(arr, updater.ndArrayHolder().get(), idx, dimensions);
    }

    /**
     * Setup an ndarray
     *
     * @param arr
     */
    @Override
    public synchronized void onNDArray(INDArray arr) {
        if (shape == null)
            updater.update(arr.reshape(1, arr.length()), updater.ndArrayHolder().get());
        else
            updater.update(arr, updater.ndArrayHolder().get());
    }

    /**
     * Do a final divide for averaging
     */
    public synchronized void finish() {
        updater.ndArrayHolder().get().divi(updater.numUpdates());
    }


}
