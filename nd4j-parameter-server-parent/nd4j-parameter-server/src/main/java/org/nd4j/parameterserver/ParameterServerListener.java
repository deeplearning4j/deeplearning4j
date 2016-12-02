package org.nd4j.parameterserver;

import lombok.Data;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.updater.SynchronousParameterUpdater;
import org.nd4j.parameterserver.updater.ParameterServerUpdater;


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
public class ParameterServerListener implements NDArrayCallback,NDArrayHolder {
    private INDArray arr;
    private ParameterServerUpdater updater = new SynchronousParameterUpdater();
    private boolean master;
    private int[] shape;

    /**
     * Shape of the ndarray
     * @param shape the shape of the array
     */
    public ParameterServerListener(int[] shape) {
        this.arr = Nd4j.create(shape);
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
    public synchronized  void onNDArrayPartial(INDArray arr, long idx, int... dimensions) {
       updater.partialUpdate(arr,this.arr,idx,dimensions);
    }

    /**
     * Setup an ndarray
     *
     * @param arr
     */
    @Override
    public synchronized void onNDArray(INDArray arr) {
        if(shape == null)
            updater.update(arr.reshape(1,arr.length()),this.arr);
        else
            updater.update(arr,this.arr);
    }

    /**
     * Do a final divide for averaging
     */
    public synchronized void finish() {
        this.arr.divi(totalUpdates());
    }

    /**
     * The number of updates
     * that have been sent to this older.
     *
     * @return
     */
    @Override
    public int totalUpdates() {
        return updater.numUpdates();
    }

    /**
     * Retrieve an ndarray
     *
     * @return
     */
    @Override
    public synchronized  INDArray get() {
        return arr;
    }

    /**
     * Retrieve a partial view of the ndarray.
     * This method uses tensor along dimension internally
     * Note this will call dup()
     *
     * @param idx        the index of the tad to get
     * @param dimensions the dimensions to use
     * @return the tensor along dimension based on the index and dimensions
     * from the master array.
     */
    @Override
    public synchronized INDArray getTad(int idx, int... dimensions) {
        return arr.tensorAlongDimension(idx,dimensions);
    }
}
