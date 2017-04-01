package org.nd4j.aeron.ndarrayholder;

import lombok.NoArgsConstructor;
import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * An in meory ndarray holder
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class InMemoryNDArrayHolder implements NDArrayHolder {

    private AtomicReference<INDArray> arr = new AtomicReference<>();
    private AtomicInteger totalUpdates = new AtomicInteger(0);


    public InMemoryNDArrayHolder(int[] shape) {
        setArray(Nd4j.zeros(shape));
    }


    public InMemoryNDArrayHolder(INDArray arr) {
        setArray(arr);
    }


    /**
     * Set the ndarray
     *
     * @param arr the ndarray for this holder
     *            to use
     */
    @Override
    public void setArray(INDArray arr) {
        if (this.arr.get() == null)
            this.arr.set(arr);
    }

    /**
     * The number of updates
     * that have been sent to this older.
     *
     * @return
     */
    @Override
    public int totalUpdates() {
        return totalUpdates.get();
    }

    /**
     * Retrieve an ndarray
     *
     * @return
     */
    @Override
    public INDArray get() {
        return arr.get();
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
    public INDArray getTad(int idx, int... dimensions) {
        return arr.get().tensorAlongDimension(idx, dimensions);
    }
}
