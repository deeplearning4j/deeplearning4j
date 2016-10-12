package org.nd4j.parameterserver.parameteraveraging;

import lombok.Data;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.deser.std.AtomicBooleanDeserializer;
import org.nd4j.shade.jackson.databind.ser.std.StdJdkSerializers;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Parameter averaging
 * listener
 * @author Adam Gibson
 */
@Data
public class ParameterAveragingListener implements NDArrayCallback,NDArrayHolder {
    private INDArray arr;
    private AtomicInteger totalN = new AtomicInteger(0);
    private boolean master;

    /**
     * Length of the listener
     * @param length the length of the array
     */
    public ParameterAveragingListener(int length) {
        this.arr = Nd4j.zeros(length);
    }


    /**
     * Setup an ndarray
     *
     * @param arr
     */
    @Override
    public synchronized void onNDArray(INDArray arr) {
        this.arr.addi(arr.reshape(1,arr.length()));
        totalN.incrementAndGet();
    }

    /**
     * Do a final divide for averaging
     */
    public synchronized void finish() {
        this.arr.divi(totalN);
    }

    /**
     * The number of updates
     * that have been sent to this older.
     *
     * @return
     */
    @Override
    public int totalUpdates() {
        return totalN.get();
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
}
