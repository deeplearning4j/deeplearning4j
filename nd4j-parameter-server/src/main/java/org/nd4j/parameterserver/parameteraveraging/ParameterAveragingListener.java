package org.nd4j.parameterserver.parameteraveraging;

import lombok.Data;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Parameter averaging
 * listener
 * @author Adam Gibson
 */
@Data
public class ParameterAveragingListener implements NDArrayCallback {
    private INDArray arr;
    private double totalN;
    private boolean master;

    public ParameterAveragingListener(int length) {
        this.arr = Nd4j.zeros(length);
        this.totalN = 0.0;
    }


    /**
     * Setup an ndarray
     *
     * @param arr
     */
    @Override
    public void onNDArray(INDArray arr) {
        this.arr.addi(arr.reshape(1,arr.length()));
        totalN++;
    }

    /**
     * Do a final divide for averaging
     */
    public void finish() {
        this.arr.divi(totalN);
    }

}
