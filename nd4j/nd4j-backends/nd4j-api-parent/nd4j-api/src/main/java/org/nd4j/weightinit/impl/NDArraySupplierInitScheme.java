package org.nd4j.weightinit.impl;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.weightinit.WeightInit;
import org.nd4j.weightinit.WeightInitScheme;

/**
 *
 */
@AllArgsConstructor
public class NDArraySupplierInitScheme implements WeightInitScheme {

    private NDArraySupplier supplier;

    /**
     * A simple {@link INDArray facade}
     */
    public  interface NDArraySupplier {
        /**
         * An array proxy method.
          * @return
         */
        INDArray getArr();
    }

    @Override
    public INDArray create(long[] shape, INDArray paramsView) {
        return supplier.getArr();
    }

    @Override
    public INDArray create(long[] shape) {
        return supplier.getArr();
    }

    @Override
    public char order() {
        return 'f';
    }

    @Override
    public WeightInit type() {
        return WeightInit.SUPPLIED;
    }
}
