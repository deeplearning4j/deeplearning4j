package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Adam Gibson
 */
public class Mean extends BaseScalarOp {

    public Mean() {
        super(0);
    }

    @Override
    public float accumulate(INDArray arr, int i, float soFar) {
        if(i < arr.length() - 1)
            return soFar + arr.get(i);
        else {
            soFar +=  arr.get(i);
            soFar /= arr.length();
            // Compute initial estimate using definitional formula
            double xbar = soFar;

            // Compute correction factor in second pass
            double correction = 0;
            for (int j = 0; j < arr.length(); j++) {
                correction += arr.get(j) - xbar;
            }

            return (float)  (xbar + (correction/arr.length()));

        }

    }
}
