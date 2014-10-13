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
    public double accumulate(INDArray arr, int i, double soFar) {
        if(i < arr.length() - 1)
            return soFar + arr.getDouble(i);
        else {
            soFar +=  arr.getDouble(i);
            soFar /= arr.length();
            // Compute initial estimate using definitional formula
            double xbar = soFar;

            // Compute correction factor in second pass
            double correction = 0;
            for (int j = 0; j < arr.length(); j++) {
                correction += arr.getDouble(j) - xbar;
            }

            return  (xbar + (correction/arr.length()));

        }

    }
}
