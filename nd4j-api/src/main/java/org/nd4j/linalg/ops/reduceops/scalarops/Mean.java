/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class Mean extends BaseScalarOp {

    public Mean() {
        super(0);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        if (i < arr.length() - 1)
            return soFar + arr.getDouble(i);
        else {
            soFar += arr.getDouble(i);
            soFar /= arr.length();
            // Compute initial estimate using definitional formula
            double xbar = soFar;

            // Compute correction factor in second pass
            double correction = 0;
            for (int j = 0; j < arr.length(); j++) {
                correction += arr.getDouble(j) - xbar;
            }

            return (xbar + (correction / arr.length()));

        }

    }
}
