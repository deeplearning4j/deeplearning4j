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

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Overall norm2 of an ndarray
 *
 * @author Adam Gibson
 */
public class Norm2 extends BaseScalarOp {
    public Norm2() {
        super(0);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        double ret = soFar + FastMath.pow(arr.getDouble(i), 2);
        if (i == arr.length() - 1)
            return FastMath.sqrt(ret);
        return ret;
    }
}
