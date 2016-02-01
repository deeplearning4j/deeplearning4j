/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.spark.impl.common;

import org.apache.spark.AccumulatorParam;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Accumulator for addition of parameters
 * @author Adam Gibson
 */
public class ParamAccumulator implements AccumulatorParam<INDArray> {
    @Override
    public INDArray addAccumulator(INDArray indArray, INDArray t1) {
        if(indArray == null) return t1;
        else if(t1 == null) return indArray;
        if(indArray.length() != t1.length())
            throw new IllegalStateException("Invalid accumulation of parameters: first length = " + indArray.length() + ", second = " + t1.length());
        return indArray.add(t1);
    }

    @Override
    public INDArray addInPlace(INDArray indArray, INDArray r1) {
        if(indArray == null) return r1;
        else if(r1 == null) return indArray;
        return indArray.addi(r1);
    }

    @Override
    public INDArray zero(INDArray indArray) {
        return indArray;
    }
}
