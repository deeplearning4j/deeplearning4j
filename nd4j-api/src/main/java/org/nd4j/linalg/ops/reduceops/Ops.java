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

package org.nd4j.linalg.ops.reduceops;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.reduceops.scalarops.*;

/**
 * Ops for scalar, dimension, and matrix
 *
 * @author Adam Gibson
 */
public class Ops {


    public static double std(INDArray arr) {
        return new StandardDeviation().apply(arr);
    }

    public static double norm1(INDArray arr) {
        return new Norm1().apply(arr);
    }

    public static double norm2(INDArray arr) {
        return new Norm2().apply(arr);
    }

    public static double normmax(INDArray arr) {
        return new NormMax().apply(arr);
    }

    public static double max(INDArray arr) {
        return new Max().apply(arr);
    }

    public static double min(INDArray arr) {
        return new Min().apply(arr);
    }

    public static double mean(INDArray arr) {
        return new Mean().apply(arr);
    }

    public static double sum(INDArray arr) {
        return new Sum().apply(arr);
    }

    public static double var(INDArray arr) {
        return new Variance().apply(arr);
    }

    public static double prod(INDArray arr) {
        return new Prod().apply(arr);
    }

    public static enum ScalarOp {
        SUM,
        MEAN,
        PROD,
        MAX,
        MIN,
        ARG_MAX,
        ARG_MIN,
        NORM_2,
        NORM_1,
        NORM_MAX
    }

    public static enum DimensionOp {
        SUM,
        MEAN,
        PROD,
        MAX,
        MIN,
        ARG_MIN,
        NORM_2,
        NORM_1,
        NORM_MAX,
        FFT
    }

    public static enum MatrixOp {
        COLUMN_MIN,
        COLUMN_MAX,
        COLUMN_SUM,
        COLUMN_MEAN,
        ROW_MIN,
        ROW_MAX,
        ROW_SUM,
        ROW_MEAN
    }
}
