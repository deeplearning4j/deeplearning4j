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

package org.deeplearning4j.spark.util;

import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Dl4j <----> MLLib
 *
 * @author Adam Gibson
 */
public class MLLibUtil {

    /**
     * Convert a matrix to an ndarray.
     * Note that the matrix will be con
     * @param arr the array
     * @return an mllib vector
     */
    public static INDArray fromMatrix(Matrix arr) {
        return Nd4j.create(arr.toArray(), new int[]{arr.numRows(), arr.numCols()});
    }

    /**
     * Convert an ndarray to a vector
     * @param arr the array
     * @return an mllib vector
     */
    public static INDArray fromVector(Vector arr) {
        return Nd4j.create(Nd4j.createBuffer(arr.toArray()));
    }


    /**
     * Convert an ndarray to a matrix.
     * Note that the matrix will be con
     * @param arr the array
     * @return an mllib vector
     */
    public static Matrix toMatrix(INDArray arr) {
        if(!arr.isMatrix()) {
            throw new IllegalArgumentException("passed in array must be a matrix");
        }
        return Matrices.dense(arr.rows(), arr.columns(), arr.data().asDouble());
    }

    /**
     * Convert an ndarray to a vector
     * @param arr the array
     * @return an mllib vector
     */
    public static Vector toVector(INDArray arr) {
        if(!arr.isVector()) {
            throw new IllegalArgumentException("passed in array must be a vector");
        }
        double[] ret = new double[arr.length()];
        for(int i = 0; i < arr.length(); i++) {
            ret[i] = arr.getDouble(i);
        }

        return Vectors.dense(ret);
    }
}
