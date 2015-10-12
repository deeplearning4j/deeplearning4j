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
 *
 */

package org.nd4j.linalg.util;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Basic INDArray ops
 *
 * @author Adam Gibson
 */
public class NDArrayUtil {





    public static INDArray exp(INDArray toExp) {
        return expi(toExp.dup());
    }

    /**
     * Returns an exponential version of this ndarray
     *
     * @param toExp the INDArray to convert
     * @return the converted ndarray
     */
    public static INDArray expi(INDArray toExp) {
        INDArray flattened = toExp.ravel();
        for (int i = 0; i < flattened.length(); i++)
            flattened.put(i, Nd4j.scalar(Math.exp((double) flattened.getScalar(i).element())));
        return flattened.reshape(toExp.shape());
    }

    /**
     * Center an array
     *
     * @param arr   the arr to center
     * @param shape the shape of the array
     * @return the center portion of the array based on the
     * specified shape
     */
    public static INDArray center(INDArray arr, int[] shape) {
        if (arr.length() < ArrayUtil.prod(shape))
            return arr;
        for (int i = 0; i < shape.length; i++)
            if (shape[i] < 1)
                shape[i] = 1;

        INDArray shapeMatrix = ArrayUtil.toNDArray(shape);
        INDArray currShape = ArrayUtil.toNDArray(arr.shape());

        INDArray startIndex = Transforms.floor(currShape.sub(shapeMatrix).divi(Nd4j.scalar(2)));
        INDArray endIndex = startIndex.add(shapeMatrix);
        INDArrayIndex[] indexes = Indices.createFromStartAndEnd(startIndex, endIndex);

        if (shapeMatrix.length() > 1)
            return arr.get(indexes);


        else {
            INDArray ret = Nd4j.create(new int[]{(int) shapeMatrix.getDouble(0)});
            int start = (int) startIndex.getDouble(0);
            int end = (int) endIndex.getDouble(0);
            int count = 0;
            for (int i = start; i < end; i++) {
                ret.putScalar(count++, arr.getDouble(i));
            }

            return ret;
        }
    }
}
