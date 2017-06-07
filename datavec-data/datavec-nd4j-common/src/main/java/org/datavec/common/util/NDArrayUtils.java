/*
 *  * Copyright 2017 Skymind, Inc.
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
 */

package org.datavec.common.util;

import org.datavec.api.transform.metadata.NDArrayMetaData;
import org.datavec.api.writable.Writable;
import org.datavec.common.data.NDArrayWritable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Utility methods for NDArrayWritable operations
 *
 * @author Alex Black
 */
public class NDArrayUtils {

    private NDArrayUtils() {
    }

    /**
     * hashCode method, taken from Java 1.8 Double.hashCode(double) method
     *
     * @param value Double value to hash
     * @return Hash code for the double value
     */
    public static int hashCode(double value) {
        long bits = Double.doubleToLongBits(value);
        return (int) (bits ^ (bits >>> 32));
    }

    /**
     * Determine if the given writable value is valid for the given NDArrayMetaData
     *
     * @param meta     Meta data
     * @param writable Writable to check
     * @return True if valid, false otherwise
     */
    public static boolean isValid(NDArrayMetaData meta, Writable writable) {
        if (!(writable instanceof NDArrayWritable)) {
            return false;
        }
        INDArray arr = ((NDArrayWritable) writable).get();
        if (arr == null) {
            return false;
        }
        int[] shape = meta.getShape();
        if (meta.isAllowVarLength()) {
            for (int i = 0; i < shape.length; i++) {
                if (shape[i] < 0) {
                    continue;
                }
                if (shape[i] != arr.size(i)) {
                    return false;
                }
            }
            return true;
        } else {
            return Arrays.equals(shape, arr.shape());
        }
    }

    /**
     * Determine if the given Object is valid for the given NDArrayMetaData
     *
     * @param meta  Meta data
     * @param input Object to check
     * @return True if valid, false otherwise
     */
    public static boolean isValid(NDArrayMetaData meta, Object input) {
        if(input == null) {
            return false;
        } else if (input instanceof Writable) {
            return isValid(meta, (Writable) input);
        } else if (input instanceof INDArray) {
            return isValid(meta, new NDArrayWritable((INDArray) input));
        } else {
            throw new UnsupportedOperationException("Unknown object type: " + input.getClass());
        }
    }

}
