/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * This is utility class, made to compare java arrays for caching purposes.
 *
 * @author raver119@gmail.com
 */
public class ArrayDescriptor {
    int[] intArray = null;
    float[] floatArray = null;
    double[] doubleArray = null;
    long[] longArray = null;

    private enum DTYPE {
        INT, FLOAT, DOUBLE, LONG
    }

    private DTYPE dtype;

    private DataType bufferType;

    public ArrayDescriptor(int[] array) {
        this.intArray = array;
        this.dtype = DTYPE.INT;
        this.bufferType = Nd4j.dataType();
    }

    public ArrayDescriptor(float[] array) {
        this.floatArray = array;
        this.dtype = DTYPE.FLOAT;
        this.bufferType = Nd4j.dataType();
    }

    public ArrayDescriptor(double[] array) {
        this.doubleArray = array;
        this.dtype = DTYPE.DOUBLE;
        this.bufferType = Nd4j.dataType();
    }

    public ArrayDescriptor(long[] array) {
        this.longArray = array;
        this.dtype = DTYPE.LONG;
        this.bufferType = Nd4j.dataType();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        ArrayDescriptor that = (ArrayDescriptor) o;

        if (this.bufferType != that.bufferType)
            return false;

        if (this.dtype != that.dtype)
            return false;

        if (intArray != null && that.intArray != null) {
            return Arrays.equals(intArray, that.intArray);
        } else if (floatArray != null && that.floatArray != null) {
            return Arrays.equals(floatArray, that.floatArray);
        } else if (doubleArray != null && that.doubleArray != null) {
            return Arrays.equals(doubleArray, that.doubleArray);
        } else if (longArray != null && that.longArray != null) {
            return Arrays.equals(longArray, that.longArray);
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        if (intArray != null) {
            return intArray.getClass().hashCode() + 31 * Arrays.hashCode(intArray) + 31 * bufferType.ordinal();
        } else if (floatArray != null) {
            return floatArray.getClass().hashCode() + 31 * Arrays.hashCode(floatArray) + 31 * bufferType.ordinal();
        } else if (doubleArray != null) {
            return doubleArray.getClass().hashCode() + 31 * Arrays.hashCode(doubleArray) + 31 * bufferType.ordinal();
        } else if (longArray != null) {
            return longArray.getClass().hashCode() + 31 * Arrays.hashCode(longArray) + 31 * bufferType.ordinal();
        } else {
            return 0;
        }
    }
}
