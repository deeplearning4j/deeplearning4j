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

package org.nd4j.linalg.util;

import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.Arrays;

/**
 * Linear algebra exceptions
 *
 * @author Adam Gibson
 */
public class LinAlgExceptions {
    /**
     * Asserts both arrays be the same length
     * @param x
     * @param z
     */
    public static void assertSameLength(INDArray x, INDArray z) {
        val lengthX = x.length();
        val lengthZ = z.length();
        if (lengthX != lengthZ && lengthX != 1 && lengthZ != 1)
            throw new IllegalStateException("Mis matched lengths: [" + x.length() + "] != [" + z.length() + "] - " +
                    "Array 1 shape: " + Arrays.toString(x.shape()) + ", array 2 shape: " + Arrays.toString(z.shape()));
    }

    public static void assertSameLength(INDArray x, INDArray y, INDArray z) {
        val lengthX = x.length();
        val lengthY = y.length();
        val lengthZ = z.length();

        if (lengthX != lengthY && lengthX != lengthZ && lengthX != 1 && lengthY != 1 && lengthZ != 1)
            throw new IllegalStateException("Mis matched lengths: [" + lengthX + "] != [" + lengthY + "] != [" + lengthZ + "] - " +
                    "Array 1 shape: " + Arrays.toString(x.shape()) + ", array 2 shape: " + Arrays.toString(y.shape()) + ", array 3 shape: " + Arrays.toString(z.shape()));
    }

    public static void assertSameShape(INDArray n, INDArray n2) {
        if (!Shape.shapeEquals(n.shape(), n2.shape()))
            throw new IllegalStateException("Mis matched shapes: " + Arrays.toString(n.shape()) + ", "
                    + Arrays.toString(n2.shape()));
    }

    public static void assertRows(INDArray n, INDArray n2) {
        if (n.rows() != n2.rows())
            throw new IllegalStateException("Mis matched rows: " + n.rows() + " != " + n2.rows());
    }


    public static void assertVector(INDArray... arr) {
        for (INDArray a1 : arr)
            assertVector(a1);
    }

    public static void assertMatrix(INDArray... arr) {
        for (INDArray a1 : arr)
            assertMatrix(a1);
    }

    public static void assertVector(INDArray arr) {
        if (!arr.isVector())
            throw new IllegalArgumentException("Array must be a vector. Array has shape: " + Arrays.toString(arr.shape()));
    }

    public static void assertMatrix(INDArray arr) {
        if (arr.shape().length > 2)
            throw new IllegalArgumentException("Array must be a matrix. Array has shape: " + Arrays.toString(arr.shape()));
    }



    /**
     * Asserts matrix multiply rules (columns of left == rows of right or rows of left == columns of right)
     *
     * @param nd1 the left ndarray
     * @param nd2 the right ndarray
     */
    public static void assertMultiplies(INDArray nd1, INDArray nd2) {
        if (nd1.rank() == 2 && nd2.rank() == 2 && nd1.columns() == nd2.rows()) {
            return;
        }

        // 1D edge case
        if (nd1.rank() == 2 && nd2.rank() == 1 && nd1.columns() == nd2.length())
            return;

        throw new ND4JIllegalStateException("Cannot execute matrix multiplication: " + Arrays.toString(nd1.shape())
                        + "x" + Arrays.toString(nd2.shape())
                        + (nd1.rank() != 2 || nd2.rank() != 2 ? ": inputs are not matrices"
                                        : ": Column of left array " + nd1.columns() + " != rows of right "
                                                        + nd2.rows()));
    }


    public static void assertColumns(INDArray n, INDArray n2) {
        if (n.columns() != n2.columns())
            throw new IllegalStateException("Mis matched columns: " + n.columns() + " != " + n2.columns());
    }

    public static void assertValidNum(INDArray n) {
        INDArray linear = n.linearView();
        for (int i = 0; i < linear.length(); i++) {
            double d = linear.getDouble(i);
            if (Double.isNaN(d) || Double.isInfinite(d))
                throw new IllegalStateException("Found infinite or nan");

        }
    }

}
