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

package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.Serializable;

/**
 * Represents a cache linear index lookup
 * @author Adam Gibson
 */
public class LinearIndexLookup implements Serializable {
    private char ordering;
    private long[][] indexes;
    private long[] shape;
    private boolean[] exists;
    private long numIndexes;

    /**
     *
     * @param shape the shape of the linear index
     * @param ordering the ordering of the linear index
     */
    public LinearIndexLookup(int[] shape, char ordering) {
        this(ArrayUtil.toLongArray(shape), ordering);
    }


    public LinearIndexLookup(long[] shape, char ordering) {
        this.shape = shape;
        this.ordering = ordering;
        numIndexes = ArrayUtil.prodLong(shape);

        // FIMXE: long!
        indexes = new long[(int) numIndexes][shape.length];
        exists = new boolean[(int) numIndexes];
    }

    /**
     * Give back a sub
     * wrt the given linear index
     * @param index the index
     * @return the sub for the given index
     */
    public long[] lookup(int index) {
        if (exists[index]) {
            return indexes[index];
        } else {
            exists[index] = true;
            indexes[index] = ordering == 'c' ? Shape.ind2subC(shape, index, numIndexes)
                            : Shape.ind2sub(shape, index, numIndexes);
            return indexes[index];
        }
    }


}
