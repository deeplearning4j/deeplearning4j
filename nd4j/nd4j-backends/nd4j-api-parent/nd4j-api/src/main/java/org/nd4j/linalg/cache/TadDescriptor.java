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

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;

/**
 * This is utility class, made to compare TADs for caching purposes.
 *
 * Idea: for any given INDArray with any specific shape,
 * TAD for specific dimension will always be the same.
 * So it can be reused as much as we want.
 *
 * Of note here is that when used as a key,
 * we preserve immutability of the shape buffer
 * in the ndarray by copying the values with
 * {@link TadDescriptor#dataBufferToArray(DataBuffer)}
 *
 *
 * @author raver119@gmail.com
 */
@Slf4j
@Data
public class TadDescriptor {
    private int dimensionLength;
    private int[] dimension;
    private long[] shape;

    /**
     * Pass in an ndarray to get the databuffer
     * and the appropriate dimensions
     * @param array the array to pass in
     *              to get the shape info from
     * @param dimension the dimensions for the TAD
     */
    public TadDescriptor(INDArray array, int[] dimension) {
        this.dimensionLength = dimension == null ? 0 : dimension.length;
        this.dimension = dimension;

        // TODO: change this to fill shapeInfo
        this.shape = dataBufferToArray(array.shapeInfoDataBuffer());
    }


    /**
     * Obtain the values from the shape buffer
     * for the array
     * @param buffer the buffer to get the values from
     * @return the int array version of this data buffer
     */
    public static long[] dataBufferToArray(DataBuffer buffer) {
        int rank = buffer.getInt(0);
        val ret = new long[Shape.shapeInfoLength(rank)];
        ret[0] = rank;
        for (int e = 1; e < Shape.shapeInfoLength(rank); e++) {
            ret[e] = buffer.getInt(e);
        }

        return ret;
    }

}
