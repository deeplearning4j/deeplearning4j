/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.api.shape;

import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class PaddingUtils {

    // ... [Previous methods remain unchanged] ...

    /**
     * Calculate padded strides
     */
    public static long[] calculatePaddedStrides(long[] paddedShape, char ordering) {
        return ordering == 'c' ? ArrayUtil.calcStrides(paddedShape) : ArrayUtil.calcStridesFortran(paddedShape);
    }

    /**
     * Calculate padded allocation size and offset
     */
    public static class PaddedAllocationResult {
        public final long paddedAllocSize;
        public final long offset;
        public final long[] paddedStrides;

        public PaddedAllocationResult(long paddedAllocSize, long offset, long[] paddedStrides) {
            this.paddedAllocSize = paddedAllocSize;
            this.offset = offset;
            this.paddedStrides = paddedStrides;
        }
    }

    public static PaddedAllocationResult calculatePaddedAllocation(long[] paddedShape, long[] paddingOffsets, char ordering, boolean empty, long ews) {
        long[] paddedStrides = calculatePaddedStrides(paddedShape, ordering);
        long paddedAllocSize = ordering == 'c' ? paddedShape[0] * paddedStrides[0] : paddedShape[paddedShape.length - 1] * paddedStrides[paddedShape.length - 1];

        long offset = (empty || ews == 1 || paddingOffsets == null || paddingOffsets.length == 0) ? 0 : ArrayUtil.calcOffset(paddedShape, paddingOffsets, paddedStrides);

        return new PaddedAllocationResult(paddedAllocSize, offset, paddedStrides);
    }

    /**
     * Perform all padding operations and return the results
     */
    public static class PaddingResult {
        public final DataBuffer data;
        public final long extras;
        public final boolean isView;
        public final long[] paddedShape;
        public final long[] paddedStrides;

        public PaddingResult(DataBuffer data, long extras, boolean isView, long[] paddedShape, long[] paddedStrides) {
            this.data = data;
            this.extras = extras;
            this.isView = isView;
            this.paddedShape = paddedShape;
            this.paddedStrides = paddedStrides;
        }
    }

    /**
     * Validate padding inputs
     */
    private static void validatePaddingInputs(long[] shape, long[] paddings, long[] paddingOffsets) {
        if (paddings == null || paddings.length != shape.length) {
            throw new IllegalArgumentException("The length of Padding should be equal to the length of Shape");
        }
        if (paddingOffsets != null && paddingOffsets.length != 0 && paddingOffsets.length != shape.length) {
            throw new IllegalArgumentException("If PaddingOffsets is not empty or zero length then its length should match the length of Paddings");
        }
    }

    /**
     * Calculate padded shape and check for empty array
     */
    private static class PaddedShapeResult {
        long[] paddedShape;
        boolean isEmpty;
        long ews;

        PaddedShapeResult(long[] paddedShape, boolean isEmpty, long ews) {
            this.paddedShape = paddedShape;
            this.isEmpty = isEmpty;
            this.ews = ews;
        }
    }

    private static PaddedShapeResult calculatePaddedShape(long[] shape, long[] paddings, long[] paddingOffsets) {
        int rank = shape.length;
        long[] paddedShape = new long[rank];
        boolean empty = false;
        long ews = 1;

        for (int i = 0; i < rank; i++) {
            paddedShape[i] = shape[i] + paddings[i];
            if (paddings[i] != 0) ews = 0;
            if (shape[i] == 0) empty = true;
            if (paddingOffsets != null && paddingOffsets[i] > paddings[i]) {
                throw new IllegalArgumentException("PaddingOffsets elements should not be greater than Paddings elements");
            }
        }

        return new PaddedShapeResult(paddedShape, empty, ews);
    }


    /**
     * Compose options for the padded array
     */
    public static long composePaddedArrayOptions(DataType type, boolean isEmpty, long ews, long offset) {
        List<Long> flags = new ArrayList<>();
        flags.add(ArrayOptionsHelper.composeTypicalChecks(type));

        if (isEmpty) {
            flags.add(ArrayOptionsHelper.ATYPE_EMPTY_BIT);
        }
        if (ews != 1) {
            flags.add(ArrayOptionsHelper.HAS_PADDED_BUFFER);
        }
        if (offset > 0) {
            flags.add(ArrayOptionsHelper.IS_VIEW);
        }

        return ArrayOptionsHelper.composeOptions(flags);
    }


    /**
     * Create data buffer for padded array
     */
    public static DataBuffer createPaddedBuffer(DataType type, long paddedAllocSize, MemoryWorkspace workspace) {
        DataBuffer buffer = Nd4j.createBuffer(type, paddedAllocSize, false, workspace);
        return  buffer;
    }


    public static PaddingResult performPadding(long[] shape, long[] paddings, long[] paddingOffsets, DataType type, char ordering, MemoryWorkspace workspace) {
        validatePaddingInputs(shape, paddings, paddingOffsets);

        PaddedShapeResult paddedShapeResult = calculatePaddedShape(shape, paddings, paddingOffsets);
        PaddedAllocationResult paddedAllocationResult = calculatePaddedAllocation(paddedShapeResult.paddedShape, paddingOffsets, ordering, paddedShapeResult.isEmpty, paddedShapeResult.ews);

        DataBuffer data = createPaddedBuffer(type, paddedAllocationResult.paddedAllocSize, workspace);
        long extras = composePaddedArrayOptions(type, paddedShapeResult.isEmpty, paddedShapeResult.ews, paddedAllocationResult.offset);
        boolean isView = paddedAllocationResult.offset > 0;

        return new PaddingResult(data, extras, isView, paddedShapeResult.paddedShape, paddedAllocationResult.paddedStrides);
    }
}