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

package org.nd4j.linalg.indexing;

import com.google.common.primitives.Longs;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LongUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 *
 * Sets up the strides, shape
 * , and offsets
 * for an indexing operation
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
public class ShapeOffsetResolution implements Serializable {

    private INDArray arr;

    private int[] fixed, prependAxis;
    private long[] offsets;
    private long[] shapes, strides;

    private long offset = -1;

    /**
     * Specify the array to use for resolution
     * @param arr the array to use
     *            for resolution
     */
    public ShapeOffsetResolution(INDArray arr) {
        this.arr = arr;
    }


    /**
     *
     * @param indexes
     * @return
     */
    public boolean tryShortCircuit(INDArrayIndex... indexes) {
        int pointIndex = 0;
        int interval = 0;
        int newAxis = 0;
        int numAll = 0;
        int numSpecified = 0;
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] instanceof PointIndex) {
                pointIndex++;
            }
            if (indexes[i] instanceof SpecifiedIndex)
                numSpecified++;
            else if (indexes[i] instanceof IntervalIndex && !(indexes[i] instanceof NDArrayIndexAll))
                interval++;
            else if (indexes[i] instanceof NewAxis)
                newAxis++;
            else if (indexes[i] instanceof NDArrayIndexAll)
                numAll++;

        }



        if (arr.isVector()) {
            //return the whole vector
            if (indexes[0] instanceof NDArrayIndexAll && indexes.length == 1) {

                offset = 0;
                this.shapes = arr.shape();
                this.strides = arr.stride();
                this.offsets = new long[arr.rank()];
                return true;
            } else if (indexes[0] instanceof PointIndex && indexes[1] instanceof NDArrayIndexAll) {
                this.shapes = new long[2];
                this.strides = new long[2];
                for (int i = 0; i < 2; i++) {
                    shapes[i] = 1;
                    strides[i] = arr.stride(i);
                }

                this.offsets = new long[arr.rank()];
                if(arr.isRowVector())
                    this.offset = indexes[0].offset() * strides[1];
                else {
                    this.offset = indexes[0].offset() * strides[0];
                }
                return true;
            }
            if (indexes[0] instanceof PointIndex && indexes.length == 1) {
                this.shapes = new long[2];
                this.strides = new long[2];
                for (int i = 0; i < 2; i++) {
                    shapes[i] = 1;
                    strides[i] = arr.stride(i);
                }

                if(arr.isRowVector())
                    this.offset = indexes[0].offset() * strides[1];
                else {
                    this.offset = indexes[0].offset() * strides[0];
                }

                return true;
            }
            //point or interval is possible
            if (arr.isRowVector()) {
                if (indexes[0] instanceof PointIndex) {
                    if (indexes.length > 1 && indexes[1] instanceof IntervalIndex) {
                        offset = indexes[1].offset();
                        this.shapes = new long[2];
                        shapes[0] = 1;
                        shapes[1] = indexes[1].length();
                        this.strides = new long[2];
                        strides[0] = 0;
                        strides[1] = indexes[1].stride();
                        this.offsets = new long[2];
                        return true;
                    }
                } else if (indexes[0] instanceof IntervalIndex) {
                    //allow through
                } else {
                    return false;

                }
            } else {
                if (indexes.length > 1 && indexes[1] instanceof PointIndex) {
                    if (indexes[0] instanceof IntervalIndex) {
                        offset = indexes[0].offset();
                        this.shapes = new long[2];
                        shapes[1] = 1;
                        shapes[0] = indexes[1].length();
                        this.strides = new long[2];
                        strides[1] = 0;
                        strides[0] = indexes[1].stride();
                        this.offsets = new long[2];
                        return true;
                    }
                } else if (indexes[0] instanceof IntervalIndex) {
                    //allow through
                } else {
                    return false;
                }
            }
        }

        //all and specified only
        if (numSpecified > 0 && interval < 1 && newAxis < 1 && numAll > 0 && pointIndex < 1 && arr.rank() == 2) {
            shapes = new long[arr.rank()];
            strides = new long[arr.rank()];
            offsets = new long[arr.rank()];
            offset = 0;
            boolean allSpecified = true;
            for (int i = 0; i < 2; i++) {
                allSpecified = allSpecified && indexes[i] instanceof SpecifiedIndex;
            }
            for (int i = 0; i < arr.rank(); i++) {
                if (indexes[i] instanceof SpecifiedIndex) {
                    SpecifiedIndex specifiedIndex = (SpecifiedIndex) indexes[i];
                    if (specifiedIndex.getIndexes().length >= arr.rank())
                        return false;
                    shapes[i] = indexes[i].length();
                    offsets[i] = indexes[i].offset();
                    if (!allSpecified || i == 0 && allSpecified)
                        offset = offsets[i] * arr.stride(i);
                    if (indexes[i].length() != 1) {
                        strides[i] = arr.stride(i) * specifiedIndex.getIndexes()[i];
                    } else
                        strides[i] = 1;
                } else if (indexes[i] instanceof NDArrayIndexAll) {
                    shapes[i] = arr.size(i);
                    strides[i] = arr.tensorAlongDimension(0, i).elementWiseStride();
                } else
                    throw new IllegalArgumentException("Illegal opType of index " + indexes[i].getClass().getName());
            }


            return true;

        }

        //specific easy case
        if (numSpecified < 1 && interval < 1 && newAxis < 1 && pointIndex > 0 && numAll > 0) {
            int minDimensions = Math.max(arr.rank() - pointIndex, 2);
            long[] shape = new long[minDimensions];
            Arrays.fill(shape, 1);
            long[] stride = new long[minDimensions];
            Arrays.fill(stride, arr.elementStride());
            long[] offsets = new long[minDimensions];
            long offset = 0;
            //used for filling in elements of the actual shape stride and offsets
            int currIndex = 0;
            //used for array item access
            int arrIndex = 0;
            for (int i = 0; i < indexes.length; i++) {
                if (indexes[i] instanceof NDArrayIndexAll) {
                    shape[currIndex] = arr.size(arrIndex);
                    stride[currIndex] = arr.stride(arrIndex);
                    currIndex++;
                    arrIndex++;
                }
                //point index
                else {
                    offset += indexes[i].offset() * arr.stride(i);
                    arrIndex++;

                }
            }

            if (arr.isMatrix() && indexes[0] instanceof PointIndex) {
                shape = ArrayUtil.reverseCopy(shape);
                stride = ArrayUtil.reverseCopy(stride);
            } else if (arr.isMatrix() && indexes[0] instanceof PointIndex && indexes[1] instanceof IntervalIndex) {
                shape = new long[2];
                shape[0] = 1;
                IntervalIndex idx = (IntervalIndex) indexes[1];
                shape[1] = idx.length();

            }

            //keep same strides
            this.strides = stride;
            this.shapes = shape;
            this.offsets = offsets;
            this.offset = offset;
            return true;

        }

        //intervals and all
        else if (numSpecified < 1 && interval > 0 && newAxis < 1 && pointIndex < 1 && numAll > 0) {
            int minDimensions = Math.max(arr.rank(), 2);
            long[] shape = new long[minDimensions];
            Arrays.fill(shape, 1);
            long[] stride = new long[minDimensions];
            Arrays.fill(stride, arr.elementStride());
            long[] offsets = new long[minDimensions];

            for (int i = 0; i < shape.length; i++) {
                if (indexes[i] instanceof NDArrayIndexAll) {
                    shape[i] = arr.size(i);
                    stride[i] = arr.stride(i);
                    offsets[i] = indexes[i].offset();
                } else if (indexes[i] instanceof IntervalIndex) {
                    shape[i] = indexes[i].length();
                    stride[i] = indexes[i].stride() * arr.stride(i);
                    offsets[i] = indexes[i].offset();
                }
            }

            this.shapes = shape;
            this.strides = stride;
            this.offsets = offsets;
            this.offset = 0;
            for (int i = 0; i < indexes.length; i++) {
                offset += offsets[i] * (stride[i] / indexes[i].stride());
            }
            return true;
        }

        //all and newaxis
        else if (numSpecified < 1 && interval < 1 && newAxis > 0 && pointIndex < 1 && numAll > 0) {
            int minDimensions = Math.max(arr.rank(), 2) + newAxis;
            //new axis dimensions + all
            long[] shape = new long[minDimensions];
            Arrays.fill(shape, 1);
            long[] stride = new long[minDimensions];
            Arrays.fill(stride, arr.elementStride());
            long[] offsets = new long[minDimensions];
            int prependNewAxes = 0;
            boolean allFirst = false;
            int shapeAxis = 0;
            int allEncountered = 0;
            for (int i = 0; i < minDimensions; i++) {
                if (i >= (indexes.length)) {
                    shape[i] = arr.size(allEncountered);
                    stride[i] = arr.stride(allEncountered);
                    allEncountered++;
                } else if (!(indexes[i] instanceof NewAxis) && indexes[i] instanceof NDArrayIndexAll) {
                    shape[allEncountered] = arr.size(allEncountered);
                    stride[allEncountered] = arr.stride(allEncountered);
                    allEncountered++;
                }

            }



            this.shapes = shape;
            this.strides = stride;
            this.offsets = offsets;
            for (int i = 0; i < indexes.length; i++) {
                offset += offsets[i] * (stride[i] / indexes[i].stride());
            }

            return true;
        }


        return false;
    }

    /**
     * Based on the passed in array
     * compute the shape,offsets, and strides
     * for the given indexes
     * @param indexes the indexes
     *                to compute this based on
     *
     */
    public void exec(INDArrayIndex... indexes) {
        val shape = arr.shape();

        if (arr.isSparse()) {
            resolveFixedDimensionsCOO(indexes);
        }

        // Check that given point indexes are not out of bounds
        for (int i = 0; i < indexes.length; i++) {
            INDArrayIndex idx = indexes[i];
            // On vectors, the first dimension can be ignored when indexing them with a single point index
            if (idx instanceof PointIndex && (arr.isVector() && indexes.length == 1 ? idx.current() >= shape[i + 1]
                    : idx.current() >= shape[i])) {
                throw new IllegalArgumentException(
                        "INDArrayIndex[" + i + "] is out of bounds (value: " + idx.current() + ")");
            }
        }

        indexes = NDArrayIndex.resolve(arr.shapeInfoDataBuffer(), indexes);
        if (tryShortCircuit(indexes)) {
            return;
        }


        int numIntervals = 0;
        //number of new axes dimensions to prepend to the beginning
        int newAxesPrepend = 0;
        //whether we have encountered an all so far
        boolean encounteredAll = false;
        int lastPrependIndex = -1;
        List<Integer> oneDimensionWithAllEncountered = new ArrayList<>();

        //accumulate the results
        List<Long> accumShape = new ArrayList<>();
        List<Long> accumStrides = new ArrayList<>();
        List<Long> accumOffsets = new ArrayList<>();
        List<Long> intervalStrides = new ArrayList<>();

        //collect the indexes of the points that get removed
        //for point purposes
        //this will be used to compute the offset
        //for the new array
        List<Long> pointStrides = new ArrayList<>();
        List<Long> pointOffsets = new ArrayList<>();
        int numPointIndexes = 0;

        //bump number to read from the shape
        int shapeIndex = 0;
        //stride index to read strides from the array
        int strideIndex = 0;
        //list of indexes to prepend to for new axes
        //if all is encountered
        List<Integer> prependNewAxes = new ArrayList<>();
        for (int i = 0; i < indexes.length; i++) {
            INDArrayIndex idx = indexes[i];
            if (idx instanceof NDArrayIndexAll) {
                encounteredAll = true;
                if (i < arr.rank() && arr.size(i) == 1)
                    oneDimensionWithAllEncountered.add(i);
                //different dimension from new axis (look for new axis dimensions
                //at at the beginning. track when the last new axis is encountered.
                if (newAxesPrepend > 0 && lastPrependIndex < 0) {
                    lastPrependIndex = i - 1;
                }
            }
            //point: do nothing but move the shape counter
            //also move the stride counter
            if (idx instanceof PointIndex) {
                pointOffsets.add(idx.offset());
                pointStrides.add((long) arr.stride(strideIndex));
                numPointIndexes++;
                shapeIndex++;
                strideIndex++;
                //different dimension from new axis (look for new axis dimensions
                //at at the beginning. track when the last new axis is encountered.
                if (newAxesPrepend > 0 && lastPrependIndex < 0) {
                    lastPrependIndex = i - 1;
                }
                continue;
            }
            //new axes encountered, need to track whether to prepend or
            //to set the new axis in the middle
            else if (idx instanceof NewAxis) {
                //prepend the new axes at different indexes
                accumShape.add(1L);
                accumOffsets.add(0L);
                accumStrides.add(0L);
                prependNewAxes.add(i);
                continue;

            }

            //points and intervals both have a direct desired length
            else if (idx instanceof IntervalIndex && !(idx instanceof NDArrayIndexAll)
                    || idx instanceof SpecifiedIndex) {
                if (idx instanceof IntervalIndex) {
                    accumStrides.add(arr.stride(strideIndex) * idx.stride());
                    //used in computing an adjusted offset for the augmented strides
                    intervalStrides.add(idx.stride());
                    numIntervals++;
                }

                else
                    accumStrides.add((long) arr.stride(strideIndex));
                accumShape.add(idx.length());
                //the stride stays the same
                //add the offset for the index
                if (idx instanceof IntervalIndex) {
                    accumOffsets.add(idx.offset());
                } else
                    accumOffsets.add(idx.offset());

                shapeIndex++;
                strideIndex++;

                //different dimension from new axis (look for new axis dimensions
                //at at the beginning. track when the last new axis is encountered.
                if (newAxesPrepend > 0 && lastPrependIndex < 0) {
                    lastPrependIndex = i - 1;
                }


                continue;
            }

            //add the shape and stride
            //based on the original stride/shape

            accumShape.add((long) shape[shapeIndex++]);
            //account for erroneous strides from dimensions of size 1
            //move the stride index if its one and fill it in at the bottom
            accumStrides.add((long) arr.stride(strideIndex++));

            //default offsets are zero
            accumOffsets.add(idx.offset());

        }



        //fill in missing strides and shapes
        while (shapeIndex < shape.length) {
            //scalar, should be 1 x 1 rather than the number of columns in the vector
            if (Shape.isVector(shape)) {
                accumShape.add(1L);
                shapeIndex++;
            } else
                accumShape.add((long) shape[shapeIndex++]);
        }


        //fill in the rest of the offsets with zero
        int delta = (shape.length <= 2 ? shape.length : shape.length - numPointIndexes);
        boolean needsFilledIn = accumShape.size() != accumStrides.size() && accumOffsets.size() != accumShape.size();
        while (accumOffsets.size() < delta && needsFilledIn)
            accumOffsets.add(0L);


        while (accumShape.size() < 2) {
            if (Shape.isRowVectorShape(arr.shape()))
                accumShape.add(0, 1L);
            else
                accumShape.add(1L);
        }

        while (strideIndex < accumShape.size()) {
            accumStrides.add((long) arr.stride(strideIndex++));
        }


        /**
         * For each dimension
         * where we want to prepend a dimension
         * we need to add it at the index such that
         * we account for the offset of the number of indexes
         * added up to that point.
         *
         * We do this by doing an offset
         * for each item added "so far"
         *
         * Note that we also have an offset of - 1
         * because we want to prepend to the given index.
         *
         * When prepend new axes for in the middle is triggered
         * i is already > 0
         */
       /* int numAdded = 0;
        for (int i = 0; i < prependNewAxes.size(); i++) {
            accumShape.add(prependNewAxes.get(i) - numAdded, 1L);
            //stride for the new axis is zero
            accumStrides.add(prependNewAxes.get(i) - numAdded, 0L);
            numAdded++;
        }
        for (int i = 0; i < newAxesPrepend; i++) {
            prependNewAxes.add(0, i);
        }

        prependAxis = Ints.toArray(prependNewAxes);
*/
        /**
         * Need to post process strides and offsets
         * for trailing ones here
         */
        //prune off extra zeros for trailing and leading ones
        int trailingZeroRemove = accumOffsets.size() - 1;
        while (accumOffsets.size() > accumShape.size()) {
            if (accumOffsets.get(trailingZeroRemove) == 0)
                accumOffsets.remove(accumOffsets.size() - 1);
            trailingZeroRemove--;
        }

        if (accumStrides.size() < accumOffsets.size())
            accumStrides.addAll(pointStrides);
        while (accumOffsets.size() < accumShape.size()) {
            if (Shape.isRowVectorShape(arr.shape()))
                accumOffsets.add(0, 0L);
            else
                accumOffsets.add(0L);
        }


        if (Shape.isMatrix(shape) && indexes[0] instanceof PointIndex && indexes[1] instanceof NDArrayIndexAll) {
            Collections.reverse(accumShape);
        }

        if (arr.isMatrix() && indexes[0] instanceof PointIndex && indexes[1] instanceof IntervalIndex) {
            this.shapes = new long[2];
            shapes[0] = 1;
            IntervalIndex idx = (IntervalIndex) indexes[1];
            shapes[1] = idx.length();

        } else
            this.shapes = Longs.toArray(accumShape);


        boolean isColumnVector = Shape.isColumnVectorShape(this.shapes);
        //finally fill in teh rest of the strides if any are left over
        while (accumStrides.size() < accumOffsets.size()) {
            if (!isColumnVector)
                accumStrides.add(0, (long) arr.elementStride());
            else
                accumStrides.add((long) arr.elementStride());
        }


        this.strides = Longs.toArray(accumStrides);
        this.offsets = Longs.toArray(accumOffsets);

        //compute point offsets differently
        /**
         * We need to prepend the strides for the point indexes
         * such that the point index offsets are counted.
         * Note here that we only use point strides
         * when points strides isn't empty.
         * When point strides is empty, this is
         * because a point index was encountered
         * but it was the lead index and therefore should
         * not be counted with the offset.
         *
         *
         * Another thing of note here is that the strides
         * and offsets should line up such that the point
         * and stride match up.
         */
        if (numPointIndexes > 0 && !pointStrides.isEmpty()) {
            //append to the end for tensors
            if (newAxesPrepend >= 1) {
                while (pointStrides.size() < accumOffsets.size()) {
                    pointStrides.add(1L);
                }
                //identify in the original accumulate strides
                //where zero was set and emulate the
                //same structure in the point strides
                for (int i = 0; i < accumStrides.size(); i++) {
                    if (accumStrides.get(i) == 0 && !(indexes[i] instanceof NewAxis) && lastPrependIndex <= 0)
                        pointStrides.set(i, 0L);
                }
            }

            //prepend any missing offsets where relevant for the dot product
            //note here we are using the point offsets and strides
            //for computing the offset
            //the point of a point index is to drop a dimension
            //and index in to a particular offset
            while (pointOffsets.size() < pointStrides.size()) {
                pointOffsets.add(0L);
            }
            //special case where offsets aren't caught
            if (arr.isRowVector() && !intervalStrides.isEmpty() && pointOffsets.get(0) == 0
                    && !(indexes[1] instanceof IntervalIndex))
                this.offset = indexes[1].offset();
            else
                this.offset = ArrayUtil.dotProductLong2(pointOffsets, pointStrides);
        } else {
            this.offset = 0;
        }
        if (numIntervals > 0 && arr.rank() > 2) {
            if (encounteredAll && arr.size(0) != 1 || indexes[0] instanceof PointIndex)
                // FIXME: LONG
                this.offset += ArrayUtil.dotProductLong2(accumOffsets, accumStrides);
            else
                // FIXME: LONG
                this.offset += ArrayUtil.dotProductLong2(accumOffsets, accumStrides);

        } else if (numIntervals > 0 && anyHaveStrideOne(indexes))
            this.offset += ArrayUtil.calcOffsetLong2(accumShape, accumOffsets, accumStrides);
        else
            this.offset += ArrayUtil.calcOffsetLong2(accumShape, accumOffsets, accumStrides)
                    / Math.max(1, numIntervals);
    }

    public void resolveFixedDimensionsCOO(INDArrayIndex... indexes) {

        fixed = new int[arr.rank()];

        int j = 0;
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] instanceof PointIndex) {
                fixed[j] = 1;
                j++;
            }
            if (indexes[i] instanceof IntervalIndex || indexes[i] instanceof NDArrayIndexAll) {
                fixed[j] = 0;
                j++;
            }
            if (indexes[i] instanceof SpecifiedIndex) {
                SpecifiedIndex idx = (SpecifiedIndex) indexes[i];
                if (idx.getIndexes().length == 1) {
                    fixed[j] = 1;
                } else {
                    fixed[j] = 0;
                }
                j++;
            }
            if (indexes[i] instanceof NewAxis) {
                //do nothing, skip the index
            }
        }
    }

    public void resolveSparseOffsetsCOO() {

    }

    private boolean anyHaveStrideOne(INDArrayIndex... indexes) {
        for (INDArrayIndex indArrayIndex : indexes)
            if (indArrayIndex.stride() == 1)
                return true;
        return false;
    }

    private boolean allIndexGreatherThanZero(INDArrayIndex... indexes) {
        for (INDArrayIndex indArrayIndex : indexes)
            if (indArrayIndex.offset() == 0)
                return false;
        return true;
    }

}
