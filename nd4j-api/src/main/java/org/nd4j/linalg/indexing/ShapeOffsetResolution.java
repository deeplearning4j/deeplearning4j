package org.nd4j.linalg.indexing;

import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.api.shape.Shape;

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
public class ShapeOffsetResolution implements Serializable {
    private INDArray arr;
    private int[] offsets,shapes,strides;
    private int offset = -1;

    /**
     * Specify the array to use for resolution
     * @param arr the array to use
     *            for resolution
     */
    public ShapeOffsetResolution(INDArray arr) {
        this.arr = arr;
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
        indexes = NDArrayIndex.resolve(arr.shape(),indexes);
        int[] shape = arr.shape();
        //number of new axes dimensions to prepend to the beginning
        int newAxesPrepend = 0;
        //whether we have encountered an all so far
        boolean encounteredAll = false;
        //accumulate the results
        List<Integer> accumShape = new ArrayList<>();
        List<Integer> accumStrides = new ArrayList<>();
        List<Integer> accumOffsets = new ArrayList<>();

        //collect the indexes of the points that get removed
        //for point purposes
        //this will be used to compute the offset
        //for the new array
        List<Integer> pointStrides = new ArrayList<>();
        int numPointIndexes = 0;

        //bump number to read from the shape
        int shapeIndex = 0;
        //stride index to read strides from the array
        int strideIndex = 0;
        //list of indexes to prepend to for new axes
        //if all is encountered
        List<Integer> prependNewAxes = new ArrayList<>();
        /**
         * Need to account for strides and offsets that are 1
         * eg: make them not count for tensors
         */


        for(int i = 0; i < indexes.length; i++) {
            INDArrayIndex idx = indexes[i];
            if (idx instanceof NDArrayIndexAll)
                encounteredAll = true;
            //point: do nothing but move the shape counter
            //also move the stride counter
            if(idx instanceof PointIndex) {
                if(idx.offset() > 0)
                    accumOffsets.add(idx.offset());
                pointStrides.add(arr.stride(i));
                numPointIndexes++;
                shapeIndex++;
                strideIndex++;
                continue;
            }
            //new axes encountered, need to track whether to prepend or
            //to set the new axis in the middle
            else if(idx instanceof NewAxis) {
                //prepend the new axes at different indexes
                if(encounteredAll) {
                    prependNewAxes.add(i);
                }
                //prepend to the beginning
                //rather than a set index
                else
                    newAxesPrepend++;
                continue;

            }

            //points and intervals both have a direct desired length
            else if(idx instanceof IntervalIndex && !(idx instanceof NDArrayIndexAll) || idx instanceof SpecifiedIndex) {
                accumShape.add(idx.length());
                //the stride stays the same
                accumStrides.add(arr.stride(strideIndex++));
                //add the offset for the index
                accumOffsets.add(idx.offset());
                shapeIndex++;
                continue;
            }

            //add the shape and stride
            //based on the original stride/shape

            accumShape.add(shape[shapeIndex++]);
            //account for erroneous strides from dimensions of size 1
            //move the stride index if its one and fill it in at the bottom
            if(accumShape.get(accumShape.size() - 1) != 1) {
                accumStrides.add(arr.stride(strideIndex++));
            }

            else
                strideIndex++;
            //default offsets are zero
            accumOffsets.add(0);

        }

        //fill in missing strides and shapes
        while(shapeIndex < shape.length) {
            //scalar, should be 1 x 1 rather than the number of columns in the vector
            if(Shape.isVector(shape)) {
                accumShape.add(1);
                shapeIndex++;
            }
            else
                accumShape.add(shape[shapeIndex++]);
        }

        while(strideIndex < arr.stride().length) {
            //scalar: stride should be element wise
            if(Shape.isVector(shape)) {
                accumStrides.add(arr.elementStride());
                strideIndex++;
            }
            else
                accumStrides.add(arr.stride(strideIndex++));
        }

        //fill in the rest of the offsets with zero
        int delta = (shape.length <= 2 ? shape.length : shape.length - numPointIndexes);
        boolean needsFilledIn = accumShape.size() != accumStrides.size() && accumOffsets.size() != accumShape.size();
        while(accumOffsets.size() < delta && needsFilledIn)
            accumOffsets.add(0);


        while(accumShape.size() < 2) {
            accumShape.add(1);
            //one stride will have been removed
            accumStrides.add(pointStrides.remove(0));

        }

        //only one index and matrix, remove the first index rather than the last
        //equivalent to this is reversing the list with the prepended one
        if(indexes.length <= 2 && indexes[0] instanceof PointIndex && shape.length == 2) {
            Collections.reverse(accumShape);
            Collections.reverse(accumStrides);
        }

        //prepend for new axes; do this first before
        //doing the indexes to prepend to
        if(newAxesPrepend > 0) {
            for(int i = 0; i < newAxesPrepend; i++) {
                accumShape.add(0, 1);
                //strides for new axis are 0
                accumStrides.add(0,0);
                //prepend offset zero to match the stride and shapes
                accumOffsets.add(0,0);
            }
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
        for(int i = 0; i < prependNewAxes.size(); i++) {
            accumShape.add(prependNewAxes.get(i) - i,1);
            //stride for the new axis is zero
            accumStrides.add(prependNewAxes.get(i) - i,0);
        }


        /**
         * Need to post process strides and offsets
         * for trailing ones here
         */
        //prune off extra zeros for trailing and leading ones
        int trailingZeroRemove = accumOffsets.size() - 1;
        while(accumOffsets.size() > accumShape.size()) {
            if(accumOffsets.get(trailingZeroRemove) == 0)
                accumOffsets.remove(accumOffsets.size() - 1);
            trailingZeroRemove--;
        }

        if(accumStrides.size() < accumOffsets.size())
            accumStrides.addAll(pointStrides);
        //finally fill in teh rest of the strides if any are left over
        while(accumStrides.size() < accumOffsets.size()) {
            accumStrides.add(arr.elementStride());
        }

        this.strides = Ints.toArray(accumStrides);
        this.shapes = Ints.toArray(accumShape);
        this.offsets = Ints.toArray(accumOffsets);

        this.offset = ArrayUtil.dotProduct(accumOffsets,accumStrides);


    }

    public INDArray getArr() {
        return arr;
    }

    public void setArr(INDArray arr) {
        this.arr = arr;
    }

    public int[] getOffsets() {
        return offsets;
    }

    public void setOffsets(int[] offsets) {
        this.offsets = offsets;
    }

    public int[] getShapes() {
        return shapes;
    }

    public void setShapes(int[] shapes) {
        this.shapes = shapes;
    }

    public int[] getStrides() {
        return strides;
    }

    public void setStrides(int[] strides) {
        this.strides = strides;
    }

    public int getOffset() {
        return offset;
    }

    public void setOffset(int offset) {
        this.offset = offset;
    }
}
