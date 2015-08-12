package org.nd4j.linalg.indexing;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.api.shape.Shape;

import java.io.Serializable;
import java.util.Arrays;

/**
 *
 * Sets up the strides, shape, and offsets
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
        indexes = NDArrayIndex.resolve(arr,indexes);
        int[] shape = Indices.shape(arr.shape(), indexes);
        int[] offsets = Indices.offsets(shape,indexes);
        if(offsets.length < shape.length) {
            int[] filledOffsets = new int[shape.length];
            System.arraycopy(offsets,0,filledOffsets,0,offsets.length);
        }

        int[] stride = Indices.stride(arr,indexes,shape);
        if(stride.length < shape.length) {
            int[] filledStrides = new int[shape.length];
            Arrays.fill(filledStrides,arr.elementStride());
            System.arraycopy(stride,0,filledStrides,0,stride.length);
        }

        /**
         * Appears to be problems with coordinates like
         * 1,1,0
         * when calculating offsets
         */
        if(shape[0] == 1 && shape.length > 2 && !Indices.isScalar(arr,indexes)) {
            boolean[] ones = new boolean[shape.length - 1];
            int[] newShape  = new int[shape.length - 1];
            boolean allOnes = true;
            for(int i = 0; i < newShape.length; i++) {
                newShape[i] = shape[i + 1];
                ones[i] = newShape[i] == 1;
                allOnes = allOnes && ones[i];
            }

            shape = newShape;

            if(allOnes) {
                int[] newOffsets = new int[newShape.length];
                int[] newStrides = new int[newOffsets.length];
                for(int i = 0; i < newShape.length; i++) {
                    if(ones[i]) {
                        if(i > 0 && i < offsets.length - 1 && i < stride.length - 1) {
                            int offsetPlaceHolder = offsets[i + 1];
                            int stridePlaceHolder = stride[i + 1];

                            newOffsets[i] = newOffsets[i - 1];
                            newStrides[i] = newStrides[i - 1];
                            newOffsets[i - 1] = offsetPlaceHolder;
                            newStrides[i - 1] = stridePlaceHolder;

                        }
                        else {
                            //grab from the previous index
                            if(offsets[i + 1] == 0 && offsets[i] > 0) {
                                newOffsets[i] = offsets[i];
                                newStrides[i] = stride[i];

                            }
                            else if(i < offsets.length - 1 && i < stride.length - 1) {
                                newOffsets[i] = offsets[i + 1];
                                newStrides[i] = stride[i + 1];

                            }
                        }

                    }
                    else if(i < offsets.length - 1 && i < stride.length - 1){
                        newOffsets[i] = offsets[i + 1];
                        newStrides[i] = stride[i + 1];
                    }
                }

                //adjust for arbitrary strides with 1 sized arrays
                for(int j  = newStrides.length - 1; j > 0; j--) {
                    if(newStrides[j] >= arr.length()) {
                        newStrides[j] = arr.elementStride();
                    }
                }

                offsets = newOffsets;
                stride = newStrides;
            }


            else {
                if(newShape.length < arr.shape().length || newShape.length < arr.stride().length) {
                    if(stride.length < offsets.length) {
                        offsets = Arrays.copyOfRange(offsets,0,offsets.length - 1);
                    }
                    int[] newOffsets = Shape.squeezeOffsets(shape,offsets);
                    int[] newStrides = ArrayUtil.removeIndex(arr.stride(),0);
                    offset = ArrayUtil.dotProduct(offsets,stride);
                    offsets = newOffsets;
                    stride = newStrides;

                }
            }

        }



        if(stride.length > offsets.length) {
            stride = Arrays.copyOfRange(stride, 1, stride.length);
        }

        if(offsets.length > shape.length) {
            offsets = ArrayUtil.keep(offsets, ArrayUtil.range(0, shape.length));
        }



        this.offsets = offsets;
        this.shapes = shape;
        this.strides = stride;
        if(offset < 0)
            this.offset = ArrayUtil.dotProduct(offsets,stride);


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
