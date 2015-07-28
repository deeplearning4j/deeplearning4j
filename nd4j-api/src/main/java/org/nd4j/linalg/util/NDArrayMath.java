package org.nd4j.linalg.util;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class NDArrayMath {


    /**
     * Compute the offset for a given slice
     * @param arr the array to compute
     *            the offset frm
     * @param slice the slice to compute the offset for
     * @return the offset for a given slice
     */
    public static int offsetForSlice(INDArray arr,int slice) {
        return slice * lengthPerSlice(arr);
    }

    /**
     * The number of elements in a slice
     * along a set of dimensions
     * @param arr the array
     *            to calculate the length per slice for
     * @param dimension the dimensions to do the calculations along
     * @return the number of elements in a slice along
     * arbitrary dimensions
     */
    public static int lengthPerSlice(INDArray arr,int...dimension) {
        return ArrayUtil.prod(ArrayUtil.removeIndex(arr.shape(), dimension));
    }

    /**
     * Return the length of a slice
     * @param arr the array to get the length of a slice for
     * @return the number of elements per slice in an array
     */
    public static int lengthPerSlice(INDArray arr) {
        return lengthPerSlice(arr,0);
    }


    /**
     * Return the number of vectors for an array
     * the number of vectors for an array
     * @param arr the array to calculate the number of vectors for
     * @return the number of vectors for the given array
     */
    public static int numVectors(INDArray arr) {
        if(arr.rank() == 1)
            return 1;
        else if(arr.rank() == 2)
            return arr.size(0);
        else {
            int prod = 1;
            for(int i = 0; i < arr.rank() - 1; i++) {
                prod *= arr.size(i);
            }

            return prod;
        }
    }


    /**
     * The number of vectors
     * in each slice of an ndarray.
     * @param arr the array to
     *            get the number
     *            of vectors per slice for
     * @return the number of vectors per slice
     */
    public static int vectorsPerSlice(INDArray arr) {
        if(arr.rank() > 2) {
            return ArrayUtil.prod(new int[]{arr.size(-1),arr.size(-2)});
        }

        return 1;
    }

    /**
     * The number of vectors
     * in each slice of an ndarray.
     * @param arr the array to
     *            get the number
     *            of vectors per slice for
     * @return the number of vectors per slice
     */
    public static int matricesPerSlice(INDArray arr) {
        if(arr.rank() == 3) {
            return 1;
        }
        else if(arr.rank() > 3) {
            int ret = 1;
            for(int i = 1; i < arr.rank() - 2; i++) {
                ret *= arr.size(i);
            }
            return ret;
        }
        return arr.size(-2);
    }

    /**
     * The number of vectors
     * in each slice of an ndarray.
     * @param arr the array to
     *            get the number
     *            of vectors per slice for
     * @param rank the dimensions to get the number of vectors per slice for
     * @return the number of vectors per slice
     */
    public static int vectorsPerSlice(INDArray arr,int...rank) {
        if(arr.rank() > 2) {
            return arr.size(-2) * arr.size(-1);
        }

        return arr.size(-1);

    }


    /**
     * This maps an index of a vector
     * on to a vector in the matrix that can be used
     * for indexing in to a tensor
     * @param index the index to map
     * @param arr the array to use
     *            for indexing
     * @return the mapped index
     */
    public static int mapIndexOntoVector(int index,INDArray arr) {
        int numVectors = NDArrayMath.numVectors(arr);
        int ret = index * arr.size(-1);
        return ret;
    }

    /**
     * Returns the slice a vector belongs to
     * @param vector the vector to get the slice for
     * @param arr the array to get the vector for
     * @param rank the dimensions to slice
     * @return the slice for a particular vector
     */
    public static int sliceForVector(int vector,INDArray arr,int...rank) {
        if(vector == 0)
            return 0;

        int mapped = NDArrayMath.mapIndexOntoVector(vector,arr);
        int ret = 0;
        for(int i = 0; i < arr.slices(); i++) {
            int offset = NDArrayMath.offsetForSlice(arr,i);
            if(mapped >= offset) {
                ret = i;
            }

        }

        return ret;




    }


}
