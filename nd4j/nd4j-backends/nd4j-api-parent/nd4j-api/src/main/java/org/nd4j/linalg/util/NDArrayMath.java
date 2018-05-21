package org.nd4j.linalg.util;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class NDArrayMath {


    private NDArrayMath() {}

    /**
     * Compute the offset for a given slice
     * @param arr the array to compute
     *            the offset frm
     * @param slice the slice to compute the offset for
     * @return the offset for a given slice
     */
    public static long offsetForSlice(INDArray arr, int slice) {
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
    public static long lengthPerSlice(INDArray arr, int... dimension) {
        long[] remove = ArrayUtil.removeIndex(arr.shape(), dimension);
        return ArrayUtil.prodLong(remove);
    }

    /**
     * Return the length of a slice
     * @param arr the array to get the length of a slice for
     * @return the number of elements per slice in an array
     */
    public static long lengthPerSlice(INDArray arr) {
        return lengthPerSlice(arr, 0);
    }


    /**
     * Return the number of vectors for an array
     * the number of vectors for an array
     * @param arr the array to calculate the number of vectors for
     * @return the number of vectors for the given array
     */
    public static long numVectors(INDArray arr) {
        if (arr.rank() == 1)
            return 1;
        else if (arr.rank() == 2)
            return arr.size(0);
        else {
            int prod = 1;
            for (int i = 0; i < arr.rank() - 1; i++) {
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
    public static long vectorsPerSlice(INDArray arr) {
        if (arr.rank() > 2) {
            return ArrayUtil.prodLong(new long[] {arr.size(-1), arr.size(-2)});
        }

        return arr.slices();
    }


    /**
     * Computes the tensors per slice
     * given a tensor shape and array
     * @param arr the array to get the tensors per slice for
     * @param tensorShape the desired tensor shape
     * @return the tensors per slice of an ndarray
     */
    public static long tensorsPerSlice(INDArray arr, int[] tensorShape) {
        return lengthPerSlice(arr) / ArrayUtil.prod(tensorShape);
    }

    /**
     * The number of vectors
     * in each slice of an ndarray.
     * @param arr the array to
     *            get the number
     *            of vectors per slice for
     * @return the number of vectors per slice
     */
    public static long matricesPerSlice(INDArray arr) {
        if (arr.rank() == 3) {
            return 1;
        } else if (arr.rank() > 3) {
            int ret = 1;
            for (int i = 1; i < arr.rank() - 2; i++) {
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
    public static long vectorsPerSlice(INDArray arr, int... rank) {
        if (arr.rank() > 2) {
            return arr.size(-2) * arr.size(-1);
        }

        return arr.size(-1);

    }

    /**
     * calculates the offset for a tensor
     * @param index
     * @param arr
     * @param tensorShape
     * @return
     */
    public static long sliceOffsetForTensor(int index, INDArray arr, int[] tensorShape) {
        long tensorLength = ArrayUtil.prodLong(tensorShape);
        long lengthPerSlice = NDArrayMath.lengthPerSlice(arr);
        long offset = index * tensorLength / lengthPerSlice;
        return offset;
    }

    public static long sliceOffsetForTensor(int index, INDArray arr, long[] tensorShape) {
        long tensorLength = ArrayUtil.prodLong(tensorShape);
        long lengthPerSlice = NDArrayMath.lengthPerSlice(arr);
        long offset = index * tensorLength / lengthPerSlice;
        return offset;
    }


    /**
     * This maps an index of a vector
     * on to a vector in the matrix that can be used
     * for indexing in to a tensor
     * @param index the index to map
     * @param arr the array to use
     *            for indexing
     * @param rank the dimensions to compute a slice for
     * @return the mapped index
     */
    public static int mapIndexOntoTensor(int index, INDArray arr, int... rank) {
        int ret = index * ArrayUtil.prod(ArrayUtil.removeIndex(arr.shape(), rank));
        return ret;
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
    public static long mapIndexOntoVector(int index, INDArray arr) {
        long ret = index * arr.size(-1);
        return ret;
    }



}
