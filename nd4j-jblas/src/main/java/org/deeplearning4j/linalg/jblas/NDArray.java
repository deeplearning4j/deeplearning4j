package org.deeplearning4j.linalg.jblas;


import org.deeplearning4j.linalg.api.ndarray.BaseNDArray;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.jblas.DoubleMatrix;


import java.util.*;


/**
 * NDArray: (think numpy)
 *
 * A few things of note.
 *
 * An NDArray can have any number of dimensions.
 *
 * An NDArray is accessed via strides.
 *
 * Strides are how to index over
 * a contiguous block of data.
 *
 * This block of data has 2 orders(as of right now):
 * fortran and c
 *
 *
 *
 * @author Adam Gibson
 */
public class NDArray extends BaseNDArray {


    public NDArray(double[][] data) {
        super(data);
    }

    /**
     * Create this ndarray with the given data and shape and 0 offset
     *
     * @param data     the data to use
     * @param shape    the shape of the ndarray
     * @param ordering
     */
    public NDArray(float[] data, int[] shape, char ordering) {
        super(data, shape, ordering);
    }

    /**
     * @param data     the data to use
     * @param shape    the shape of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering of the ndarray
     */
    public NDArray(float[] data, int[] shape, int offset, char ordering) {
        super(data, shape, offset, ordering);
    }

    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering of the ndarray
     */
    public NDArray(int[] shape, int[] stride, int offset, char ordering) {
        super(shape, stride, offset, ordering);
    }

    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param ordering the ordering of the ndarray
     */
    public NDArray(int[] shape, int[] stride, char ordering) {
        super(shape, stride, ordering);
    }

    public NDArray(int[] shape, int offset, char ordering) {
        super(shape, offset, ordering);
    }

    public NDArray(int[] shape) {
        super(shape);
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     * @param ordering
     */
    public NDArray(int newRows, int newColumns, char ordering) {
        super(newRows, newColumns, ordering);
    }

    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices   the slices to merge
     * @param shape    the shape of the ndarray
     * @param ordering
     */
    public NDArray(List<INDArray> slices, int[] shape, char ordering) {
        super(slices, shape, ordering);
    }

    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices   the slices to merge
     * @param shape    the shape of the ndarray
     * @param stride
     * @param ordering
     */
    public NDArray(List<INDArray> slices, int[] shape, int[] stride, char ordering) {
        super(slices, shape, stride, ordering);
    }

    public NDArray(float[] data, int[] shape, int[] stride, char ordering) {
        super(data, shape, stride, ordering);
    }

    public NDArray(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    /**
     * Create this ndarray with the given data and shape and 0 offset
     *
     * @param data  the data to use
     * @param shape the shape of the ndarray
     */
    public NDArray(float[] data, int[] shape) {
        super(data, shape);
    }

    public NDArray(float[] data, int[] shape, int offset) {
        super(data, shape, offset);
    }

    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public NDArray(int[] shape, int[] stride, int offset) {
        super(shape, stride, offset);
    }

    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public NDArray(int[] shape, int[] stride) {
        super(shape, stride);
    }

    public NDArray(int[] shape, int offset) {
        super(shape, offset);
    }

    public NDArray(int[] shape, char ordering) {
        super(shape, ordering);
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public NDArray(int newRows, int newColumns) {
        super(newRows, newColumns);
    }

    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     */
    public NDArray(List<INDArray> slices, int[] shape) {
        super(slices, shape);
    }

    /**
     * Create an ndarray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one ndarray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the ndarray
     * @param stride
     */
    public NDArray(List<INDArray> slices, int[] shape, int[] stride) {
        super(slices, shape, stride);
    }

    public NDArray(float[] data, int[] shape, int[] stride) {
        super(data, shape, stride);
    }



    public NDArray(float[] data, int[] shape, int[] stride, int offset) {
        super(data, shape, stride, offset);
    }


    public NDArray(DoubleMatrix doubleMatrix) {
        this(new int[]{doubleMatrix.rows,doubleMatrix.columns});
        this.data = ArrayUtil.floatCopyOf(doubleMatrix.data);

    }

    public NDArray(double[] data, int[] shape, int[] stride, int offset) {
        this.data = ArrayUtil.floatCopyOf(data);
        this.stride = stride;
        this.offset = offset;
        initShape(shape);
    }

    public NDArray(float[][] floats) {
        super(floats);
    }
}