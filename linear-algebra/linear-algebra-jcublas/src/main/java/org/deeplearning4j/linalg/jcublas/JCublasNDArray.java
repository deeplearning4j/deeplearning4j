package org.deeplearning4j.linalg.jcublas;

/**
 * Created by mjk on 8/23/14.
 *
 * @author mjk
 * @author Adam Gibson
 */



import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.deeplearning4j.linalg.api.ndarray.BaseNDArray;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.ArrayUtil;


import java.util.*;


public class JCublasNDArray extends BaseNDArray {


    private Pointer pointer;


    public JCublasNDArray(double[][] data) {
        super(data);

    }

    /**
     * Create this JCublasNDArray with the given data and shape and 0 offset
     *
     * @param data     the data to use
     * @param shape    the shape of the JCublasNDArray
     * @param ordering
     */
    public JCublasNDArray(float[] data, int[] shape, char ordering) {
        super(data, shape, ordering);
        setupJcuBlas();

    }

    /**
     * @param data     the data to use
     * @param shape    the shape of the JCublasNDArray
     * @param offset   the desired offset
     * @param ordering the ordering of the JCublasNDArray
     */
    public JCublasNDArray(float[] data, int[] shape, int offset, char ordering) {
        super(data, shape, offset, ordering);
        setupJcuBlas();

    }

    /**
     * Construct an JCublasNDArray of the specified shape
     * with an empty data array
     *
     * @param shape    the shape of the JCublasNDArray
     * @param stride   the stride of the JCublasNDArray
     * @param offset   the desired offset
     * @param ordering the ordering of the JCublasNDArray
     */
    public JCublasNDArray(int[] shape, int[] stride, int offset, char ordering) {
        super(shape, stride, offset, ordering);
        setupJcuBlas();
    }

    /**
     * Create the JCublasNDArray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape    the shape of the JCublasNDArray
     * @param stride   the stride of the JCublasNDArray
     * @param ordering the ordering of the JCublasNDArray
     */
    public JCublasNDArray(int[] shape, int[] stride, char ordering) {

        super(shape, stride, ordering);
        setupJcuBlas();
    }

    public JCublasNDArray(int[] shape, int offset, char ordering) {

        super(shape, offset, ordering);
        setupJcuBlas();
    }

    public JCublasNDArray(int[] shape) {

        super(shape);
        setupJcuBlas();
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     * @param ordering
     */
    public JCublasNDArray(int newRows, int newColumns, char ordering) {
        super(newRows, newColumns, ordering);
        setupJcuBlas();
    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices   the slices to merge
     * @param shape    the shape of the JCublasNDArray
     * @param ordering
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape, char ordering) {

        super(slices, shape, ordering);
        setupJcuBlas();
    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices   the slices to merge
     * @param shape    the shape of the JCublasNDArray
     * @param stride
     * @param ordering
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape, int[] stride, char ordering) {
        super(slices, shape, stride, ordering);
        setupJcuBlas();
    }

    public JCublasNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        super(data, shape, stride, ordering);
        setupJcuBlas();
    }

    public JCublasNDArray(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        super(data, shape, stride, offset, ordering);
        setupJcuBlas();
    }

    /**
     * Create this JCublasNDArray with the given data and shape and 0 offset
     *
     * @param data  the data to use
     * @param shape the shape of the JCublasNDArray
     */
    public JCublasNDArray(float[] data, int[] shape) {
        super(data, shape);
    }

    public JCublasNDArray(float[] data, int[] shape, int offset) {

        super(data, shape, offset);
        setupJcuBlas();
    }

    /**
     * Construct an JCublasNDArray of the specified shape
     * with an empty data array
     *
     * @param shape  the shape of the JCublasNDArray
     * @param stride the stride of the JCublasNDArray
     * @param offset the desired offset
     */
    public JCublasNDArray(int[] shape, int[] stride, int offset) {

        super(shape, stride, offset);
        setupJcuBlas();
    }

    /**
     * Create the JCublasNDArray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the JCublasNDArray
     * @param stride the stride of the JCublasNDArray
     */
    public JCublasNDArray(int[] shape, int[] stride) {
        super(shape, stride);
        setupJcuBlas();
    }

    public JCublasNDArray(int[] shape, int offset) {
        super(shape, offset);
        setupJcuBlas();
    }

    public JCublasNDArray(int[] shape, char ordering) {
        super(shape, ordering);
        setupJcuBlas();
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public JCublasNDArray(int newRows, int newColumns) {
        super(newRows, newColumns);
        setupJcuBlas();
    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the JCublasNDArray
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape) {
        super(slices, shape);
        setupJcuBlas();
    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the JCublasNDArray
     * @param stride
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape, int[] stride) {
        super(slices, shape, stride);
        setupJcuBlas();
    }

    public JCublasNDArray(float[] data, int[] shape, int[] stride) {
        super(data, shape, stride);
        setupJcuBlas();
    }



    public JCublasNDArray(float[] data, int[] shape, int[] stride, int offset) {
        super(data, shape, stride, offset);
        setupJcuBlas();
    }


    public JCublasNDArray(JCublasNDArray doubleMatrix) {
        this(new int[]{doubleMatrix.rows,doubleMatrix.columns});
        this.data = dup().data();
        setupJcuBlas();
    }

    public JCublasNDArray(double[] data, int[] shape, int[] stride, int offset) {
        this.data = ArrayUtil.floatCopyOf(data);
        this.stride = stride;
        this.offset = offset;
        initShape(shape);
        setupJcuBlas();
    }

    public JCublasNDArray(float[][] floats) {
        super(floats);
        setupJcuBlas();
    }


    protected void setupJcuBlas() {
        if(pointer != null)
            return;
        pointer = new Pointer().withByteOffset(offset());
        JCublas.cublasAlloc(length, Sizeof.FLOAT, pointer);
        JCublas.cublasSetVector(length, Sizeof.FLOAT, Pointer.to(data()), stride[0], pointer, stride[0]);

    }

    public void getData() {
        JCublas.cublasGetVector(length, Sizeof.FLOAT, pointer, stride[0], Pointer.to(data()), stride[0]);

    }



    public Pointer pointer() {
        return pointer;
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other  the other matrix to perform matrix multiply with
     * @param result the result ndarray
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmuli(INDArray other, INDArray result) {
        return super.mmuli(other, result);
    }
}