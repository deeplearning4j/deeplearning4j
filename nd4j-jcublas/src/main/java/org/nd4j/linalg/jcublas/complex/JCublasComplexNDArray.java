package org.nd4j.linalg.jcublas.complex;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import org.nd4j.linalg.api.complex.BaseComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Created by mjk on 8/23/14.
 *
 * @author mjk
 * @author Adam Gibson
 */
public class JCublasComplexNDArray  extends BaseComplexNDArray {

    private Pointer pointer,dataPointer;

    public JCublasComplexNDArray(int[] shape, int offset, char ordering) {

        super(shape, offset, ordering);
        setupJcuBlas();
    }

    public JCublasComplexNDArray(int[] shape) {

        super(shape);
        setupJcuBlas();
    }


    public JCublasComplexNDArray(int[] shape, char ordering) {

        super(shape, ordering);
        setupJcuBlas();
    }

    /**
     * Initialize the given ndarray as the real component
     *
     * @param m        the real component
     * @param stride   the stride of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public JCublasComplexNDArray(INDArray m, int[] stride, char ordering) {
        super(m, stride, ordering);
        setupJcuBlas();
    }

    /**
     * Construct a complex matrix from a realComponent matrix.
     *
     * @param m
     * @param ordering
     */
    public JCublasComplexNDArray(INDArray m, char ordering) {

        super(m, ordering);
        setupJcuBlas();
    }

    /**
     * Construct a complex matrix from a realComponent matrix.
     *
     * @param m
     */
    public JCublasComplexNDArray(INDArray m) {

        super(m);
        setupJcuBlas();
    }

    /**
     * Create with the specified ndarray as the real component
     * and the given stride
     *
     * @param m      the ndarray to use as the stride
     * @param stride the stride of the ndarray
     */
    public JCublasComplexNDArray(INDArray m, int[] stride) {

        super(m, stride);
        setupJcuBlas();
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices the slices of the ndarray
     * @param shape  the final shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public JCublasComplexNDArray(List<IComplexNDArray> slices, int[] shape, int[] stride) {
        super(slices, shape, stride);
        setupJcuBlas();
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices   the slices of the ndarray
     * @param shape    the final shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public JCublasComplexNDArray(List<IComplexNDArray> slices, int[] shape, int[] stride, char ordering) {
        super(slices, shape, stride, ordering);
        setupJcuBlas();
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices   the slices of the ndarray
     * @param shape    the final shape of the ndarray
     * @param ordering the ordering of the ndarray
     */
    public JCublasComplexNDArray(List<IComplexNDArray> slices, int[] shape, char ordering) {
        super(slices, shape, ordering);
        setupJcuBlas();
    }

    /**
     * Create an ndarray from the specified slices
     * and the given shape
     *
     * @param slices the slices of the ndarray
     * @param shape  the final shape of the ndarray
     */
    public JCublasComplexNDArray(List<IComplexNDArray> slices, int[] shape) {

        super(slices, shape);
        setupJcuBlas();
    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new double[]
     *
     * @param newData the new data for this array
     * @param shape   the shape of the ndarray
     */
    public JCublasComplexNDArray(IComplexNumber[] newData, int[] shape) {

        super(newData, shape);
        setupJcuBlas();
    }


    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new double[]
     *
     * @param newData the new data for this array
     * @param shape   the shape of the ndarray
     * @param stride
     */
    public JCublasComplexNDArray(IComplexNumber[] newData, int[] shape, int[] stride) {

        super(newData, shape, stride);
        setupJcuBlas();
    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new double[]
     *
     * @param newData the new data for this array
     * @param shape   the shape of the ndarray
     */
    public JCublasComplexNDArray(IComplexDouble[] newData, int[] shape) {

        super(newData, shape);
        setupJcuBlas();
    }

    /**
     * Create a complex ndarray with the given complex doubles.
     * Note that this maybe an easier setup than the new double[]
     *
     * @param newData  the new data for this array
     * @param shape    the shape of the ndarray
     * @param ordering the ordering for the ndarray
     */
    public JCublasComplexNDArray(IComplexDouble[] newData, int[] shape, char ordering) {
        super(newData, shape, ordering);
        setupJcuBlas();
    }


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public JCublasComplexNDArray(int[] shape, int[] stride, int offset) {

        super(shape, stride, offset);
        setupJcuBlas();
    }

    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param offset   the desired offset
     * @param ordering the ordering for the ndarray
     */
    public JCublasComplexNDArray(int[] shape, int[] stride, int offset, char ordering) {
        super(shape, stride, offset, ordering);
        setupJcuBlas();
    }

    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape    the shape of the ndarray
     * @param stride   the stride of the ndarray
     * @param ordering
     */
    public JCublasComplexNDArray(int[] shape, int[] stride, char ordering) {
        super(shape, stride, ordering);
        setupJcuBlas();
    }

    /**
     * Create the ndarray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the ndarray
     * @param stride the stride of the ndarray
     */
    public JCublasComplexNDArray(int[] shape, int[] stride) {

        super(shape, stride);
        setupJcuBlas();
    }

    /**
     * @param shape
     * @param offset
     */
    public JCublasComplexNDArray(int[] shape, int offset) {

        super(shape, offset);
        setupJcuBlas();
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>ComplexDoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public JCublasComplexNDArray(int newRows, int newColumns) {

        super(newRows, newColumns);
        setupJcuBlas();
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>ComplexDoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     * @param ordering   the ordering of the ndarray
     */
    public JCublasComplexNDArray(int newRows, int newColumns, char ordering) {

        super(newRows, newColumns, ordering);
        setupJcuBlas();
    }

    /**
     * Float overloading for constructor
     *
     * @param data     the data to use
     * @param shape    the shape to use
     * @param stride   the stride of the ndarray
     * @param offset   the offset of the ndarray
     * @param ordering the ordering for the ndarrayg
     */
    public JCublasComplexNDArray(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        super(data, shape, stride, offset, ordering);
        setupJcuBlas();
    }

    public JCublasComplexNDArray(float[] data, int[] shape, int[] stride, int offset) {
        super(data, shape, stride, offset);
        setupJcuBlas();
    }



    public JCublasComplexNDArray(float[] floats, int[] shape, int offset, char ordering) {
        super(floats, shape, offset, ordering);
        setupJcuBlas();
    }

    public JCublasComplexNDArray(float[] floats, int[] shape, int offset) {

        super(floats, shape, offset);
        setupJcuBlas();
    }


    public void getData() {
        JCublas.cublasGetVector(length, Sizeof.FLOAT, pointer, stride[0], Pointer.to(data()), stride[0]);

    }


    public void alloc() {
        JCublas.cublasAlloc(length, Sizeof.FLOAT, pointer);
        JCublas.cublasSetVector(length, Sizeof.FLOAT, dataPointer, majorStride(), pointer, 1);




    }


    protected void setupJcuBlas() {
        if(pointer != null)
            return;
        pointer = new Pointer().withByteOffset(offset());
        dataPointer = Pointer.to(data()).withByteOffset(offset());
    }

    public void free() {
        JCublas.cublasFree(pointer);
    }



    public Pointer pointer() {
        return pointer;
    }

}
