package org.deeplearning4j.linalg;

import jcuda.jcublas.JCublas;
import jcuda.Pointer;
import jcuda.Sizeof;

import org.deeplearning4j.linalg.api.ndarray.SizeException;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.TwoArrayOps;
import org.deeplearning4j.linalg.ops.elementwise.AddOp;
import org.deeplearning4j.linalg.ops.elementwise.DivideOp;
import org.deeplearning4j.linalg.ops.elementwise.MultiplyOp;
import org.deeplearning4j.linalg.ops.elementwise.SubtractOp;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.deeplearning4j.linalg.util.IterationResult;
import org.deeplearning4j.linalg.util.Shape;






import java.io.*;
import java.util.*;

import static org.deeplearning4j.linalg.util.ArrayUtil.calcStrides;
import static org.deeplearning4j.linalg.util.ArrayUtil.reverseCopy;

public class JCublasNDArray implements INDArray {
    private int[] shape;
    private int[] stride;
    private int offset = 0;
    public int rows;
    /** Number of columns. */
    public int columns;
    /** Total number of elements (for convenience). */
    public int length;
    /** The actual data stored by rows (that is, row 0, row 1...). */
    public double[] data = null; // rows are contiguous


    /**
     * Construct an ndarray of the specified shape
     * with an empty data array
     * @param shape the shape of the ndarray
     * @param stride the stride of the ndarray
     * @param offset the desired offset
     */
    public JCublasNDArray(int[] shape,int[] stride,int offset) {this(new double[ArrayUtil.prod(shape)],shape,stride,offset);}
    public JCublasNDArray(int[] shape,int offset) {
        this(shape,calcStrides(shape),offset);
    }
    public JCublasNDArray(int[] shape) { this(shape,0); }
    public JCublasNDArray(int[] shape,int[] stride){
        this(shape,stride,0);
    }
    public JCublasNDArray(double[] data,int[] shape) {
        this(data,shape,0);
    }
    public JCublasNDArray(double[] data,int[] shape,int offset) { this(data, shape, calcStrides(shape), offset); }
    public JCublasNDArray(int newRows, int newColumns) {
        this(newRows, newColumns, new double[newRows * newColumns]);
    }
    public JCublasNDArray(int newRows, int newColumns, double... newData) {
        rows = newRows;
        columns = newColumns;
        length = rows * columns;

        if (newData != null && newData.length != newRows * newColumns) {
            throw new IllegalArgumentException(
                    "Passed data must match matrix dimensions.");
        }

        data = newData;
        //System.err.printf("%d * %d matrix created\n", rows, columns);
    }
    public JCublasNDArray(double[] data,int[] shape,int[] stride,int offset) {
        if(offset >= data.length)
            throw new IllegalArgumentException("Invalid offset: must be < data.length");

        this.offset = offset;
        this.stride = stride;

        initShape(shape);

        if(data != null  && data.length > 0)
            this.data = data;
    }

    public JCublasNDArray divColumnVector(INDArray columnVector) {
        return dup().diviColumnVector(columnVector);
    }

    public JCublasNDArray diviColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).divi(columnVector.getScalar(i));
        }
        return this;
    }

    public JCublasNDArray dup() {
        double[] dupData = new double[data.length];
        System.arraycopy(data,0,dupData,0,dupData.length);
        JCublasNDArray ret = new JCublasNDArray(dupData,shape,stride,offset);
        return ret;
    }


    public JCublasNDArray subColumnVector(INDArray columnVector) {
        return dup().subiColumnVector(columnVector);
    }
    public JCublasNDArray subiColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).subi(columnVector.getScalar(i));
        }
        return this;
    }


    private void initShape(int[] shape) {
        this.shape = shape;

        if(this.shape.length == 1) {
            rows = 1;
            columns = this.shape[0];
        }
        else if(this.shape().length == 2) {
            if(shape[0] == 1) {
                this.shape = new int[1];
                this.shape[0] = shape[1];
                rows = 1;
                columns = shape[1];
            }
            else {
                rows = shape[0];
                columns = shape[1];
            }


        }

        //default row vector
        else if(this.shape.length == 1) {
            columns = this.shape[0];
            rows = 1;
        }



        this.length = ArrayUtil.prod(this.shape);
        if(this.stride == null)
            this.stride = ArrayUtil.calcStrides(this.shape);

        //recalculate stride: this should only happen with row vectors
        if(this.stride.length != this.shape.length) {
            this.stride = ArrayUtil.calcStrides(this.shape);
        }

    }

    /**
     * Number of columns (shape[1]), throws an exception when
     * called when not 2d
     * @return the number of columns in the array (only 2d)
     */
    public int columns() {
        if(isMatrix()) {
            if (shape().length > 2)
                return Shape.squeeze(shape)[1];
            else if (shape().length == 2)
                return shape[1];
        }
        if(isVector()) {
            if(isColumnVector())
                return 1;
            else
                return shape[0];
        }
        throw new IllegalStateException("Unable to getFromOrigin number of of rows for a non 2d matrix");
    }

    /**
     * Flattens the array for linear indexing
     * @return the flattened version of this array
     */
    public JCublasNDArray ravel() {
        JCublasNDArray ret = new JCublasNDArray(new int[]{1,length});
        List<JCublasNDArray> list = new ArrayList<>();
        sliceVectors(list);
        int count = 0;
        for(int i = 0; i < list.size(); i++) {
            for(int j = 0; j < list.get(i).length; j++)
                ret.put(count++,list.get(i).getScalar(j));
        }
        return ret;
    }

    /**
     * Flattens the array for linear indexing
     * @return the flattened version of this array
     */
    private void sliceVectors(java.util.List<JCublasNDArray> list) {
        if(isVector())
            list.add(this);
        else {
            for(int i = 0; i < slices(); i++) {
                slice(i).sliceVectors(list);
            }
        }
    }
    /**
     * Returns the specified slice of this matrix.
     * In matlab, this would be equivalent to (given a 2 x 2 x 2):
     * A(:,:,x) where x is the slice you want to return.
     *
     * The slice is always relative to the final dimension of the matrix.
     *
     * @param slice the slice to return
     * @return the specified slice of this matrix
     */
    public JCublasNDArray slice(int slice) {

        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d NDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1)
            return new JCublasNDArray(data,ArrayUtil.empty(),ArrayUtil.empty(),offset + slice * stride[0]);


            //slice of a matrix is a vector
        else if (shape.length == 2) {
            JCublasNDArray slice2 =  new JCublasNDArray(
                    data,
                    ArrayUtil.of(shape[1]),
                    Arrays.copyOfRange(stride,1,stride.length),
                    offset + slice * stride[0]
            );
            return slice2;

        }

        else
            return new JCublasNDArray(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                    Arrays.copyOfRange(stride, 1, stride.length),
                    offset + (slice * stride[0]));

    }


    /**
     * Returns the slice of this from the specified dimension
     * @param slice the dimension to return from
     * @param dimension the dimension of the slice to return
     * @return the slice of this matrix from the specified dimension
     * and dimension
     */
    public JCublasNDArray slice(int slice, int dimension) {
        if (slice == 0)
            return slice(dimension);
        if (shape.length == 2) {
            if (slice != 1)
                throw new IllegalArgumentException("Unable to retrieve dimension " + slice + " from a 2d array");
            return new JCublasNDArray(data,
                    ArrayUtil.of(shape[0]),
                    ArrayUtil.of(stride[0]),
                    offset + dimension * stride[1]
            );
        }

        return new JCublasNDArray (
                data,
                ArrayUtil.removeIndex(shape,dimension),
                ArrayUtil.removeIndex(stride,dimension),
                offset + dimension * stride[slice]
        );
    }
    public JCublasNDArray get(int[] indices) {
        JCublasNDArray result = new JCublasNDArray(data,new int[]{1,indices.length},stride,offset);

        for (int i = 0; i < indices.length; i++) {
            result.put(i, getScalar(indices[i]));
        }

        return result;
    }

    public JCublasNDArray getScalar(int i) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int idx = linearIndex(i);
        return JCublasNDArray.scalar(data[idx]);
    }

    public int linearIndex(int i) {
        int realStride = getRealStrideForLinearIndex();
        int idx = offset + i * realStride;
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }

    private int getRealStrideForLinearIndex() {
        if(stride == null || stride().length < 1)
            return 1;
        if(stride.length == 2 && shape[0] == 1)
            return stride[1];
        if(stride().length == 2 && shape[1] == 1)
            return stride[0];
        return stride[0];
    }

    public static JCublasNDArray scalar(JCublasNDArray from,int index) {
        return new JCublasNDArray(from.data,new int[]{1},new int[]{1},index);
    }

    public JCublasNDArray addi(INDArray other) {
        return addi(other,this);
    }

    /**
     * in place addition of two matrices
     *
     * @param other  the second ndarray to add
     * @param result the result ndarray
     * @return the result of the addition
     */
    @Override
    public JCublasNDArray addi(INDArray other, INDArray result) {
        new TwoArrayOps().from(this).other(other).op(AddOp.class)
                .to(result).build().exec();
        return (JCublasNDArray) result;
    }

    public static JCublasNDArray scalar(double num) {
        return new JCublasNDArray(new double[]{num},new int[]{1},new int[]{1},0);
    }

    public JCublasNDArray muliRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).muli(rowVector.getScalar(i));
        }
        return this;
    }


    public JCublasNDArray mulRowVector(INDArray rowVector) {
        return dup().muliRowVector(rowVector);
    }

    public JCublasNDArray add(JCublasNDArray other) {
        return addi(other, new JCublasNDArray(rows, columns));
    }
    public JCublasNDArray addiRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).addi(rowVector.getScalar(i));
        }
        return this;
    }

    /**
     * In place addition of a column vector
     *
     * @param rowVector the column vector to add
     * @return the result of the addition
     */
    @Override
    public JCublasNDArray addRowVector(INDArray rowVector) {
        return dup().addiRowVector(rowVector);
    }

    /**
     * Reshape the ndarray in to the specified dimensions,
     * possible errors being thrown for invalid shapes
     * @param shape
     * @return
     */
    public JCublasNDArray reshape(int[] shape) {
        long ec = 1;
        for (int i = 0; i < shape.length; i++) {
            int si = shape[i];
            if (( ec * si ) != (((int) ec ) * si ))
                throw new IllegalArgumentException("Too many elements");
            ec *= shape[i];
        }
        int n = (int) ec;

        if (ec != n)
            throw new IllegalArgumentException("Too many elements");

        JCublasNDArray ndArray = new JCublasNDArray(data,shape,stride,offset);
        return ndArray;

    }

    public boolean multipliesWith(JCublasNDArray a) {
        return columns == a.rows;
    }


    /** Throws SizeException unless matrices can be multiplied with one another. */
    public void assertMultipliesWith(JCublasNDArray a) {
        if (!multipliesWith(a)) {
            throw new SizeException("Number of columns of left matrix must be equal to number of rows of right matrix.");
        }
    }
    public boolean sameSize(JCublasNDArray a) {
        return rows == a.rows && columns == a.columns;
    }
    /** Resize the matrix. All elements will be set to zero. */
    public void resize(int newRows, int newColumns) {
        rows = newRows;
        columns = newColumns;
        length = newRows * newColumns;
        data = new double[rows * columns];
    }
    /** Copy DoubleMatrix a to this. this a is resized if necessary. */
    public JCublasNDArray copy(JCublasNDArray a) {
        if (!sameSize(a)) {
            resize(a.rows, a.columns);
        }

        System.arraycopy(a.data, 0, data, 0, length);
        return a;
    }
    /** Matrix-matrix multiplication (in-place). */
    public JCublasNDArray mmuli(JCublasNDArray other, JCublasNDArray result) {
        JCublasNDArray otherArray = JCublasNDArray.wrap(other);
        JCublasNDArray resultArray = JCublasNDArray.wrap(result);

        if (other.isScalar()) {
            return muli(otherArray.scalar(), resultArray);
        }
        if (isScalar()) {
            return otherArray.muli(scalar(), resultArray);
        }

        /* check sizes and resize if necessary */
        assertMultipliesWith(other);


        if (result == this || result == other) {
            /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
             * allocating a temporary object on the side and copy the result later.
             */
            JCublasNDArray temp = new JCublasNDArray(resultArray.shape(),ArrayUtil.calcStridesFortran(resultArray.shape()));

            if (otherArray.columns() == 1) {
                Pointer d_A = new Pointer();
                Pointer d_B = new Pointer();

                JCublas.cublasSetVector(
                        otherArray.length(),
                        Sizeof.FLOAT,
                        Pointer.to(otherArray.data()),
                        1,
                        d_A,
                        1);
                JCublas.cublasSetVector(
                        length(),
                        Sizeof.FLOAT,
                        Pointer.to(data()),
                        1,
                        d_B,
                        1);

                JCublas.cublasDgemv(
                        'n',
                        otherArray.rows(),
                        otherArray.columns(),
                        1,
                        d_A,
                        1,
                        d_A,
                        1,
                        1,
                        d_B,
                        1);

                JCublas.cublasGetVector(
                        length(),
                        Sizeof.FLOAT,
                        d_B,
                        1,
                        Pointer.to(resultArray.data()),
                        1);

                //NDArrayBlas.gemv(1.0, this, otherArray, 0.0, temp);
            } else {
                Pointer d_A = new Pointer();
                Pointer d_B = new Pointer();
                Pointer d_C = new Pointer();

                JCublas.cublasSetMatrix(
                        columns(),
                        rows(),
                        Sizeof.FLOAT,
                        Pointer.to(data()),
                        1,
                        d_A,
                        1
                        );
                JCublas.cublasSetMatrix(
                        other.columns(),
                        other.rows(),
                        Sizeof.FLOAT,
                        Pointer.to(otherArray.data()),
                        1,
                        d_B,
                        1
                );
                JCublas.cublasSgemm(
                        'n',
                        'n',
                        otherArray.rows(),
                        columns(),
                        otherArray.columns(),
                        1,
                        d_A,
                        1,
                        d_B,
                        1,
                        0,
                        d_C,
                        1);
                JCublas.cublasGetVector(
                        length(),
                        Sizeof.FLOAT,
                        d_B,
                        1,
                        Pointer.to(resultArray.data()),
                        1);
                //NDArrayBlas.gemm(1.0, this, otherArray, 0.0, temp);
            }

            JCublasNDArray.copy(temp, resultArray);


        } else {
            if (otherArray.columns() == 1) {
                Pointer d_A = new Pointer();
                Pointer d_B = new Pointer();

                JCublas.cublasSetVector(
                        otherArray.length(),
                        Sizeof.FLOAT,
                        Pointer.to(otherArray.data()),
                        1,
                        d_A,
                        1);
                JCublas.cublasSetVector(
                        length(),
                        Sizeof.FLOAT,
                        Pointer.to(data()),
                        1,
                        d_B,
                        1);

                JCublas.cublasDgemv(
                        'n',
                        otherArray.rows(),
                        otherArray.columns(),
                        1,
                        d_A,
                        1,
                        d_A,
                        1,
                        1,
                        d_B,
                        1);

                JCublas.cublasGetVector(
                        length(),
                        Sizeof.FLOAT,
                        d_B,
                        1,
                        Pointer.to(resultArray.data()),
                        1);

                //NDArrayBlas.gemv(1.0, this, otherArray, 0.0, resultArray);
            }
            else {
                Pointer d_A = new Pointer();
                Pointer d_B = new Pointer();
                Pointer d_C = new Pointer();

                JCublas.cublasSetMatrix(
                        columns(),
                        rows(),
                        Sizeof.FLOAT,
                        Pointer.to(data()),
                        1,
                        d_A,
                        1
                );
                JCublas.cublasSetMatrix(
                        other.columns(),
                        other.rows(),
                        Sizeof.FLOAT,
                        Pointer.to(otherArray.data()),
                        1,
                        d_B,
                        1
                );
                JCublas.cublasSgemm(
                        'n',
                        'n',
                        otherArray.rows(),
                        columns(),
                        otherArray.columns(),
                        1,
                        d_A,
                        1,
                        d_B,
                        1,
                        0,
                        d_C,
                        1);
                JCublas.cublasGetVector(
                        length(),
                        Sizeof.FLOAT,
                        d_B,
                        1,
                        Pointer.to(resultArray.data()),
                        1);
                //NDArrayBlas.gemm(1.0, this, otherArray, 0.0, resultArray);
            }
        }
        return resultArray;
    }

    public JCublasNDArray mmul(JCublasNDArray a) {
        int[] shape = {rows(),JCublasNDArray.wrap(a).columns()};
        return mmuli(a,new JCublasNDArray(shape));
    }

    /**
     * Wrap toWrap with the specified shape, and dimensions from
     * the passed in ndArray
     * @param ndArray the way to wrap a matrix
     * @param toWrap the matrix to wrap
     * @return the wrapped matrix
     */
    public static JCublasNDArray wrap(JCublasNDArray ndArray,JCublasNDArray toWrap) {
        if(toWrap instanceof JCublasNDArray)
            return (JCublasNDArray) toWrap;
        int[] stride = ndArray.stride();
        JCublasNDArray ret = new JCublasNDArray(toWrap.data,ndArray.shape(),stride,ndArray.offset());
        return ret;
    }


    /**
     * Wrap a matrix in to an ndarray
     * @param toWrap the matrix to wrap
     * @return the wrapped matrix
     */
    public static JCublasNDArray wrap(JCublasNDArray toWrap) {
        if(toWrap instanceof JCublasNDArray)
            return (JCublasNDArray) toWrap;
        int[]  shape = new int[]{toWrap.rows,toWrap.columns};
        JCublasNDArray ret = new JCublasNDArray(toWrap.data,shape);
        return ret;
    }

}
