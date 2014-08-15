package org.deeplearning4j.linalg;

import jcuda.jcublas.JCublas;
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
    private void sliceVectors(List<JCublasNDArray> list) {
        if(isVector())
            list.add(this);
        else {
            for(int i = 0; i < slices(); i++) {
                slice(i).sliceVectors(list);
            }
        }
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

    private int linearIndex(int i) {
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
}
