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

    public JCublasNDArray(double[] data,int[] shape,int[] stride,int offset) {
        if(offset >= data.length)
            throw new IllegalArgumentException("Invalid offset: must be < data.length");



        this.offset = offset;
        this.stride = stride;

        initShape(shape);

        if(data != null  && data.length > 0)
            this.data = data;
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
}
