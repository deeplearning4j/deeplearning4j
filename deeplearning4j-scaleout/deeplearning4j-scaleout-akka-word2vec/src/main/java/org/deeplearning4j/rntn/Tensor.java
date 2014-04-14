package org.deeplearning4j.rntn;

import org.apache.commons.math3.distribution.RealDistribution;
import org.jblas.DoubleMatrix;

import java.io.Serializable;

/**
 * Tensor represents a set of matrices of all the same dimensions.
 * Based on the Recursive Neural Tensor Network by Socher et. al
 * @author Adam Gibson
 */
public class Tensor implements Serializable {

    private DoubleMatrix[] slices;
    private int rows,cols;


    public Tensor(int rows, int columns,int slices) {
        this.rows = rows;
        this.cols = columns;
        this.slices = new DoubleMatrix[slices];

        for(int i = 0; i < slices; i++) {
            this.slices[i] = DoubleMatrix.zeros(rows,columns);
        }
    }

    public Tensor(DoubleMatrix[] slices) {
        ensureSameSize(slices);
        this.slices = new DoubleMatrix[slices.length];
        for(int i = 0; i < slices.length; i++) {
            this.slices[i] = slices[i].dup();
        }

    }

    private void ensureSameSize(DoubleMatrix[] slices) {
        if(slices == null || slices.length < 1)
            throw new IllegalArgumentException("Illegal argument, please pass in slices array >= length 1");
        this.rows = slices[0].rows;
        this.cols = slices[0].columns;
        for(int i = 1; i < slices.length; i++) {
            if(slices[i].rows != rows || slices[i].columns != cols)
                throw new IllegalArgumentException("All slices must be of the same number of rows and columns");
        }
    }


    public DoubleMatrix getSlice(int index) {
        return slices[index];
    }

    public void setSlice(int index,DoubleMatrix slice) {
       if(slice.rows != rows || slice.columns != cols)
            throw new IllegalArgumentException("Illegal matrix passed in, must be of same dimenions as specified slices");

        slices[index] = slice;
    }

    public double sum() {
        double sum = 0.0;
        for(int i = 0;i < slices(); i++)
            sum += slices[i].sum();
        return sum;
    }


    /**
     * Returns a column vector where each entry is the nth bilinear
     * product of the nth slices of the two tensors.
     */
    public DoubleMatrix bilinearProducts(DoubleMatrix in) {
        if (in.columns != 1) {
            throw new AssertionError("Expected a column vector");
        }
        if (in.rows != columns()) {
            throw new AssertionError("Number of rows in the input does not match number of columns in tensor");
        }
        if (rows != cols) {
            throw new AssertionError("Can only perform this operation on a SimpleTensor with square slices");
        }

        DoubleMatrix inT = in.transpose();
        DoubleMatrix out = new DoubleMatrix(slices.length, 1);
        for (int slice = 0; slice < slices.length; ++slice) {
            double result = inT.mul(slices[slice]).mul(in).get(0);
            out.put(slice, result);
        }

        return out;
    }

    /**
     * Returns a random tensor sampling from the given distribution
     * @param rows the number of rows
     * @param cols the number of columns
     * @param slices the slices
     * @param dist the distribution to sample from
     * @return the tensor with the specified slices and the random matrices
     */
    public Tensor rand(int rows, int cols,int slices,RealDistribution dist) {
        Tensor t = new Tensor(rows,cols,slices);
        for(int i = 0; i < slices; i++) {
            DoubleMatrix d = new DoubleMatrix(rows,cols);
            for(int j = 0; j < d.length; j++)
                d.put(i,dist.sample());
            t.slices[i] = d;
        }

        return t;

    }

    /**
     * Element wise addition of each slice of the passed in tensor
     * @param tensor the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public Tensor add(Tensor tensor) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].add(tensor.slices[i]);
        return t;
    }

    /**
     * Element wise multiplication of each slice of the passed in tensor
     * @param tensor the tensor to multiply by
     * @return the element wise multiplication of the passed in tensor
     * with this tensor
     */
    public Tensor mul(Tensor tensor) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].mul(tensor.slices[i]);
        return t;
    }

    public Tensor scale(double value) {
        Tensor t = new Tensor(rows,columns(),slices());
        for(int i = 0;i < slices(); i++)
            t.slices[i] = slices[i].mul(value);
        return t;
    }


    public void assign(double val) {
        for(int i = 0; i < slices.length; i++)
            slices[i] = DoubleMatrix.zeros(rows,columns()).add(val);
    }


    public int numElements() {
        return rows * cols * slices.length;
    }


    public int columns() {
       return cols;
    }

    public int rows() {
        return rows;
    }

    public int slices() {
        return slices.length;
    }




}
