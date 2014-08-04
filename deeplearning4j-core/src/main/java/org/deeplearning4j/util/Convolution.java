package org.deeplearning4j.util;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.fft.*;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.jblas.*;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



/**
 * Convolution is the code for applying the convolution operator.
 *  http://www.inf.ufpr.br/danielw/pos/ci724/20102/HIPR2/flatjavasrc/Convolution.java
 *
 * @author Adam Gibson
 */
public class Convolution {

    private static Logger log = LoggerFactory.getLogger(Convolution.class);

    /**
     *
     *
     * Default no-arg constructor.
     */
    private Convolution() {
    }

    public static enum Type {
        FULL,VALID,SAME
    }


    //range for the convolution to return
    private static Pair<Integer,Integer> rangeFor(int m,int n,Type type) {
        switch (type) {
            case SAME:
                return new Pair<>((int) Math.ceil(n / 2), m);
            case FULL:
                return new Pair<>(0, m + n);
            case VALID:
                return new Pair<>(m, n);
            default:
                throw new IllegalStateException("This should never happen");
        }

    }


    /**
     * ND Convolution
     * @param input the input to transform
     * @param kernel the kernel to transform with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    public static NDArray convn(NDArray input,NDArray kernel,Type type) {
       //        ret = ifftn(fftn(in1, fshape) * fftn(in2, fshape))[fslice].copy()
        if(kernel.isScalar() && input.isScalar())
            return kernel.mul(input);
        DoubleMatrix shape = MatrixUtil.toMatrix(input.shape()).add(MatrixUtil.toMatrix(kernel.shape())).subi(1);
        int[] intShape = MatrixUtil.toInts(shape);
        int[] axes = ArrayUtil.range(0,intShape.length);

        ComplexNDArray ret = FFT.rawifftn(FFT.rawfftn(new ComplexNDArray(input),intShape,axes).muli(FFT.rawfftn(new ComplexNDArray(kernel), intShape, axes)),intShape,axes);


        switch(type) {
            case FULL:
                return ret.getReal();
            case SAME:
                return ComplexNDArrayUtil.center(ret,input.shape()).getReal();
            case VALID:
                return ComplexNDArrayUtil.center(ret,MatrixUtil.toInts(MatrixUtil.toMatrix(input.shape()).sub(MatrixUtil.toMatrix(kernel.shape())).addi(1))).getReal();

        }


        return ret.getReal();
    }




    /**
     * ND Convolution
     * @param input the input to transform
     * @param kernel the kernel to transform with
     * @param type the type of convolution
     * @return the convolution of the given input and kernel
     */
    public static ComplexNDArray convn(ComplexNDArray input,ComplexNDArray kernel,Type type) {

        if(kernel.isScalar() && input.isScalar())
            return kernel.mul(input);

        DoubleMatrix shape = MatrixUtil.toMatrix(input.shape()).add(MatrixUtil.toMatrix(kernel.shape())).subi(1);
        int[] intShape = MatrixUtil.toInts(shape);
        int[] axes = ArrayUtil.range(0,intShape.length);

        ComplexNDArray ret = FFT.rawifftn(FFT.rawfftn(input,intShape,axes).muli(FFT.rawfftn(kernel,intShape,axes)),intShape,axes);


        switch(type) {
            case FULL:
                 return ret;
            case SAME:
                return ComplexNDArrayUtil.center(ret,input.shape());
            case VALID:
                return ComplexNDArrayUtil.center(ret,MatrixUtil.toInts(MatrixUtil.toMatrix(input.shape()).sub(MatrixUtil.toMatrix(kernel.shape())).addi(1)));

        }

        return ret;
    }





    public static DoubleMatrix conv2d(DoubleMatrix input,DoubleMatrix kernel,Type type) {

        DoubleMatrix xShape = new DoubleMatrix(1,2);
        xShape.put(0,input.rows);
        xShape.put(1,input.columns);


        DoubleMatrix yShape = new DoubleMatrix(1,2);
        yShape.put(0,kernel.rows);
        yShape.put(1,kernel.columns);


        DoubleMatrix zShape = xShape.addi(yShape).subi(1);
        int retRows = (int) zShape.get(0);
        int retCols = (int) zShape.get(1);

        ComplexDoubleMatrix fftInput = complexDisceteFourierTransform(new ComplexDoubleMatrix(input), retRows, retCols);
        ComplexDoubleMatrix fftKernel = complexDisceteFourierTransform(new ComplexDoubleMatrix(kernel), retRows, retCols);
        ComplexDoubleMatrix mul = fftKernel.muli(fftInput);
        ComplexDoubleMatrix retComplex = complexInverseDisceteFourierTransform(mul,mul.rows,mul.columns);
        DoubleMatrix ret = retComplex.getReal();

        if(type == Type.VALID) {

            DoubleMatrix validShape = xShape.subi(yShape).addi(1);

            DoubleMatrix start = zShape.subi(validShape).divi(2);
            DoubleMatrix end = start.addi(validShape);
            if(start.get(0) < 1 || start.get(1) < 1)
                throw new IllegalStateException("Illegal row index " + start);
            if(end.get(0) < 1 || end.get(1) < 1)
                throw new IllegalStateException("Illegal column index " + end);

            ret = ret.get(RangeUtils.interval((int) start.get(0),(int) end.get(0)),RangeUtils.interval((int) start.get(1),(int) end.get(1)));




        }

        return ret;
    }





    /**
     * Discrete fourier transform 2d
     * @param input the input to transform
     * @param rows the number of rows in the transformed output matrix
     * @param cols the number of columns in the transformed output matrix
     * @return the discrete fourier transform of the input
     */
    public static ComplexDoubleMatrix complexDisceteFourierTransform(ComplexDoubleMatrix input,int rows,int cols) {
        ComplexDoubleMatrix base;

        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = input.dup();
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }
        else
            base = input.dup();

        ComplexDoubleMatrix temp = new ComplexDoubleMatrix(base.rows,base.columns);
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexDoubleMatrix column = base.getColumn(i);
            temp.putColumn(i,new VectorFFT(column.length).apply((ComplexNDArray.wrap(column))));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexDoubleMatrix row = temp.getRow(i);
            ret.putRow(i,new VectorFFT(row.length).apply((ComplexNDArray.wrap(row))));
        }
        return ret;

    }




    /**
     * Performs an inverse discrete fourier transform with the solution
     * being the number of rows and number of columns.
     * See matlab's iftt2 for more examples
     * @param input the input to transform
     * @param rows the number of rows for the transform
     * @param cols the number of columns for the transform
     * @return the 2d inverse discrete fourier transform
     */
    public static ComplexDoubleMatrix complexInverseDisceteFourierTransform(ComplexDoubleMatrix input,int rows,int cols) {
        ComplexDoubleMatrix base;

        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = input.dup();
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }

        else
            base = input.dup();

        ComplexDoubleMatrix temp = new ComplexDoubleMatrix(base.rows,base.columns);
        ComplexDoubleMatrix ret = new ComplexDoubleMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexDoubleMatrix column = base.getColumn(i);
            temp.putColumn(i,new VectorIFFT(column.length).apply(ComplexNDArray.wrap(column)));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexDoubleMatrix row = temp.getRow(i);
            ret.putRow(i,new VectorIFFT(row.length).apply(ComplexNDArray.wrap(row)));
        }

        return ret;
    }




    public static FloatMatrix conv2d(FloatMatrix input,FloatMatrix kernel,Type type) {

        FloatMatrix xShape = new FloatMatrix(1,2);
        xShape.put(0,input.rows);
        xShape.put(1,input.columns);


        FloatMatrix yShape = new FloatMatrix(1,2);
        yShape.put(0,kernel.rows);
        yShape.put(1,kernel.columns);


        FloatMatrix zShape = xShape.add(yShape).sub(1);
        int retRows = (int) zShape.get(0);
        int retCols = (int) zShape.get(1);

        ComplexFloatMatrix fftInput = complexDisceteFourierTransform(input, retRows, retCols);
        ComplexFloatMatrix fftKernel = complexDisceteFourierTransform(kernel, retRows, retCols);
        ComplexFloatMatrix mul = fftKernel.mul(fftInput);
        ComplexFloatMatrix retComplex = complexInverseDisceteFourierTransform(mul,mul.rows,mul.columns);

        FloatMatrix ret = retComplex.getReal();

        if(type == Type.VALID) {

            FloatMatrix validShape = xShape.subi(yShape).add(1);

            FloatMatrix start = zShape.sub(validShape).div(2);
            FloatMatrix end = start.add(validShape);
            if(start.get(0) < 1 || start.get(1) < 1)
                throw new IllegalStateException("Illegal row index " + start);
            if(end.get(0) < 1 || end.get(1) < 1)
                throw new IllegalStateException("Illegal column index " + end);

            ret = ret.get(RangeUtils.interval((int) start.get(0),(int) end.get(0)),RangeUtils.interval((int) start.get(1),(int) end.get(1)));




        }

        return ret;
    }



    public static ComplexFloatMatrix complexInverseDisceteFourierTransform(ComplexFloatMatrix input,int rows,int cols) {
        ComplexFloatMatrix base;

        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.padWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = input.dup();
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }

        else
            base = input.dup();

        ComplexFloatMatrix temp = new ComplexFloatMatrix(base.rows,base.columns);
        ComplexFloatMatrix ret = new ComplexFloatMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexFloatMatrix column = base.getColumn(i);
            temp.putColumn(i,new VectorFloatIFFT(column.length).apply(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexFloatMatrix row = temp.getRow(i);
            ret.putRow(i,new VectorFloatIFFT(row.length).apply(row));
        }

        return ret;

    }





    /**
     * Performs an inverse discrete fourier transform with the solution
     * being the number of rows and number of columns.
     * See matlab's iftt2 for more examples
     * @param input the input to transform
     * @param rows the number of rows for the transform
     * @param cols the number of columns for the transform
     * @return the 2d inverse discrete fourier transform
     */
    public static ComplexFloatMatrix complexInverseDisceteFourierTransform(FloatMatrix input,int rows,int cols) {
        ComplexFloatMatrix base = null;
        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = new ComplexFloatMatrix(input);
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }

        else
            base = new ComplexFloatMatrix(input);

        ComplexFloatMatrix temp = new ComplexFloatMatrix(base.rows,base.columns);
        ComplexFloatMatrix ret = new ComplexFloatMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexFloatMatrix column = base.getColumn(i);
            temp.putColumn(i,new VectorFloatIFFT(column.length).apply(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexFloatMatrix row = temp.getRow(i);
            ret.putRow(i,new VectorFloatIFFT(row.length).apply(row));
        }
        return ret;
    }

    /**
     * Discrete fourier transform 2d
     * @param input the input to transform
     * @param rows the number of rows in the transformed output matrix
     * @param cols the number of columns in the transformed output matrix
     * @return the discrete fourier transform of the input
     */
    public static ComplexFloatMatrix complexDisceteFourierTransform(FloatMatrix input,int rows,int cols) {
        ComplexFloatMatrix base;

        //pad
        if(input.rows < rows || input.columns < cols)
            base = MatrixUtil.complexPadWithZeros(input, rows, cols);
            //truncation
        else if(input.rows > rows || input.columns > cols) {
            base = new ComplexFloatMatrix(input);
            base = base.get(MatrixUtil.toIndices(RangeUtils.interval(0,rows)),MatrixUtil.toIndices(RangeUtils.interval(0,cols)));
        }
        else
            base = new ComplexFloatMatrix(input);

        ComplexFloatMatrix temp = new ComplexFloatMatrix(base.rows,base.columns);
        ComplexFloatMatrix ret = new ComplexFloatMatrix(base.rows,base.columns);
        for(int i = 0; i < base.columns; i++) {
            ComplexFloatMatrix column = base.getColumn(i);
            temp.putColumn(i,new VectorFloatFFT(column.length).apply(column));
        }

        for(int i = 0; i < ret.rows; i++) {
            ComplexFloatMatrix row = temp.getRow(i);
            ret.putRow(i,new VectorFloatFFT(row.length).apply(row));
        }
        return ret;

    }


}
