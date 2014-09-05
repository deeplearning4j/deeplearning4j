package org.nd4j.linalg.util;


import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * IComplexNDArray operations
 *
 * @author Adam Gibson
 */
public class ComplexNDArrayUtil {

    private static Logger log = LoggerFactory.getLogger(ComplexNDArrayUtil.class);

    public static enum ScalarOp {
        SUM,
        MEAN,
        PROD,
        MAX,
        MIN,
        ARG_MAX,
        ARG_MIN,
        NORM_2,
        NORM_1,
        NORM_MAX
    }


    public static enum DimensionOp {
        SUM,
        MEAN,
        PROD,
        MAX,
        MIN,
        ARG_MIN,
        NORM_2,
        NORM_1,
        NORM_MAX,
        FFT
    }


    public static enum MatrixOp {
        COLUMN_MIN,
        COLUMN_MAX,
        COLUMN_SUM,
        COLUMN_MEAN,
        ROW_MIN,
        ROW_MAX,
        ROW_SUM,
        ROW_MEAN
    }


    public static IComplexNDArray exp(IComplexNDArray toExp) {
        return expi(toExp.dup());
    }

    /**
     * Returns the exponential of a complex ndarray
     *
     * @param toExp the ndarray to convert
     * @return the exponential of the specified
     * ndarray
     */
    public static IComplexNDArray expi(IComplexNDArray toExp) {
        IComplexNDArray flattened = toExp.ravel();
        for (int i = 0; i < flattened.length(); i++) {
            IComplexNumber n = (IComplexNumber) flattened.getScalar(i).element();
            flattened.put(i, Nd4j.scalar(ComplexUtil.exp(n)));
        }
        return flattened.reshape(toExp.shape());
    }


    /**
     * Center an array
     *
     * @param arr   the arr to center
     * @param shape the shape of the array
     * @return the center portion of the array based on the
     * specified shape
     */
    public static IComplexNDArray center(IComplexNDArray arr, int[] shape) {
        if (arr.length() < ArrayUtil.prod(shape))
            return arr;

        INDArray shapeMatrix = ArrayUtil.toNDArray(shape);
        INDArray currShape =  ArrayUtil.toNDArray(arr.shape());

        INDArray startIndex = currShape.sub(shapeMatrix).divi(Nd4j.scalar(2));
        INDArray endIndex = startIndex.add(shapeMatrix);
        if (shapeMatrix.length() > 1) {
            //arr = arr.get(RangeUtils.interval((int) startIndex.getScalar(0).element(), (int) endIndex.getScalar(0).element()), RangeUtils.interval((int) startIndex.getScalar(1).element(), (int) endIndex.getScalar(1).element()));
            arr = Nd4j.createComplex(arr.get(NDArrayIndex.interval((int) startIndex.getScalar(0).element(), (int) endIndex.getScalar(0).element()), NDArrayIndex.interval((int) startIndex.getScalar(1).element(), (int) endIndex.getScalar(1).element())));
        }
        else {
            IComplexNDArray ret = Nd4j.createComplex(new int[]{(int) shapeMatrix.getScalar(0).element()});
            int start = (int) startIndex.getScalar(0).element();
            int end = (int) endIndex.getScalar(0).element();
            int count = 0;
            for (int i = start; i < end; i++) {
                ret.put(count++, arr.getScalar(i));
            }
        }


        return arr;
    }


    /**
     * Truncates an ndarray to the specified shape.
     * If the shape is the same or greater, it just returns
     * the original array
     *
     * @param nd the ndarray to truncate
     * @param n  the number of elements to truncate to
     * @return the truncated ndarray
     */
    public static IComplexNDArray truncate(IComplexNDArray nd, int n, int dimension) {


        if (nd.isVector()) {
            IComplexNDArray truncated = Nd4j.createComplex(new int[]{n});
            for (int i = 0; i < n; i++)
                truncated.put(i, nd.getScalar(i));
            return truncated;
        }


        if (nd.size(dimension) > n) {
            int[] targetShape = ArrayUtil.copy(nd.shape());
            targetShape[dimension] = n;
            int numRequired = ArrayUtil.prod(targetShape);
            if (nd.isVector()) {
                IComplexNDArray ret = Nd4j.createComplex(targetShape);
                int count = 0;
                for (int i = 0; i < nd.length(); i += nd.stride()[dimension]) {
                    ret.put(count++, nd.getScalar(i));

                }
                return ret;
            } else if (nd.isMatrix()) {
                List<IComplexDouble> list = new ArrayList<>();
                //row
                if (dimension == 0) {
                    for (int i = 0; i < nd.rows(); i++) {
                        IComplexNDArray row = nd.getRow(i);
                        for (int j = 0; j < row.length(); j++) {
                            if (list.size() == numRequired)
                                return Nd4j.createComplex(list.toArray(new IComplexDouble[0]), targetShape);

                            list.add((IComplexDouble) row.getScalar(j).element());
                        }
                    }
                } else if (dimension == 1) {
                    for (int i = 0; i < nd.columns(); i++) {
                        IComplexNDArray row = nd.getColumn(i);
                        for (int j = 0; j < row.length(); j++) {
                            if (list.size() == numRequired)
                                return Nd4j.createComplex(list.toArray(new IComplexDouble[0]), targetShape);

                            list.add((IComplexDouble) row.getScalar(j).element());
                        }
                    }
                } else
                    throw new IllegalArgumentException("Illegal dimension for matrix " + dimension);


                return Nd4j.createComplex(list.toArray(new IComplexDouble[0]), targetShape);

            }


            if (dimension == 0) {
                List<IComplexNDArray> slices = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    IComplexNDArray slice = nd.slice(i);
                    slices.add(slice);
                }

                return Nd4j.createComplex(slices, targetShape);

            } else {
                List<IComplexDouble> list = new ArrayList<>();
                int numElementsPerSlice = ArrayUtil.prod(ArrayUtil.removeIndex(targetShape, 0));
                for (int i = 0; i < nd.slices(); i++) {
                    IComplexNDArray slice = nd.slice(i).ravel();
                    for (int j = 0; j < numElementsPerSlice; j++)
                        list.add((IComplexDouble) slice.getScalar(j).element());
                }

                assert list.size() == ArrayUtil.prod(targetShape) : "Illegal shape for length " + list.size();

                return Nd4j.createComplex(list.toArray(new IComplexDouble[0]), targetShape);

            }


        }

        return nd;

    }


    /**
     * Pads an ndarray with zeros
     *
     * @param nd          the ndarray to pad
     * @param targetShape the the new shape
     * @return the padded ndarray
     */
    public static IComplexNDArray padWithZeros(IComplexNDArray nd, int[] targetShape) {
        if (Arrays.equals(nd.shape(), targetShape))
            return nd;
        //no padding required
        if (ArrayUtil.prod(nd.shape()) >= ArrayUtil.prod(targetShape))
            return nd;

        IComplexNDArray ret = Nd4j.createComplex(targetShape);
        System.arraycopy(nd.data(), 0, ret.data(), 0, nd.data().length);
        return ret;

    }


    private static boolean isRowOp(MatrixOp op) {
        return
                op == MatrixOp.ROW_MIN ||
                        op == MatrixOp.ROW_MAX ||
                        op == MatrixOp.ROW_SUM ||
                        op == MatrixOp.ROW_MEAN;
    }


    private static boolean isColumnOp(MatrixOp op) {
        return
                op == MatrixOp.COLUMN_MIN ||
                        op == MatrixOp.COLUMN_MAX ||
                        op == MatrixOp.COLUMN_SUM ||
                        op == MatrixOp.COLUMN_MEAN;
    }





    /**
     * Does slice wise ops on matrices and
     * returns the aggregate results in one matrix
     *
     * @param op  the operation to perform
     * @param arr the array  to do operations on
     * @return the slice wise operations
     */
    public static IComplexNDArray doSliceWise(MatrixOp op, IComplexNDArray arr) {
        int columns = isColumnOp(op) ? arr.columns() : arr.rows();
        int[] shape = {arr.slices(), columns};

        IComplexNDArray ret = Nd4j.createComplex(shape);

        for (int i = 0; i < arr.slices(); i++) {
            switch (op) {
                case COLUMN_SUM:
                    ret.putSlice(i, arr.slice(i).sum(1));
                    break;
                case COLUMN_MEAN:
                    ret.putSlice(i, arr.slice(i).mean(1));
                    break;
                case ROW_SUM:
                    ret.putSlice(i, arr.slice(i).sum(0));
                    break;
                case ROW_MEAN:
                    ret.putSlice(i, arr.slice(i).mean(0));
                    break;
            }
        }


        return ret;
    }


    /**
     * Execute an element wise operation over the whole array
     * @param op the operation to execute
     * @param arr the array to perform operations on
     * @return the result over the whole nd array
     */
    public static Object doSliceWise(ScalarOp op, IComplexNDArray arr) {
        arr = arr.reshape(new int[]{1,arr.length()});

        if(op == ScalarOp.NORM_1) {
            return Nd4j.getBlasWrapper().asum(arr);
        }

        else if(op == ScalarOp.NORM_2) {
            return (Nd4j.getBlasWrapper().nrm2(arr));

        }

        else if(op == ScalarOp.NORM_MAX) {
            int i = Nd4j.getBlasWrapper().iamax(arr);
            return arr.getScalar(i).element();
        }

        IComplexDouble s = Nd4j.createDouble(0.0, 0);
        for (int i = 0; i < arr.length(); i++) {
            IComplexDouble curr = (IComplexDouble) arr.getScalar(i).element();

            switch (op) {
                case SUM:
                    s.addi(curr);
                    break;
                case MEAN:
                    s.addi(curr);
                    break;
                case MAX:
                    if (curr.absoluteValue().doubleValue() > s.absoluteValue().doubleValue())
                        s.set(curr.realComponent().doubleValue(), curr.imaginaryComponent().doubleValue());
                    break;
                case MIN:
                    if (curr.absoluteValue().doubleValue() < s.absoluteValue().doubleValue())
                        s.set(curr.realComponent().doubleValue(), curr.imaginaryComponent().doubleValue());
                case PROD:
                    s.muli(curr);
                    break;


            }


        }

        if(op == ScalarOp.MEAN)
            s.divi(arr.length());


        return s;
    }



}
