package org.deeplearning4j.linalg.jcublas.complex;

import org.deeplearning4j.linalg.jcublas.complex.ComplexDouble;
import org.deeplearning4j.linalg.jcublas.complex.ComplexJCublasNDArray;
import org.deeplearning4j.linalg.jcublas.util.MatrixUtil;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.deeplearning4j.linalg.util.ComplexUtil;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * ComplexNDArray operations
 *
 * @author Adam Gibson
 */
public class ComplexJCublasNDArrayUtil {
    private static Logger log = LoggerFactory.getLogger(ComplexJCublasNDArrayUtil.class);

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


    public static ComplexJCublasNDArray exp(ComplexJCublasNDArray toExp) {
        return expi(toExp.dup());
    }

    /**
     * Returns the exponential of a complex ndarray
     *
     * @param toExp the ndarray to convert
     * @return the exponential of the specified
     * ndarray
     */
    public static ComplexJCublasNDArray expi(ComplexJCublasNDArray toExp) {
        ComplexJCublasNDArray flattened = toExp.ravel();
        for (int i = 0; i < flattened.length; i++) {
            ComplexDouble d = (ComplexDouble) ComplexUtil.exp(flattened.get(i));
            flattened.put(i, d);
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
    public static ComplexJCublasNDArray center(ComplexJCublasNDArray arr, int[] shape) {
        if (arr.length < ArrayUtil.prod(shape))
            return arr;

        DoubleMatrix shapeMatrix = MatrixUtil.toMatrix(shape);
        DoubleMatrix currShape = MatrixUtil.toMatrix(arr.shape());

        DoubleMatrix startIndex = currShape.sub(shapeMatrix).div(2);
        DoubleMatrix endIndex = startIndex.add(shapeMatrix);
        if (shapeMatrix.length > 1)
            arr = ComplexJCublasNDArray.wrap(arr.get(RangeUtils.interval((int) startIndex.get(0), (int) endIndex.get(0)), RangeUtils.interval((int) startIndex.get(1), (int) endIndex.get(1))));
        else {
            ComplexJCublasNDArray ret = new ComplexJCublasNDArray(new int[]{(int) shapeMatrix.get(0)});
            int start = (int) startIndex.get(0);
            int end = (int) endIndex.get(0);
            int count = 0;
            for (int i = start; i < end; i++) {
                ret.put(count++, arr.get(i));
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
    public static ComplexJCublasNDArray truncate(ComplexJCublasNDArray nd, int n, int dimension) {


        if (nd.isVector()) {
            ComplexJCublasNDArray truncated = new ComplexJCublasNDArray(new int[]{n});
            for (int i = 0; i < n; i++)
                truncated.put(i, nd.get(i));
            return truncated;
        }


        if (nd.size(dimension) > n) {
            int[] targetShape = ArrayUtil.copy(nd.shape());
            targetShape[dimension] = n;
            int numRequired = ArrayUtil.prod(targetShape);
            if (nd.isVector()) {
                ComplexJCublasNDArray ret = new ComplexJCublasNDArray(targetShape);
                int count = 0;
                for (int i = 0; i < nd.length; i += nd.stride()[dimension]) {
                    ret.put(count++, nd.get(i));

                }
                return ret;
            } else if (nd.isMatrix()) {
                List<ComplexDouble> list = new ArrayList<>();
                //row
                if (dimension == 0) {
                    for (int i = 0; i < nd.rows(); i++) {
                        ComplexJCublasNDArray row = nd.getRow(i);
                        for (int j = 0; j < row.length; j++) {
                            if (list.size() == numRequired)
                                return new ComplexJCublasNDArray(list.toArray(new ComplexDouble[0]), targetShape);

                            list.add(row.get(j));
                        }
                    }
                } else if (dimension == 1) {
                    for (int i = 0; i < nd.columns(); i++) {
                        ComplexJCublasNDArray row = nd.getColumn(i);
                        for (int j = 0; j < row.length; j++) {
                            if (list.size() == numRequired)
                                return new ComplexJCublasNDArray(list.toArray(new ComplexDouble[0]), targetShape);

                            list.add(row.get(j));
                        }
                    }
                } else
                    throw new IllegalArgumentException("Illegal dimension for matrix " + dimension);


                return new ComplexJCublasNDArray(list.toArray(new ComplexDouble[0]), targetShape);

            }


            if (dimension == 0) {
                List<ComplexJCublasNDArray> slices = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    ComplexJCublasNDArray slice = nd.slice(i);
                    slices.add(slice);
                }

                return new ComplexJCublasNDArray(slices, targetShape);

            } else {
                List<ComplexDouble> list = new ArrayList<>();
                int numElementsPerSlice = ArrayUtil.prod(ArrayUtil.removeIndex(targetShape, 0));
                for (int i = 0; i < nd.slices(); i++) {
                    ComplexJCublasNDArray slice = nd.slice(i).ravel();
                    for (int j = 0; j < numElementsPerSlice; j++)
                        list.add(slice.get(j));
                }

                assert list.size() == ArrayUtil.prod(targetShape) : "Illegal shape for length " + list.size();

                return new ComplexJCublasNDArray(list.toArray(new ComplexDouble[0]), targetShape);

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
    public static ComplexJCublasNDArray padWithZeros(ComplexJCublasNDArray nd, int[] targetShape) {
        if (Arrays.equals(nd.shape(), targetShape))
            return nd;
        //no padding required
        if (ArrayUtil.prod(nd.shape()) >= ArrayUtil.prod(targetShape))
            return nd;

        ComplexJCublasNDArray ret = new ComplexJCublasNDArray(targetShape);
        System.arraycopy(nd.data, 0, ret.data, 0, nd.data.length);
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
    public static ComplexJCublasNDArray doSliceWise(MatrixOp op, ComplexJCublasNDArray arr) {
        int columns = isColumnOp(op) ? arr.columns() : arr.rows();
        int[] shape = {arr.slices(), columns};

        ComplexJCublasNDArray ret = new ComplexJCublasNDArray(shape);

        for (int i = 0; i < arr.slices(); i++) {
            switch (op) {
                case COLUMN_SUM:
                    ret.putSlice(i, arr.slice(i).columnSums());
                    break;
                case COLUMN_MEAN:
                    ret.putSlice(i, arr.slice(i).columnMeans());
                    break;
                case ROW_SUM:
                    ret.putSlice(i, arr.slice(i).rowSums());
                    break;
                case ROW_MEAN:
                    ret.putSlice(i, arr.slice(i).rowMeans());
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
    public static ComplexDouble doSliceWise(ScalarOp op, ComplexJCublasNDArray arr) {
        arr = arr.reshape(new int[]{1,arr.length});

        if(op == ScalarOp.NORM_1) {
            return new ComplexDouble(ComplexJCublasNDArray.asum(arr));
        }

        else if(op == ScalarOp.NORM_2) {
            return new ComplexDouble(ComplexJCublasNDArray.nrm2(arr));

        }

        else if(op == ScalarOp.NORM_MAX) {
            int i = ComplexJCublasNDArray.iamax(arr);
            return arr.unSafeGet(i);
        }

        ComplexDouble s = new ComplexDouble(0.0);
        for (int i = 0; i < arr.length; i++) {
            org.jblas.ComplexDouble curr = arr.get(i);

            switch (op) {
                case SUM:
                    s.addi(curr);
                    break;
                case MEAN:
                    s.addi(curr);
                    break;
                case MAX:
                    if (arr.get(i).absoluteValue() > s.abs())
                        s.set(arr.get(i).realComponent(), arr.get(i).imaginaryComponent());
                    break;
                case MIN:
                    if (arr.get(i).absoluteValue() < s.abs())
                        s.set(arr.get(i).realComponent(), arr.get(i).imaginaryComponent());
                case PROD:
                    s.muli(curr);
                    break;


            }


        }

        if(op == ScalarOp.MEAN)
            s.divi(arr.length);


        return s;
    }



}
