package org.deeplearning4j.linalg.jblas.util;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.jblas.NDArray;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *  Basic NDArray ops
 *
 *  @author Adam Gibson
 */
public class NDArrayUtil {
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


    public static INDArray exp(NDArray toExp) {
        return expi(toExp.dup());
    }

    /**
     * Returns an exponential version of this ndarray
     * @param toExp the ndarray to convert
     * @return the converted ndarray
     */
    public static INDArray expi(NDArray toExp) {
        INDArray flattened = toExp.ravel();
        for(int i = 0; i < flattened.length(); i++)
            flattened.put(i,NDArray.scalar(Math.exp((double) flattened.getScalar(i).element())));
        return flattened.reshape(toExp.shape());
    }


    /**
     * Center an array
     * @param arr the arr to center
     * @param shape the shape of the array
     * @return the center portion of the array based on the
     * specified shape
     */
    public static INDArray center(INDArray arr,int[] shape) {
        DoubleMatrix shapeMatrix = MatrixUtil.toMatrix(shape);
        DoubleMatrix currShape = MatrixUtil.toMatrix(arr.shape());
        NDArray centered = (NDArray) arr;
        DoubleMatrix startIndex = currShape.sub(shapeMatrix).div(2);
        DoubleMatrix endIndex = startIndex.add(shapeMatrix);
        arr = NDArray.wrap(centered.get(RangeUtils.interval((int) startIndex.get(0), (int) endIndex.get(0)),RangeUtils.interval((int) startIndex.get(1),(int) endIndex.get(1))));



        return arr;
    }


    /**
     * Truncates an ndarray to the specified shape.
     * If the shape is the same or greater, it just returns
     * the original array
     * @param nd the ndarray to truncate
     * @param n the number of elements to truncate to
     * @return the truncated ndarray
     */
    public static INDArray truncate(NDArray nd,final int n,int dimension) {

        if(nd.isVector()) {
            NDArray truncated = new NDArray(new int[]{n});
            for(int i = 0;i  < n; i++)
                truncated.put(i,nd.getScalar(i));
            return truncated;
        }

        if(nd.size(dimension) > n) {
            int[] targetShape = ArrayUtil.copy(nd.shape());
            targetShape[dimension] = n;
            int numRequired = ArrayUtil.prod(targetShape);
            if(nd.isVector()) {
                NDArray ret = new NDArray(targetShape);
                int count = 0;
                for(int i = 0; i < nd.length; i+= nd.stride()[dimension]) {
                    ret.put(count++,nd.getScalar(i));

                }
                return ret;
            }

            else if(nd.isMatrix()) {
                List<Double> list = new ArrayList<>();
                //row
                if(dimension == 0) {
                    for(int i = 0;i < nd.rows(); i++) {
                        INDArray row = nd.getRow(i);
                        for(int j = 0; j < row.length(); j++) {
                            if(list.size() == numRequired)
                                return new NDArray(ArrayUtil.toArrayDouble(list),targetShape);

                            list.add((Double) row.getScalar(j).element());
                        }
                    }
                }
                else if(dimension == 1) {
                    for(int i = 0;i < nd.columns(); i++) {
                        INDArray row = nd.getColumn(i);
                        for(int j = 0; j < row.length(); j++) {
                            if(list.size() == numRequired)
                                return new NDArray(ArrayUtil.toArrayDouble(list),targetShape);

                            list.add((Double) row.getScalar(j).element());
                        }
                    }
                }

                else
                    throw new IllegalArgumentException("Illegal dimension for matrix " + dimension);


                return new NDArray(ArrayUtil.toArrayDouble(list),targetShape);

            }


            if(dimension == 0) {
                List<INDArray> slices = new ArrayList<>();
                for(int i = 0; i < n; i++) {
                    INDArray slice = nd.slice(i);
                    slices.add(slice);
                }

                return new NDArray(slices,targetShape);

            }
            else {
                List<Double> list = new ArrayList<>();
                int numElementsPerSlice = ArrayUtil.prod(ArrayUtil.removeIndex(targetShape,0));
                for(int i = 0; i < nd.slices(); i++) {
                    INDArray slice = nd.slice(i).ravel();
                    for(int j = 0; j < numElementsPerSlice; j++)
                        list.add((Double) slice.getScalar(j).element());
                }

                assert list.size() == ArrayUtil.prod(targetShape) : "Illegal shape for length " + list.size();

                return new NDArray(ArrayUtil.toArrayDouble(list),targetShape);

            }




        }

        return nd;

    }
    /**
     * Pads an ndarray with zeros
     * @param nd the ndarray to pad
     * @param targetShape the the new shape
     * @return the padded ndarray
     */
    public static NDArray padWithZeros(NDArray nd,int[] targetShape) {
        if(Arrays.equals(nd.shape(),targetShape))
            return nd;
        //no padding required
        if(ArrayUtil.prod(nd.shape()) >= ArrayUtil.prod(targetShape))
            return nd;

        NDArray ret = new NDArray(targetShape);
        System.arraycopy(nd.data,0,ret.data,0,nd.data.length);
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
     * @param op the operation to perform
     * @param arr the array  to do operations on
     * @return the slice wise operations
     */
    public static NDArray doSliceWise(MatrixOp op,NDArray arr) {
        int columns = isColumnOp(op) ? arr.columns() : arr.rows();
        int[] shape = {arr.slices(),columns};

        NDArray ret = new NDArray(shape);

        for(int i = 0; i < arr.slices(); i++) {
            switch(op) {
                case COLUMN_MIN:
                    ret.putSlice(i,arr.slice(i).columnMins());
                    break;
                case COLUMN_MAX:
                    ret.putSlice(i,arr.slice(i).columnMaxs());
                    break;
                case COLUMN_SUM:
                    ret.putSlice(i,arr.slice(i).columnSums());
                    break;
                case COLUMN_MEAN:
                    ret.putSlice(i,arr.slice(i).columnMeans());
                    break;
                case ROW_MIN:
                    ret.putSlice(i,arr.slice(i).rowMins());
                    break;
                case ROW_MAX:
                    ret.putSlice(i,arr.slice(i).rowMaxs());
                    break;
                case ROW_SUM:
                    ret.putSlice(i,arr.slice(i).rowSums());
                    break;
                case ROW_MEAN:
                    ret.putSlice(i,arr.slice(i).rowMeans());
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
    public static INDArray doSliceWise(ScalarOp op,INDArray arr) {

        arr = arr.reshape(new int[]{1,arr.length()});

        NDArray cast = (NDArray) arr;

        if(op == ScalarOp.NORM_1) {
            return NDArray.scalar(NDArrayBlas.asum(cast));
        }

        else if(op == ScalarOp.NORM_2) {
            return NDArray.scalar(NDArrayBlas.nrm2(cast));

        }

        else if(op == ScalarOp.NORM_MAX) {
            int i = NDArrayBlas.iamax(cast);
            return arr.getScalar(i);
        }

        INDArray s = NDArray.scalar(0);
        for (int i = 0; i < arr.length(); i++) {
            switch (op) {
                case SUM:
                    s.addi(arr.getScalar(i));
                    break;
                case MEAN:
                    s.addi(arr.getScalar(i));
                    break;
                case MAX:
                    double curr = (double) arr.getScalar(i).element();
                    double sEle1 = (double) arr.getScalar(i).element();
                    if (curr > sEle1)
                        s = arr.getScalar(i);
                    break;
                case MIN:
                    double curr2 = (double) arr.getScalar(i).element();
                    double sEle2 = (double) arr.getScalar(i).element();
                    if (curr2 < sEle2)
                        s = arr.getScalar(i);

                    break;
                case PROD:
                    s.muli(arr.getScalar(i));
                    break;


            }


        }

        if(op == ScalarOp.MEAN)
            s.divi(NDArray.scalar(arr.length()));


        return s;
    }



}
