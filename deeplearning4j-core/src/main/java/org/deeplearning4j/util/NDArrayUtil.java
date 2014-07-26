package org.deeplearning4j.util;

import org.deeplearning4j.nn.linalg.DimensionSlice;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.nn.linalg.SliceOp;
import org.jblas.NativeBlas;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

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


    /**
     * Truncates an ndarray to the specified shape.
     * If the shape is the same or greater, it just returns
     * the original array
     * @param nd the ndarray to truncate
     * @param targetShape the new shape
     * @return the truncated ndarray
     */
    public static NDArray truncate(NDArray nd, final int[] targetShape,int dimension) {
        if(Arrays.equals(nd.shape(),targetShape))
            return nd;

        //same length: just need to reshape, the reason for this is different dimensions maybe of different sizes
        if(ArrayUtil.prod(nd.shape()) == ArrayUtil.prod(targetShape))
            return nd.reshape(targetShape);

        final NDArray ret = new NDArray(targetShape);
        if(ret.isVector())  {
            final AtomicInteger currentSlice = new AtomicInteger(0);
            nd.iterateOverDimension(dimension,new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                      NDArray result = (NDArray) nd.getResult();
                      for(int i = 0; i < 1; i++) {
                          ret.put(currentSlice.getAndIncrement(),result.get(i));
                      }
                }
            });




            return ret;
        }

        int[] sliceShape = ArrayUtil.removeIndex(targetShape,0);
        for(int i = 0; i < ret.slices(); i++) {
            ret.putSlice(i,truncate(nd.slice(i),sliceShape,dimension));
        }

        return ret;


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
     * Dimension wise operation along an ndarray
     * @param op the op to do
     * @param arr the array  to do operations on
     * @param dimension the dimension to do operations on
     * @return the
     */
    public static NDArray dimensionOp(DimensionOp op,NDArray arr,int dimension) {
        if(dimension >= arr.slices())
            return arr;

        int[] shape = ArrayUtil.removeIndex(arr.shape(),dimension);


        List<NDArray> list = new ArrayList<>();

        for(int i = 0; i < arr.slices(); i++) {
            switch(op) {
                case SUM:
                    //list.add(arr.slice(i,dimension).sum());
                    break;
                case MAX:
                    list.add(arr.slice(i,dimension).columnMaxs());
                    break;
                case MEAN:
                    list.add(arr.slice(i,dimension).columnSums());
                    break;
                case PROD:
                    list.add(arr.slice(i,dimension).columnMeans());
                    break;
            }
        }


        return arr;
    }


    public static double[] collectForOp(NDArray from,int dimension) {
        //number of operations per op
        int num = from.shape()[dimension];

        //how to isolate blocks from the matrix
        double[] d = new double[num];
        int idx = 0;
        for(int k = 0; k < d.length; k++) {
            d[k] = from.data[idx];
            idx += num;
        }


        return d;
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






    public static double doSliceWise(ScalarOp op,NDArray arr) {
        if(arr.isScalar())
            return arr.get(0);

        else {
            double ret = 0;

            for(int i = 0; i < arr.slices(); i++) {
                switch(op) {
                    case MEAN:
                        ret += arr.slice(i).mean();
                        break;
                    case SUM :
                        ret += arr.slice(i).sum();
                        break;
                    case PROD :
                        ret += arr.slice(i).prod();
                        break;
                    case MAX :
                        double max = arr.slice(i).max();
                        if(max > ret)
                            ret = max;
                        break;
                    case MIN :
                        double min = arr.slice(i).min();
                        if(min < ret)
                            ret = min;
                        break;
                    case ARG_MIN:
                        double argMin = arr.slice(i).argmin();
                        if(argMin < ret)
                            ret = argMin;
                        break;
                    case ARG_MAX:
                        double argMax = arr.slice(i).argmax();
                        if(argMax > ret)
                            ret = argMax;
                        break;
                    case NORM_1:
                        ret += arr.slice(i).norm1();
                        break;
                    case NORM_2:
                        ret += arr.slice(i).norm2();
                        break;
                    case NORM_MAX:
                        ret += arr.slice(i).normmax();
                        break;


                }
            }

            return ret;
        }

    }


}
