package org.nd4j.linalg.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *  Basic INDArray ops
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


    public static INDArray exp(INDArray toExp) {
        return expi(toExp.dup());
    }

    /**
     * Returns an exponential version of this ndarray
     * @param toExp the INDArray to convert
     * @return the converted ndarray
     */
    public static INDArray expi(INDArray toExp) {
        INDArray flattened = toExp.ravel();
        for(int i = 0; i < flattened.length(); i++)
            flattened.put(i, Nd4j.scalar(Math.exp((double) flattened.getScalar(i).element())));
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
        INDArray shapeMatrix = ArrayUtil.toNDArray(shape);
        INDArray currShape =  ArrayUtil.toNDArray(arr.shape());
        INDArray centered =  arr;
        INDArray startIndex = currShape.sub(shapeMatrix).div(2);
        INDArray endIndex = startIndex.add(shapeMatrix);
        arr = centered.get(NDArrayIndex.interval((int) startIndex.get(0), (int) startIndex.get(0)), NDArrayIndex.interval((int) startIndex.get(1), (int) endIndex.get(1)));



        return arr;
    }


    /**
     * Truncates an INDArray to the specified shape.
     * If the shape is the same or greater, it just returns
     * the original array
     * @param nd the INDArray to truncate
     * @param n the number of elements to truncate to
     * @return the truncated ndarray
     */
    public static INDArray truncate(INDArray nd,final int n,int dimension) {

        if(nd.isVector()) {
            INDArray truncated = Nd4j.create(new int[]{n});
            for(int i = 0;i  < n; i++)
                truncated.put(i,nd.getScalar(i));
            return truncated;
        }

        if(nd.size(dimension) > n) {
            int[] targetShape = ArrayUtil.copy(nd.shape());
            targetShape[dimension] = n;
            int numRequired = ArrayUtil.prod(targetShape);
            if(nd.isVector()) {
                INDArray ret = Nd4j.create(targetShape);
                int count = 0;
                for(int i = 0; i < nd.length(); i+= nd.stride()[dimension]) {
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
                                return Nd4j.create(ArrayUtil.toArrayDouble(list), targetShape);

                            list.add((Double) row.getScalar(j).element());
                        }
                    }
                }
                else if(dimension == 1) {
                    for(int i = 0;i < nd.columns(); i++) {
                        INDArray row = nd.getColumn(i);
                        for(int j = 0; j < row.length(); j++) {
                            if(list.size() == numRequired)
                                return Nd4j.create(ArrayUtil.toArrayDouble(list), targetShape);

                            list.add((Double) row.getScalar(j).element());
                        }
                    }
                }

                else
                    throw new IllegalArgumentException("Illegal dimension for matrix " + dimension);


                return Nd4j.create(ArrayUtil.toArrayDouble(list), targetShape);

            }


            if(dimension == 0) {
                List<INDArray> slices = new ArrayList<>();
                for(int i = 0; i < n; i++) {
                    INDArray slice = nd.slice(i);
                    slices.add(slice);
                }

                return Nd4j.create(slices, targetShape);

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

                return Nd4j.create(ArrayUtil.toArrayDouble(list), targetShape);

            }




        }

        return nd;

    }
    /**
     * Pads an INDArray with zeros
     * @param nd the INDArray to pad
     * @param targetShape the the new shape
     * @return the padded ndarray
     */
    public static INDArray padWithZeros(INDArray nd,int[] targetShape) {
        if(Arrays.equals(nd.shape(),targetShape))
            return nd;
        //no padding required
        if(ArrayUtil.prod(nd.shape()) >= ArrayUtil.prod(targetShape))
            return nd;

        INDArray ret = Nd4j.create(targetShape);
        System.arraycopy(nd.data(),0,ret.data(),0,nd.data().length);
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
     * Execute an element wise operation over the whole array
     * @param op the operation to execute
     * @param arr the array to perform operations on
     * @return the result over the whole nd array
     */
    public static INDArray doSliceWise(ScalarOp op,INDArray arr) {

        arr = arr.reshape(new int[]{1,arr.length()});

        INDArray cast =   arr;

        if(op == ScalarOp.NORM_1) {
            return Nd4j.scalar(Nd4j.getBlasWrapper().asum(cast));
        }

        else if(op == ScalarOp.NORM_2) {
            return Nd4j.scalar(Nd4j.getBlasWrapper().nrm2(cast));

        }

        else if(op == ScalarOp.NORM_MAX) {
            int i = Nd4j.getBlasWrapper().iamax(cast);
            return arr.getScalar(i);
        }

        INDArray s = Nd4j.scalar(0);
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
            s.divi(Nd4j.scalar(arr.length()));


        return s;
    }



}
