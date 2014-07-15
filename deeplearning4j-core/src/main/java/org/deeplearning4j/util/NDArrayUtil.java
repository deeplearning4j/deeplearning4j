package org.deeplearning4j.util;

import org.deeplearning4j.nn.NDArray;

/**
 *  Basic NDArray ops
 *
 *  @author Adam Gibson
 */
public class NDArrayUtil {
    public static enum ScalarOp {
        SUM,MEAN,PROD,MAX,MIN,ARG_MAX,ARG_MIN
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


        return arr;
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

                }
            }

            return ret;
        }

    }


}
