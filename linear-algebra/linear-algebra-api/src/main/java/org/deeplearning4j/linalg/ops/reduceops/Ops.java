package org.deeplearning4j.linalg.ops.reduceops;

/**
 * Ops for scalar, dimension, and matrix
 *
 * @author Adam Gibson
 */
public class Ops {


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

}
