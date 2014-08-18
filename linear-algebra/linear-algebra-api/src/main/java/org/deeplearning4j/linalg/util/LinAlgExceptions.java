package org.deeplearning4j.linalg.util;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Linear algebra exceptions
 *
 * @author Adam Gibson
 */
public class LinAlgExceptions {

    public static void assertSameShape(INDArray n,INDArray n2) {
        assert Shape.shapeEquals(n.shape(), n2.shape()) : "Mis matched shapes";
    }

    public static void assertRows(INDArray n,INDArray n2) {
        assert n.rows() == n2.rows() : "Mis matched rows";
    }


    public static void assertColumns(INDArray n,INDArray n2) {
        assert n.columns() == n2.columns() : "Mis matched rows";
    }

    public static void assertValidNum(INDArray n) {
        n = n.ravel();
        for(int i = 0; i < n.length(); i++) {
            double d = (double) n.getScalar(i).element();
            assert !(Double.isNaN(d) || Double.isInfinite(d)) : "Found infinite or nan";

        }
    }

}
