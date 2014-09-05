package org.nd4j.linalg.util;

import org.nd4j.linalg.api.ndarray.INDArray;

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

    /**
     * Asserts matrix multiply rules (columns of left == rows of right)
     * @param nd1 the left ndarray
     * @param nd2 the right ndarray
     */
    public static void assertMultiplies(INDArray nd1,INDArray nd2) {
        assert nd1.columns() == nd2.rows() : "Column of left " + nd1.columns() + " != rows of right " + nd2.rows();
    }



    public static void assertColumns(INDArray n,INDArray n2) {
        assert n.columns() == n2.columns() : "Mis matched rows";
    }

    public static void assertValidNum(INDArray n) {
        n = n.ravel();
        for(int i = 0; i < n.length(); i++) {
            float d = (float) n.getScalar(i).element();
            assert !(Double.isNaN(d) || Double.isInfinite(d)) : "Found infinite or nan";

        }
    }

}
