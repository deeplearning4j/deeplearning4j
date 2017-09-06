package org.nd4j.linalg;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.util.ArrayUtil;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by rcorbish
 */
@RunWith(Parameterized.class)
public class TestEigen extends BaseNd4jTest {

    protected DataBuffer.Type initialType;

    public TestEigen(Nd4jBackend backend) {
        super(backend);
        initialType = Nd4j.dataType();
    }

    @Before
    public void before() {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @After
    public void after() {
        Nd4j.setDataType(initialType);
    }

    // test of functions added by Luke Czapla
    // Compares solution of A x = L x  to solution to A x = L B x when it is simple
    @Test
    public void test2Syev() {
        double[][] matrix = new double[][] {{0.0427, -0.04, 0, 0, 0, 0}, {-0.04, 0.0427, 0, 0, 0, 0},
                        {0, 0.00, 0.0597, 0, 0, 0}, {0, 0, 0, 50, 0, 0}, {0, 0, 0, 0, 50, 0}, {0, 0, 0, 0, 0, 50}};
        INDArray m = Nd4j.create(ArrayUtil.flattenDoubleArray(matrix), new int[] {6, 6});
        INDArray res = Eigen.symmetricGeneralizedEigenvalues(m, true);

        INDArray n = Nd4j.create(ArrayUtil.flattenDoubleArray(matrix), new int[] {6, 6});
        INDArray res2 = Eigen.symmetricGeneralizedEigenvalues(n, Nd4j.eye(6).mul(2.0), true);

        for (int i = 0; i < 6; i++) {
            assertEquals(res.getDouble(i), 2 * res2.getDouble(i), 0.000001);
        }

    }

    @Test
    public void testSyev() {
        INDArray A = Nd4j.create(new float[][] {{1.96f, -6.49f, -0.47f, -7.20f, -0.65f},
                        {-6.49f, 3.80f, -6.39f, 1.50f, -6.34f}, {-0.47f, -6.39f, 4.17f, -1.51f, 2.67f},
                        {-7.20f, 1.50f, -1.51f, 5.70f, 1.80f}, {-0.65f, -6.34f, 2.67f, 1.80f, -7.10f}});

        INDArray B = A.dup();
        INDArray e = Eigen.symmetricGeneralizedEigenvalues(A);

        for (int i = 0; i < A.rows(); i++) {
            INDArray LHS = B.mmul(A.slice(i, 1));
            INDArray RHS = A.slice(i, 1).mul(e.getFloat(i));

            for (int j = 0; j < LHS.length(); j++) {
                assertEquals(LHS.getFloat(j), RHS.getFloat(j), 0.001f);
            }
        }
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
