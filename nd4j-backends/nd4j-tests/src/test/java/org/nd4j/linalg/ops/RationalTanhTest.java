package org.nd4j.linalg.ops;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.RationalTanhDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertTrue;

/**
 * Rational tanh approximation from
 * https://arxiv.org/pdf/1508.01292v3
 * https://github.com/deeplearning4j/libnd4j/issues/351
 */
@RunWith(Parameterized.class)
public class RationalTanhTest extends BaseNd4jTest {

    public RationalTanhTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void gradientCheck() {

        double eps = 1e-6;
        INDArray A = Nd4j.linspace(-3, 3, 10).reshape(2, 5);
        INDArray ADer = Nd4j.getExecutioner().execAndReturn(new RationalTanhDerivative(A.dup()));

        double[] a = A.data().asDouble();
        double[] aDer = ADer.data().asDouble();

        for (int i = 0; i < 10; i++) {
            double empirical = (f(a[i] + eps) - f(a[i] - eps)) / (2 * eps);
            double analytic = aDer[i];
            assertTrue(Math.abs(empirical - analytic) / (Math.abs(empirical) + Math.abs(analytic)) < 0.001);
        }

    }

    public static double f(double x) {
        return 1.7159 * tanhApprox(2.0 / 3 * x);
    }

    /*
    public static INDArray fDeriv(double x){
        //return C1 * 2.0/3 * tanhDeriv(2.0 / 3 * x);
    }
    */

    public static double tanhApprox(double y) {
        return Math.signum(y) * (1.0 - 1.0 / (1 + Math.abs(y) + y * y + 1.41645 * Math.pow(y, 4.0)));
    }

    /*
    public static double tanhDeriv(double y){
        double a = 1 + Math.abs(y) + y*y + C * Math.pow(y,4);
        return (1 + Math.signum(y) * (2*y + 4*C*Math.pow(y,3))) / (a * a);
    }
    */

    @Override
    public char ordering() {
        return 'f';
    }
}
