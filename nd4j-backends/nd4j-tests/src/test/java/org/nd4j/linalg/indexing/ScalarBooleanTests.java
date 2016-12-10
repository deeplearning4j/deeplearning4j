package org.nd4j.linalg.indexing;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class ScalarBooleanTests extends BaseNd4jTest {

    public ScalarBooleanTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testEq1() {
        INDArray x = Nd4j.create(new double[]{0, 1, 2, 1});
        INDArray exp = Nd4j.create(new double[]{0, 0, 1, 0});

        INDArray z = x.eq(2);

        assertEquals(exp, z);
    }

    @Test
    public void testEq2() {
        INDArray x = Nd4j.create(new double[]{0, 1, 2, 1});
        INDArray exp = Nd4j.create(new double[]{0, 0, 1, 0});

        x.eqi(2);

        assertEquals(exp, x);
    }


    @Test
    public void testNEq1() {
        INDArray x = Nd4j.create(new double[]{0, 1, 2, 1});
        INDArray exp = Nd4j.create(new double[]{1, 0, 1, 0});

        INDArray z = x.neq(1);

        assertEquals(exp, z);
    }

    @Test
    public void testLT1() {
        INDArray x = Nd4j.create(new double[]{0, 1, 2, 1});
        INDArray exp = Nd4j.create(new double[]{1, 1, 0, 1});

        INDArray z = x.lt(2);

        assertEquals(exp, z);
    }

    @Test
    public void testLTE1() {
        INDArray x = Nd4j.create(new double[]{0, 1, 2, 1});
        INDArray exp = Nd4j.create(new double[]{1, 1, 1, 1});

        x.ltei(2);

        assertEquals(exp, x);
    }

    @Test
    public void testGT1() {
        INDArray x = Nd4j.create(new double[]{0, 1, 2, 4});
        INDArray exp = Nd4j.create(new double[]{0, 0, 1, 1});

        INDArray z = x.gt(1);

        assertEquals(exp, z);
    }

    @Test
    public void testGTE1() {
        INDArray x = Nd4j.create(new double[]{0, 1, 2, 4});
        INDArray exp = Nd4j.create(new double[]{0, 0, 1, 1});

        x.gtei(2);

        assertEquals(exp, x);
    }

    @Test
    public void testScalarMinMax1() {
        INDArray x = Nd4j.create(new double[]{1, 3, 5, 7});
        INDArray exp1 = Nd4j.create(new double[]{1, 3, 5, 7});
        INDArray exp2 = Nd4j.create(new double[]{1e-5, 1e-5, 1e-5, 1e-5});

        INDArray z1 = Transforms.max(x, Nd4j.EPS_THRESHOLD, true);
        INDArray z2 = Transforms.min(x, Nd4j.EPS_THRESHOLD, true);

        assertEquals(exp1, z1);
        assertEquals(exp2, z2);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
