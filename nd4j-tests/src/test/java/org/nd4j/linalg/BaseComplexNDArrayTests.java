package org.nd4j.linalg;


import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;

/**
 * Base class for all complex ndarray tests
 * Any tests here will share ordering (such as vectors)
 *
 * @author Adam Gibson
 */
public abstract class BaseComplexNDArrayTests extends BaseNd4jTest {
    public BaseComplexNDArrayTests() {
    }

    public BaseComplexNDArrayTests(Nd4jBackend backend) {
        super(backend);
    }

    public BaseComplexNDArrayTests(String name) {
        super(name);
    }

    public BaseComplexNDArrayTests(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    @Before
    public void before() {
        super.before();
    }

    @After
    public void after() {
        super.after();
    }


    @Test
    public void testOtherLinearView() {
        IComplexNDArray arr = Nd4j.complexLinSpace(1,8,8).reshape(2,4);
        IComplexNDArray slices = arr.slice(0);
        IComplexNDArray slice1 = arr.slice(1);
        IComplexNDArray arrLinear = arr.linearView();
        System.out.println(arrLinear);
    }



    @Test
    public void testWrap() {
        IComplexNDArray c = Nd4j.createComplex(Nd4j.linspace(1, 4, 4).reshape(2, 2));
        assertEquals(true, Arrays.equals(new int[]{2, 2}, c.shape()));

        IComplexNDArray vec = Nd4j.createComplex(Nd4j.linspace(1, 4, 4));
        assertEquals(true, vec.isVector());
        assertEquals(true, Shape.shapeEquals(new int[]{4}, vec.shape()));

    }


    protected void verifyElements(IComplexNDArray d, IComplexNDArray d2) {
        for (int i = 0; i < d.rows(); i++) {
            for (int j = 0; j < d.columns(); j++) {
                IComplexNumber test1 = d.getComplex(i, j);
                IComplexNumber test2 = d2.getComplex(i, j);
                assertEquals(test1.realComponent().doubleValue(), test2.realComponent().doubleValue(), 1e-6);
                assertEquals(test1.imaginaryComponent().doubleValue(), test2.imaginaryComponent().doubleValue(), 1e-6);

            }
        }
    }
}
