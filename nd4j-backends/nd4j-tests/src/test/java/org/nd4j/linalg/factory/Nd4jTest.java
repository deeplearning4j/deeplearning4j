package org.nd4j.linalg.factory;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 */
@RunWith(Parameterized.class)
public class Nd4jTest extends BaseNd4jTest {
    public Nd4jTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testRandShapeAndRNG() {
        INDArray ret = Nd4j.rand(new int[] {4, 2}, Nd4j.getRandomFactory().getNewRandomInstance(123));
        INDArray ret2 = Nd4j.rand(new int[] {4, 2}, Nd4j.getRandomFactory().getNewRandomInstance(123));

        assertEquals(ret, ret2);
    }

    @Test
    public void testRandShapeAndMinMax() {
        INDArray ret = Nd4j.rand(new int[] {4, 2}, -0.125f, 0.125f, Nd4j.getRandomFactory().getNewRandomInstance(123));
        INDArray ret2 = Nd4j.rand(new int[] {4, 2}, -0.125f, 0.125f, Nd4j.getRandomFactory().getNewRandomInstance(123));
        assertEquals(ret, ret2);
    }

    @Test
    public void testCreateShape() {
        INDArray ret = Nd4j.create(new int[] {4, 2});

        assertEquals(ret.length(), 8);
    }

    @Test
    public void testCreateFromList() {
        List<Double> doubles = Arrays.asList(1.0, 2.0);
        INDArray NdarrayDobules = Nd4j.create(doubles);

        assertEquals((Double)NdarrayDobules.getDouble(0),doubles.get(0));
        assertEquals((Double)NdarrayDobules.getDouble(1),doubles.get(1));

        List<Float> floats = Arrays.asList(3.0f, 4.0f);
        INDArray NdarrayFloats = Nd4j.create(floats);
        assertEquals((Float)NdarrayFloats.getFloat(0),floats.get(0));
        assertEquals((Float)NdarrayFloats.getFloat(1),floats.get(1));
    }

    @Test
    public void testGetRandom() {
        Random r = Nd4j.getRandom();
        Random t = Nd4j.getRandom();

        assertEquals(r, t);
    }

    @Test
    public void testGetRandomSetSeed() {
        Random r = Nd4j.getRandom();
        Random t = Nd4j.getRandom();

        assertEquals(r, t);
        r.setSeed(123);
        assertEquals(r, t);
    }

    @Test
    public void testOrdering() {
        INDArray fNDArray = Nd4j.create(new float[] {1f}, NDArrayFactory.FORTRAN);
        assertEquals(NDArrayFactory.FORTRAN, fNDArray.ordering());
        INDArray cNDArray = Nd4j.create(new float[] {1f}, NDArrayFactory.C);
        assertEquals(NDArrayFactory.C, cNDArray.ordering());
    }

    @Override
    public char ordering() {
        return 'c';
    }


    @Test
    public void testMean() {
        INDArray data = Nd4j.create(new double[] {4., 4., 4., 4., 8., 8., 8., 8., 4., 4., 4., 4., 8., 8., 8., 8., 4.,
                        4., 4., 4., 8., 8., 8., 8., 4., 4., 4., 4., 8., 8., 8., 8, 2., 2., 2., 2., 4., 4., 4., 4., 2.,
                        2., 2., 2., 4., 4., 4., 4., 2., 2., 2., 2., 4., 4., 4., 4., 2., 2., 2., 2., 4., 4., 4., 4.},
                        new int[] {2, 2, 4, 4});

        INDArray actualResult = data.mean(0);
        INDArray expectedResult = Nd4j.create(new double[] {3., 3., 3., 3., 6., 6., 6., 6., 3., 3., 3., 3., 6., 6., 6.,
                        6., 3., 3., 3., 3., 6., 6., 6., 6., 3., 3., 3., 3., 6., 6., 6., 6.}, new int[] {2, 4, 4});
        assertEquals(getFailureMessage(), expectedResult, actualResult);
    }


    @Test
    public void testVar() {
        INDArray data = Nd4j.create(new double[] {4., 4., 4., 4., 8., 8., 8., 8., 4., 4., 4., 4., 8., 8., 8., 8., 4.,
                        4., 4., 4., 8., 8., 8., 8., 4., 4., 4., 4., 8., 8., 8., 8, 2., 2., 2., 2., 4., 4., 4., 4., 2.,
                        2., 2., 2., 4., 4., 4., 4., 2., 2., 2., 2., 4., 4., 4., 4., 2., 2., 2., 2., 4., 4., 4., 4.},
                        new int[] {2, 2, 4, 4});

        INDArray actualResult = data.var(false, 0);
        INDArray expectedResult = Nd4j.create(new double[] {1., 1., 1., 1., 4., 4., 4., 4., 1., 1., 1., 1., 4., 4., 4.,
                        4., 1., 1., 1., 1., 4., 4., 4., 4., 1., 1., 1., 1., 4., 4., 4., 4.}, new int[] {2, 4, 4});
        assertEquals(getFailureMessage(), expectedResult, actualResult);
    }

    @Test
    public void testVar2() {
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray var = arr.var(false, 0);
        assertEquals(Nd4j.create(new double[] {2.25, 2.25, 2.25}), var);
    }


}
