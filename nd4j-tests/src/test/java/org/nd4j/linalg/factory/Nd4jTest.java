package org.nd4j.linalg.factory;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import static org.junit.Assert.assertEquals;

/**
 * Created by willow on 6/16/15.
 */
public class Nd4jTest extends BaseNd4jTest {
    public Nd4jTest() {
        super();
    }

    public Nd4jTest(String name) {
        super(name);
    }

    public Nd4jTest(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public Nd4jTest(Nd4jBackend backend) {
        super(backend);
    }
    @Test
    public void testRandShapeAndRNG() {
        INDArray ret = Nd4j.rand(new int[]{4, 2}, new DefaultRandom(123));
        INDArray ret2 = Nd4j.rand(new int[]{4, 2}, new DefaultRandom(123));

        assertEquals(ret, ret2);
    }

    @Test
    public void testRandShapeAndMinMax() {
        INDArray ret = Nd4j.rand(new int[]{4, 2}, -0.125f, 0.125f, new DefaultRandom(123));
        INDArray ret2 = Nd4j.rand(new int[]{4, 2}, -0.125f, 0.125f, new DefaultRandom(123));
        assertEquals(ret, ret2);
    }

    @Test
    public void testCreateShape() {
        INDArray ret = Nd4j.create(new int[]{4, 2});

        assertEquals(ret.length(), 8);
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

    @Override
    public char ordering() {
        return 'c';
    }
}
