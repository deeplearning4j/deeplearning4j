package org.nd4j.linalg.factory;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import static org.junit.Assert.assertEquals;

/**
 * Created by willow on 6/16/15.
 */
public class Nd4jTest {

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


}
