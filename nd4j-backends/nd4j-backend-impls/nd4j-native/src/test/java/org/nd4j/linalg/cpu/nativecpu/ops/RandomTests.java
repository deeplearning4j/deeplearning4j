package org.nd4j.linalg.cpu.nativecpu.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.*;
import org.nd4j.linalg.api.ops.random.impl.BinomialDistribution;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.factory.DefaultDistributionFactory;
import org.nd4j.linalg.api.rng.distribution.impl.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Tests for NativeRandom with respect to backend
 *
 * @author raver119@gmail.com
 */
public class RandomTests {

    @Test
    public void testDistribution1() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z1 = Nd4j.create(1000);
        INDArray z2 = Nd4j.create(1000);
        UniformDistribution distribution = new UniformDistribution(z1, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution, random1);
        UniformDistribution distribution2 = new UniformDistribution(z2, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution2, random2);

        System.out.println("Data: " + z1);
        System.out.println("Data: " + z2);
        for (int e = 0; e < z1.length(); e++) {
            double val = z1.getDouble(e);
            assertTrue(val >= 1.0 && val <= 2.0);
        }

        assertEquals(z1, z2);
    }


    @Test
    public void testDistribution2() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z1 = Nd4j.create(32);
        INDArray z2 = Nd4j.create(32);
        UniformDistribution distribution = new UniformDistribution(z1, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution, random1);
        UniformDistribution distribution2 = new UniformDistribution(z2, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution2, random2);

        System.out.println("Data: " + z1);
        System.out.println("Data: " + z2);
        for (int e = 0; e < z1.length(); e++) {
            double val = z1.getDouble(e);
            assertTrue(val >= 1.0 && val <= 2.0);
        }

        assertEquals(z1, z2);
    }


    @Test
    public void testLinspace1() throws Exception {
        INDArray z1 = Nd4j.linspace(1, 100, 200);

        Linspace linspace = new Linspace(1, 100, 200);
        Nd4j.getExecutioner().exec(linspace, Nd4j.getRandom());

        INDArray z2 = linspace.z();

        assertEquals(z1, z2);
    }


    @Test
    public void testDropoutInverted1() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z1 = Nd4j.ones(300);
        INDArray z2 = Nd4j.ones(300);
        INDArray zDup = z1.dup();

        DropOutInverted op1 = new DropOutInverted(z1, z1, 0.10);
        Nd4j.getExecutioner().exec(op1, random1);

        DropOutInverted op2 = new DropOutInverted(z2, z2, 0.10);
        Nd4j.getExecutioner().exec(op2, random2);

        assertNotEquals(zDup, z1);

        assertEquals(z1, z2);
    }

    @Test
    public void testDropout1() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z1 = Nd4j.ones(300);
        INDArray z2 = Nd4j.ones(300);
        INDArray zDup = z1.dup();

        DropOut op1 = new DropOut(z1, z1, 0.10);
        Nd4j.getExecutioner().exec(op1, random1);

        DropOut op2 = new DropOut(z2, z2, 0.10);
        Nd4j.getExecutioner().exec(op2, random2);

        assertNotEquals(zDup, z1);

        assertEquals(z1, z2);
    }

    @Test
    public void testGaussianDistribution1() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z1 = Nd4j.create(100000);
        INDArray z2 = Nd4j.create(100000);

        GaussianDistribution op1 = new GaussianDistribution(z1, 0.0, 1.0);
        Nd4j.getExecutioner().exec(op1, random1);

        GaussianDistribution op2 = new GaussianDistribution(z2, 0.0, 1.0);
        Nd4j.getExecutioner().exec(op2, random2);

        assertEquals(0.0, z1.meanNumber().doubleValue(), 0.01);

        assertEquals(z1, z2);
    }


    @Test
    public void testGaussianDistribution2() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random3 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random4 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z1 = Nd4j.create(100000);
        INDArray z2 = Nd4j.create(100000);
        INDArray z3 = Nd4j.create(100000);
        INDArray z4 = Nd4j.create(100000);

        random3.reSeed(8231);
        random4.reSeed(4453523);

        GaussianDistribution op1 = new GaussianDistribution(z1, 0.0, 1.0);
        Nd4j.getExecutioner().exec(op1, random1);

        GaussianDistribution op2 = new GaussianDistribution(z2, 0.0, 1.0);
        Nd4j.getExecutioner().exec(op2, random2);

        GaussianDistribution op3 = new GaussianDistribution(z3, 0.0, 1.0);
        Nd4j.getExecutioner().exec(op3, random3);

        GaussianDistribution op4 = new GaussianDistribution(z4, 0.0, 1.0);
        Nd4j.getExecutioner().exec(op4, random4);

        assertEquals(0.0, z1.meanNumber().doubleValue(), 0.01);

        assertEquals(z1, z2);

        assertNotEquals(z1, z3);
        assertNotEquals(z2, z4);
        assertNotEquals(z3, z4);
    }

    @Test
    public void testSetSeed1() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z01 = Nd4j.create(1000);
        INDArray z11 = Nd4j.create(1000);

        UniformDistribution distribution01 = new UniformDistribution(z01, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution01, random1);

        UniformDistribution distribution11 = new UniformDistribution(z11, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution11, random2);

        random1.setSeed(1999);
        random2.setSeed(1999);

        INDArray z02 = Nd4j.create(100);
        UniformDistribution distribution02 = new UniformDistribution(z02, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution02, random1);

        INDArray z12 = Nd4j.create(100);
        UniformDistribution distribution12 = new UniformDistribution(z12, 1.0, 2.0);
        Nd4j.getExecutioner().exec(distribution12, random2);


        assertEquals(z01, z11);
        assertEquals(z02, z12);
    }


    @Test
    public void testJavaSide1() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        float array1[] = new float[1000];
        float array2[] = new float[1000];

        for (int e = 0; e < array1.length; e++) {
            array1[e] = random1.nextFloat();
            array2[e] = random2.nextFloat();

            assertTrue(array1[e] <= 1.0f);
        }

        assertArrayEquals(array1, array2, 1e-5f);
    }


    @Test
    public void testJavaSide2() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        int array1[] = new int[1000];
        int array2[] = new int[1000];

        for (int e = 0; e < array1.length; e++) {
            array1[e] = random1.nextInt();
            array2[e] = random2.nextInt();

            assertEquals(array1[e], array2[e]);
            assertTrue(array1[e] >= 0);
        }

        assertArrayEquals(array1, array2);
    }

    @Test
    public void testJavaSide3() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        int array1[] = new int[10000];
        int array2[] = new int[10000];

        for (int e = 0; e < array1.length; e++) {
            array1[e] = random1.nextInt(9823);
            array2[e] = random2.nextInt(9823);

            assertTrue(array1[e] >= 0);
            assertTrue(array1[e] < 9823);
        }

        assertArrayEquals(array1, array2);
    }

    /**
     * This test checks reSeed mechanics for native side
     *
     * @throws Exception
     */
    @Test
    public void testJavaSide4() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        int array1[] = new int[1000];
        int array2[] = new int[1000];

        for (int e = 0; e < array1.length; e++) {
            array1[e] = random1.nextInt();
            array2[e] = random2.nextInt();

            assertEquals(array1[e], array2[e]);
            assertTrue(array1[e] >= 0);
        }

        assertArrayEquals(array1, array2);

        random1.reSeed();
        random1.reSeed();

        int array3[] = new int[1000];
        int array4[] = new int[1000];

        for (int e = 0; e < array1.length; e++) {
            array3[e] = random1.nextInt();
            array4[e] = random2.nextInt();

            assertNotEquals(array3[e], array4[e]);
            assertTrue(array1[e] >= 0);
        }
    }


    @Test
    public void testBernoulliDistribution1() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z1 = Nd4j.zeros(1000);
        INDArray z2 = Nd4j.zeros(1000);
        INDArray z1Dup = Nd4j.zeros(1000);

        BernoulliDistribution op1 = new BernoulliDistribution(z1, 0.25);
        BernoulliDistribution op2 = new BernoulliDistribution(z2, 0.25);

        Nd4j.getExecutioner().exec(op1, random1);
        Nd4j.getExecutioner().exec(op2, random2);

        assertNotEquals(z1Dup, z1);

        assertEquals(z1, z2);
    }

    @Test
    public void testBinomialDistribution1() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z1 = Nd4j.zeros(1000);
        INDArray z2 = Nd4j.zeros(1000);
        INDArray z1Dup = Nd4j.zeros(1000);

        BinomialDistribution op1 = new BinomialDistribution(z1, 5, 0.25);
        BinomialDistribution op2 = new BinomialDistribution(z2, 5, 0.25);

        Nd4j.getExecutioner().exec(op1, random1);
        Nd4j.getExecutioner().exec(op2, random2);

        assertNotEquals(z1Dup, z1);

        assertEquals(z1, z2);

        BooleanIndexing.and(z1, Conditions.lessThanOrEqual(5.0));
        BooleanIndexing.and(z1, Conditions.greaterThanOrEqual(0.0));
    }

    @Test
    public void testBinomialDistribution2() throws Exception {
        Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
        Random random2 = Nd4j.getRandomFactory().getNewRandomInstance(119);

        INDArray z1 = Nd4j.zeros(1000);
        INDArray z2 = Nd4j.zeros(1000);
        INDArray z1Dup = Nd4j.zeros(1000);

        INDArray probs = Nd4j.create(new float[]{0.25f, 0.43f, 0.55f, 0.43f, 0.25f});

        BinomialDistribution op1 = new BinomialDistribution(z1, 5, probs);
        BinomialDistribution op2 = new BinomialDistribution(z2, 5, probs);

        Nd4j.getExecutioner().exec(op1, random1);
        Nd4j.getExecutioner().exec(op2, random2);

        assertNotEquals(z1Dup, z1);

        assertEquals(z1, z2);

        BooleanIndexing.and(z1, Conditions.lessThanOrEqual(5.0));
        BooleanIndexing.and(z1, Conditions.greaterThanOrEqual(0.0));
    }

    @Ignore
    @Test
    public void testDeallocation1() throws Exception {

        while (true) {
            Random random1 = Nd4j.getRandomFactory().getNewRandomInstance(119);
            random1.nextInt();

            System.gc();
            Thread.sleep(50);
        }
    }
}
