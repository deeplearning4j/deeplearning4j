package org.nd4j.linalg.cpu.nativecpu.ops;

import org.apache.commons.lang3.RandomUtils;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.*;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * Tests for NativeRandom with respect to backend
 *
 * @author raver119@gmail.com
 */
public class RandomTests {
    private static CpuNativeRandom random = new CpuNativeRandom(119, 10000000);

    @Test
    public void testDistribution1() throws Exception {
        CpuNativeRandom random1 = new CpuNativeRandom(119, 100000);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 100000);

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
        CpuNativeRandom random1 = new CpuNativeRandom(119, 20);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 20);

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
        Nd4j.getExecutioner().exec(linspace, random);

        INDArray z2 = linspace.z();

        assertEquals(z1, z2);
    }


    @Test
    public void testDropoutInverted1() throws Exception {
        CpuNativeRandom random1 = new CpuNativeRandom(119, 100000);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 100000);

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
        CpuNativeRandom random1 = new CpuNativeRandom(119, 100000);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 100000);

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
        CpuNativeRandom random1 = new CpuNativeRandom(119, 10000000);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 10000000);

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
    public void testSetSeed1() throws Exception {
        CpuNativeRandom random1 = new CpuNativeRandom(119, 10000000);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 10000000);

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
        CpuNativeRandom random1 = new CpuNativeRandom(119, 10000000);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 10000000);

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
        CpuNativeRandom random1 = new CpuNativeRandom(119, 10000000);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 10000000);

        int array1[] = new int[1000];
        int array2[] = new int[1000];

        for (int e = 0; e < array1.length; e++) {
            array1[e] = random1.nextInt();
            array2[e] = random2.nextInt();

            assertTrue(array1[e] >= 0);
        }

        assertArrayEquals(array1, array2);
    }

    @Test
    public void testJavaSide3() throws Exception {
        CpuNativeRandom random1 = new CpuNativeRandom(119, 10000000);
        CpuNativeRandom random2 = new CpuNativeRandom(119, 10000000);

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
}
