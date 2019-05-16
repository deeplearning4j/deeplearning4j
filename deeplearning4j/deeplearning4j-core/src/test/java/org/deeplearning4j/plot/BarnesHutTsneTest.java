/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.plot;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.BarnesHutSymmetrize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

// import org.nd4j.jita.conf.CudaEnvironment;

/**
 * Created by agibsonccc on 10/1/14.
 */
public class BarnesHutTsneTest extends BaseDL4JTest {
    @Before
    public void setUp() {
        //   CudaEnvironment.getInstance().getConfiguration().enableDebug(true).setVerbose(false);
    }

    @Test
    public void testTsne() throws Exception {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).setMaxIter(10).theta(0.5).learningRate(500)
                        .useAdaGrad(false).build();

        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getTempFileFromArchive();
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(), "   ").get(NDArrayIndex.interval(0, 100),
                        NDArrayIndex.interval(0, 784));

        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream()).subList(0, 100);
        b.fit(data);
    }

    @Test
    public void testBuilderFields() throws Exception {
        final double theta = 0;
        final boolean invert = false;
        final String similarityFunctions = "euclidean";
        final int maxIter = 1;
        final double realMin = 1.0;
        final double initialMomentum = 2.0;
        final double finalMomentum = 3.0;
        final double momentum = 4.0;
        final int switchMomentumIteration = 1;
        final boolean normalize = false;
        final int stopLyingIteration = 100;
        final double tolerance = 1e-1;
        final double learningRate = 100;
        final boolean useAdaGrad = false;
        final double perplexity = 1.0;
        final double minGain = 1.0;

        BarnesHutTsne b = new BarnesHutTsne.Builder().theta(theta).invertDistanceMetric(invert)
                        .similarityFunction(similarityFunctions).setMaxIter(maxIter).setRealMin(realMin)
                        .setInitialMomentum(initialMomentum).setFinalMomentum(finalMomentum).setMomentum(momentum)
                        .setSwitchMomentumIteration(switchMomentumIteration).normalize(normalize)
                        .stopLyingIteration(stopLyingIteration).tolerance(tolerance).learningRate(learningRate)
                        .perplexity(perplexity).minGain(minGain).build();

        final double DELTA = 1e-15;

        assertEquals(theta, b.getTheta(), DELTA);
        assertEquals("invert", invert, b.isInvert());
        assertEquals("similarityFunctions", similarityFunctions, b.getSimiarlityFunction());
        assertEquals("maxIter", maxIter, b.maxIter);
        assertEquals(realMin, b.realMin, DELTA);
        assertEquals(initialMomentum, b.initialMomentum, DELTA);
        assertEquals(finalMomentum, b.finalMomentum, DELTA);
        assertEquals(momentum, b.momentum, DELTA);
        assertEquals("switchMomentumnIteration", switchMomentumIteration, b.switchMomentumIteration);
        assertEquals("normalize", normalize, b.normalize);
        assertEquals("stopLyingInMemoryLookupTable.javaIteration", stopLyingIteration, b.stopLyingIteration);
        assertEquals(tolerance, b.tolerance, DELTA);
        assertEquals(learningRate, b.learningRate, DELTA);
        assertEquals("useAdaGrad", useAdaGrad, b.useAdaGrad);
        assertEquals(perplexity, b.getPerplexity(), DELTA);
        assertEquals(minGain, b.minGain, DELTA);
    }

    @Test
    public void testPerplexity() throws Exception {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).setMaxIter(10).theta(0.5).learningRate(500)
                .useAdaGrad(false).build();

        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getTempFileFromArchive();
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(), "   ").get(NDArrayIndex.interval(0, 100),
                NDArrayIndex.interval(0, 784));

        INDArray perplexityOutput = b.computeGaussianPerplexity(data, 30.0);
        System.out.println(perplexityOutput);
    }

    @Test
    public void testReproducibility() {
        Nd4j.getRandom().setSeed(10);
        INDArray input = Nd4j.createFromArray(new double[]{ 0.4681,    0.2971,
                0.2938,    0.3655,
                0.3968,    0.0990,
                0.0796,    0.9245}).reshape(4,2);

        BarnesHutTsne b1 = new BarnesHutTsne.Builder().perplexity(1.0).build(),
                b2 = new BarnesHutTsne.Builder().perplexity(1.0).build();
        b1.setSimiarlityFunction(Distance.EUCLIDIAN.toString());
        b2.setSimiarlityFunction(Distance.EUCLIDIAN.toString());

        b1.fit(input);
        INDArray ret1 = b1.getData();

        Nd4j.getRandom().setSeed(10);
        b2.fit(input);
        INDArray ret2 = b2.getData();
        assertEquals(ret1, ret2);
    }

    @Ignore
    @Test
    public void testCorrectness() throws IOException {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(20.0).numDimension(55).learningRate(500)
                .useAdaGrad(false).build();

        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getTempFileFromArchive();
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(), "   ");
        StopWatch watch = new StopWatch();
        watch.start();
        b.fit(data);
        watch.stop();
        System.out.println("Fit done in " + watch);
        assertEquals(2500, b.getData().size(0));
        System.out.println(b.getData());

        INDArray a1 = b.getData().getRow(0);
        INDArray a2 = b.getData().getRow(1);
        INDArray a3 = b.getData().getRow(1000);
        INDArray a4 = b.getData().getRow(2498);
        INDArray a5 = b.getData().getRow(2499);

        INDArray expectedRow0 = Nd4j.createFromArray(new double[]{   167.8292,   32.5092,   75.6999,  -27.1170,   17.6490,  107.4103,   46.2925,    0.4640,  -30.7644,   -5.6178,   18.9462,    0.0773,   16.9440,   82.9042,   82.0447,   57.1004,  -65.7106,   21.9009,   31.2762,  -46.9130,  -79.2331,  -47.1991,  -84.3263,   53.6706,   90.2068,  -35.2406,  -39.4955,  -34.6930,  -27.5715,   -4.8603, -126.0396,  -58.8744, -101.5482,   -0.2450,  -12.1293,   74.7684,   69.9875,  -42.2529,  -23.4274,   24.8436,    1.4931,    3.3617,  -85.8046,   31.6360,   29.9752, -118.0233,   65.4318,  -16.9101,   65.3177,  -37.1838,   21.2493,   32.0591,    2.8582,  -62.2490,  -61.2909});
        INDArray expectedRow1 = Nd4j.createFromArray(new double[]{   32.3478,  118.7499,   -5.2345,   18.1522,   -5.7661,   55.0841,   19.1792,    0.6082,   18.7637,  145.1893,   56.9232,   95.6905,    0.6450,   54.9728,  -47.6037,   18.9907,   44.9000,   62.0607,   11.3163,   12.5538,   71.6602,   62.7464,   26.8367,    9.9804,   21.2930,   26.7346,  -25.4178,    0.8815,  127.8388,   95.7059,   61.8721,  198.7351,    3.7012,   38.8855,   56.8623,   -1.9203,  -21.2366,   26.3412,  -15.0002,   -5.5686,  -70.1437,  -75.2662,    5.2471,   32.7884,    9.0304,   25.5222,   52.0305,  -25.6134,   48.3513,   24.0128,  -15.4485, -139.3574,    7.2340,   82.3224,   12.1519});
        INDArray expectedRow1000 = Nd4j.createFromArray(new double[]{  30.8645,  -15.0904,   -8.3493,    3.7487,  -24.4678,    8.1096,   42.3257,   15.6477,  -45.1260,   31.5830,   40.2178,  -28.7947,  -83.6021,   -4.2135,   -9.8731,    0.3819,   -5.6642,  -34.0559,  -67.8494,  -33.4919,   -0.6254,    6.2422,  -56.9254,  -16.5402,   52.7575,  -72.3746,   18.7587,  -47.5842,   12.8834,  -20.3063,   21.7613,  -59.9718,    9.4924,   49.3242,  -36.5622,  -83.7369,   24.9921,   20.6678,    0.0452,  -69.3666,   13.2417,  -63.0318,    8.8107,  -34.4605,   -7.9497,  -12.0326,   27.4876,   -5.1647,    0.4363,  -24.6792,   -7.2241,   47.9472,   16.9052,   -8.1184,  -35.9715});
        INDArray expectedRow2498 = Nd4j.createFromArray(new double[]{  -0.0919, -153.8959,  -51.5028,  -73.8650,   -0.1183,  -14.4633,  -13.5049,   43.3787,   80.7100,    3.4296,   16.9782,  -75.3470,  103.3307,   13.8846,   -6.9218,   96.0892,    6.9730,   -2.1582,  -24.3647,   39.9077,  -10.5426, -135.5623,   -3.5470,   27.1481,  -24.0933,  -47.3872,    4.5534, -118.1384, -100.2693,  -64.9634,  -85.7244,   64.6426,  -48.8833,  -31.1378,  -93.3141,   37.8991,    8.5912,  -58.7564,   93.5057,   43.7609,  -34.8800,  -26.4699,  -37.5039,   10.8743,   22.7238,  -46.8137,   22.4390,  -12.9343,   32.6593,  -11.9136, -123.9708,   -5.3310,  -65.2792,  -72.1379,   36.7171});
        INDArray expectedRow2499 = Nd4j.createFromArray(new double[]{  -48.1854,   54.6014,   61.4287,    7.2306,   67.0068,   97.8297,   79.4408,   40.5714,  -18.2712,   -0.4891,   36.9610,   70.8634,  109.1919,  -28.6810,   13.5949,   -4.6143,   11.4054,  -95.5810,   20.6512,   77.8442,   33.2472,   53.7065,    4.3208,  -85.9796,   38.1717,   -9.6965,   44.0203,    1.0427,  -17.6281,  -54.7104,  -88.1742,  -24.6297,   33.5158,  -10.4808,   16.7051,   21.7057,   42.1260,   61.4450,   -9.4028,  -68.3737,   18.8957,   45.0714,   14.3170,   84.0521,   80.0860,  -15.4343,  -73.6115,  -15.5358,  -41.5067,  -55.7111,    0.1811,  -75.5584,   16.4112, -128.0799,  119.3907});

        assertArrayEquals(expectedRow0.toDoubleVector(), b.getData().getRow(0).toDoubleVector(), 1e-4);
        assertArrayEquals(expectedRow1.toDoubleVector(), b.getData().getRow(1).toDoubleVector(), 1e-4);
        assertArrayEquals(expectedRow1000.toDoubleVector(), b.getData().getRow(1000).toDoubleVector(), 1e-4);
        assertArrayEquals(expectedRow2498.toDoubleVector(), b.getData().getRow(2498).toDoubleVector(), 1e-4);
        assertArrayEquals(expectedRow2499.toDoubleVector(), b.getData().getRow(2499).toDoubleVector(), 1e-4);
    }

    @Test
    public void testCorrectness1() {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);

        double[] aData = new double[]{
                0.2999816948164936, 0.26252049735806526, 0.2673853427498767, 0.8604464129156685, 0.4802652829902563, 0.10959096539488711, 0.7950242948008909, 0.5917848948003486,
                0.2738285999345498, 0.9519684328285567, 0.9690024759209738, 0.8585615547624705, 0.8087760944312002, 0.5337951589543348, 0.5960876109129123, 0.7187130179825856,
                0.4629777327445964, 0.08665909175584818, 0.7748005397731237, 0.48020186965468536, 0.24927351841378798, 0.32272599988270445, 0.306414968984427, 0.6980212149215657,
                0.7977183964212472, 0.7673513094629704, 0.1679681724796478, 0.3107359484804584, 0.021701726051792103, 0.13797462786662518, 0.8618953518813538, 0.841333838365635,
                0.5284957375170422, 0.9703367685039823, 0.677388096913733, 0.2624474979832243, 0.43740966353106536, 0.15685545957858893, 0.11072929134449871, 0.06007395961283357,
                0.4093918718557811,  0.9563909195720572, 0.5994144944480242, 0.8278927844215804, 0.38586830957105667, 0.6201844716257464, 0.7603829079070265, 0.07875691596842949,
                0.08651136699915507, 0.7445210640026082, 0.6547649514127559, 0.3384719042666908, 0.05816723105860,0.6248951423054205, 0.7431868493349041};
        INDArray data = Nd4j.createFromArray(aData).reshape(11,5);

        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(250).setMaxIter(2).perplexity(3.0).theta(0.5).
                invertDistanceMetric(false).similarityFunction(Distance.EUCLIDIAN.toString())
                .setMomentum(0.5).learningRate(200)
                .useAdaGrad(false).staticInit(data).build();

        b.fit(data);
        System.out.println(b.getData());

        double[] expectedData = new double[]{ 95.7422,    204.1027,    93.4250,   37.2965,  720.5467,
                                                201.7626,  281.1831,  112.6211,  253.5442,  281.5215,
                                                298.2705,  341.9751,  138.9898,  306.1721,  253.5843,
                                                216.1541,  163.5028,   72.5626,  285.4470,  217.4066,
                                                261.8770,  287.8564,  146.9964,  430.8185,  355.4268,
                                                125.8167,  138.6875,   59.9295,   88.6333,  115.2455,
                                                183.1120,  185.3690,  154.1721,  136.4739,  139.4099,
                                                396.2162,  135.6869,  124.6483,  146.5401,  277.8906,
                                                582.9872,  453.6362,  349.6344,  407.7697,  392.8740,
                                                228.0742,  240.4862,  110.6878,  272.0466,  313.7355,
                                                172.7794,  117.0965,   35.9576,  226.7186,  188.8074};
        INDArray expectedArray = Nd4j.createFromArray(expectedData).reshape(11,5);
        for (int i = 0; i < expectedArray.rows(); ++i)
            assertArrayEquals(expectedArray.getRow(i).toDoubleVector(), b.getData().getRow(i).toDoubleVector(), 1e-3);
    }

    @Test
    public void testComputePerplexity() {
        double[] input = new double[]{0.3000, 0.2625, 0.2674, 0.8604, 0.4803,
                0.1096, 0.7950, 0.5918, 0.2738, 0.9520,
                0.9690, 0.8586, 0.8088, 0.5338, 0.5961,
                0.7187, 0.4630, 0.0867, 0.7748, 0.4802,
                0.2493, 0.3227, 0.3064, 0.6980, 0.7977,
                0.7674, 0.1680, 0.3107, 0.0217, 0.1380,
                0.8619, 0.8413, 0.5285, 0.9703, 0.6774,
                0.2624, 0.4374, 0.1569, 0.1107, 0.0601,
                0.4094, 0.9564, 0.5994, 0.8279, 0.3859,
                0.6202, 0.7604, 0.0788, 0.0865, 0.7445,
                0.6548, 0.3385, 0.0582, 0.6249, 0.7432};
        INDArray ndinput = Nd4j.createFromArray(input).reshape(11, 5);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(3.0).similarityFunction(Distance.EUCLIDIAN.toString()).invertDistanceMetric(false).theta(0.5)
                .useAdaGrad(false).build();
        b.computeGaussianPerplexity(ndinput, 3.0);
        INDArray expectedRows = Nd4j.createFromArray(new int[]{0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99});
        INDArray expectedValues = Nd4j.createFromArray(new double[]{0.6200, 0.1964, 0.1382, 0.0195, 0.0089,
                0.0084, 0.0033, 0.0026, 0.0026, 0.5877, 0.2825, 0.0810, 0.0149, 0.0122, 0.0115,
                0.0042, 0.0035, 0.0025, 0.6777, 0.1832, 0.0402, 0.0294, 0.0216, 0.0199, 0.0117,
                0.0084, 0.0078, 0.6771, 0.1662, 0.0604, 0.0465, 0.0169, 0.0146, 0.0064, 0.0061,
                0.0059, 0.6278, 0.2351, 0.0702, 0.0309, 0.0123, 0.0092, 0.0082, 0.0043, 0.0019,
                0.7123, 0.0786, 0.0706, 0.0672, 0.0290, 0.0178, 0.0148, 0.0055, 0.0042, 0.5267,
                0.3304, 0.1093, 0.0185, 0.0070, 0.0064, 0.0011, 0.0007, 3.1246e-5, 0.7176, 0.0874,
                0.0593, 0.0466, 0.0329, 0.0299, 0.0134, 0.0106, 0.0023, 0.6892, 0.1398, 0.0544,
                0.0544, 0.0287, 0.0210, 0.0072, 0.0033, 0.0021, 0.6824, 0.1345, 0.0871, 0.0429,
                0.0254, 0.0169, 0.0072, 0.0019, 0.0016, 0.6426, 0.1847, 0.1090, 0.0347, 0.0133,
                0.0051, 0.0038, 0.0038, 0.0030});
        assertArrayEquals(expectedRows.toIntVector(), b.getRows().toIntVector());
        assertArrayEquals(expectedValues.toDoubleVector(), b.getVals().toDoubleVector(), 1e-4);
    }

    @Test
    public void testSymmetrized() {
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(3.0).similarityFunction(Distance.EUCLIDIAN.toString()).invertDistanceMetric(false).theta(0.5)
                .useAdaGrad(false).build();
        INDArray expectedSymmetrized = Nd4j.createFromArray(new double[]{0.6239, 0.1813, 0.12359999999999999, 0.03695, 0.00795, 0.03385, 0.0074, 0.0158, 0.0013, 0.0042, 0.0074, 0.3093, 0.2085, 0.051000000000000004, 0.00895, 0.016050000000000002, 0.00245, 0.00705, 0.00125, 0.0021, 0.016050000000000002, 0.6022, 0.1615, 0.0233, 0.0183, 0.0108, 0.0068000000000000005, 0.0042, 0.011300000000000001, 0.00115, 0.1813, 0.00125, 0.0233, 0.65985, 0.0653, 0.0779, 0.03565, 0.05085, 0.038349999999999995, 0.026250000000000002, 0.6239, 0.3093, 0.0068000000000000005, 0.0653, 0.2099, 0.0205, 0.0173, 0.007300000000000001, 0.0171, 0.0089, 0.0158, 0.011300000000000001, 0.038349999999999995, 0.71495, 0.04775, 0.03615, 0.0089, 0.00275, 0.0021, 1.5623E-5, 0.00795, 0.00245, 0.6022, 0.0779, 0.007300000000000001, 0.5098, 0.015899999999999997, 0.00135, 1.5623E-5, 0.03385, 0.00705, 0.026250000000000002, 0.0171, 0.71495, 0.06515, 0.018349999999999998, 0.00775, 0.00115, 0.03695, 0.051000000000000004, 0.1615, 0.03565, 0.0205, 0.00275, 0.5098, 0.00775, 0.0055, 0.0026, 0.0013, 0.2085, 0.0183, 0.05085, 0.0173, 0.04775, 0.00135, 0.06515, 0.0026, 0.35855, 0.12359999999999999, 0.00895, 0.0108, 0.65985, 0.2099, 0.03615, 0.015899999999999997, 0.018349999999999998, 0.0055, 0.35855});
        INDArray rowsP = Nd4j.createFromArray(new int[]{0,         9,        18,        27,        36,        45,        54,        63,        72,        81,        90,        99});
        INDArray colsP = Nd4j.createFromArray(new int[]{4,         3,        10,         8,         6,         7,         1,         5,         9,         4,         9,         8,        10,         2,         0,         6,         7,         3,         6,         8,         3,         9,        10,         1,         4,         0,         5,        10,         0,         4,         6,         8,         9,         2,         5,         7,         0,        10,         3,         1,         8,         9,         6,         7,         2,         7,         9,         3,        10,         0,         4,         2,         8,         1,         2,         8,         3,        10,         0,         4,         9,         1,         5,         5,         9,         0,         3,        10,         4,         8,         1,         2,         6,         2,         0,         3,         4,         1,        10,         9,         7,        10,         1,         3,         7,         4,         5,         2,         8,         6,         3,         4,         0,         9,         6,         5,         8,         7,         1});
        INDArray valsP = Nd4j.createFromArray(new double[]{0.6200,    0.1964,    0.1382,    0.0195,    0.0089,    0.0084,    0.0033,    0.0026,    0.0026,    0.5877,    0.2825,    0.0810,    0.0149,    0.0122,    0.0115,    0.0042,    0.0035,    0.0025,    0.6777,    0.1832,    0.0402,    0.0294,    0.0216,    0.0199,    0.0117,    0.0084,    0.0078,    0.6771,    0.1662,    0.0604,    0.0465,    0.0169,    0.0146,    0.0064,    0.0061,    0.0059,    0.6278,    0.2351,    0.0702,    0.0309,    0.0123,    0.0092,    0.0082,    0.0043,    0.0019,    0.7123,    0.0786,    0.0706,    0.0672,    0.0290,    0.0178,    0.0148,    0.0055,    0.0042,    0.5267,    0.3304,    0.1093,    0.0185,    0.0070,    0.0064,    0.0011,    0.0007, 3.1246e-5,    0.7176,    0.0874,    0.0593,    0.0466,    0.0329,    0.0299,    0.0134,    0.0106,    0.0023,    0.6892,    0.1398,    0.0544,    0.0544,    0.0287,    0.0210,    0.0072,    0.0033,    0.0021,    0.6824,    0.1345,    0.0871,    0.0429,    0.0254,    0.0169,    0.0072,    0.0019,    0.0016,    0.6426,    0.1847,    0.1090,    0.0347,    0.0133,    0.0051,    0.0038,    0.0038,    0.0030});
        b.setN(11);
        INDArray actualSymmetrized = b.symmetrized(rowsP, colsP, valsP);
        System.out.println("Symmetrized from Java:" + actualSymmetrized);
        assertArrayEquals(expectedSymmetrized.toDoubleVector(), actualSymmetrized.toDoubleVector(), 1e-4);

        INDArray outRowsP = Nd4j.create(new int[]{rowsP.rows(),rowsP.columns()});
        INDArray outColsP = Nd4j.create(new int[]{colsP.rows(),colsP.columns()});
        BarnesHutSymmetrize op = new BarnesHutSymmetrize(rowsP, colsP, valsP, 11, outRowsP, outColsP);
        Nd4j.getExecutioner().exec(op);
        INDArray output = op.getResult();
        valsP = output;
        System.out.println("Symmetrized from C++: " + valsP);
        assertArrayEquals(expectedSymmetrized.toDoubleVector(), valsP.toDoubleVector(), 1e-2);
    }

    @Test
    public void testVPTree() {
        MemoryWorkspace workspace = new DummyWorkspace();
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            double[] d = new double[]{0.3000, 0.2625, 0.2674, 0.8604, 0.4803,
                    0.1096, 0.7950, 0.5918, 0.2738, 0.9520,
                    0.9690, 0.8586, 0.8088, 0.5338, 0.5961,
                    0.7187, 0.4630, 0.0867, 0.7748, 0.4802,
                    0.2493, 0.3227, 0.3064, 0.6980, 0.7977,
                    0.7674, 0.1680, 0.3107, 0.0217, 0.1380,
                    0.8619, 0.8413, 0.5285, 0.9703, 0.6774,
                    0.2624, 0.4374, 0.1569, 0.1107, 0.0601,
                    0.4094, 0.9564, 0.5994, 0.8279, 0.3859,
                    0.6202, 0.7604, 0.0788, 0.0865, 0.7445,
                    0.6548, 0.3385, 0.0582, 0.6249, 0.7432};
            VPTree tree = new VPTree(Nd4j.createFromArray(d).reshape(11,5), "euclidean", 1, false);
            INDArray target = Nd4j.createFromArray(new double[]{0.3000, 0.2625, 0.2674, 0.8604, 0.4803});
            List<DataPoint> results = new ArrayList<>();
            List<Double> distances = new ArrayList<>();
            tree.search(target, 11, results, distances);
            System.out.println("Results:" + results);
            System.out.println("Distances:" + distances);
        }
    }
}
