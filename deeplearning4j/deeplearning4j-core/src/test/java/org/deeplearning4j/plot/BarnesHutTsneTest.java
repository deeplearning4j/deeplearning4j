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
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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



        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream()).subList(0, 100);
        //b.fit(data);

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

    @Test
    public void testCorrectness() throws IOException {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(20.0).numDimension(55).learningRate(500)
                .useAdaGrad(false).build();

        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getTempFileFromArchive();
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(), "   ");

        b.fit(data);
        assertEquals(2500, b.getData().size(0));

        INDArray expectedRow0 = Nd4j.createFromArray(new double[]{   42.3586,   -3.2896,   24.2033,  -26.8413,    1.5113,   26.0389,    0.5837,   -3.8597,   12.3350,   10.3633,    1.1458,   20.1349,   16.3618,    5.6676,   15.3672,    2.3500,   -7.9997,    4.1572,   13.7659,  -17.2263,    5.8056,  -19.1045,    5.1821,   30.1253,    4.9556,  -12.8043,   -0.2874,   -4.9177,    3.2284,   -2.5196,  -10.4413,  -12.4002,   -4.5725,    5.0985,   16.8147,   33.2440,    9.9996,   -8.2500,    2.8619,   19.9243,   30.7432,   -1.6374,   -7.6915,    1.0639,   -0.0596,  -12.5669,   19.8835,    1.3108,   24.5337,  -15.5811,  -21.2687,   -7.5473,    9.2393,    2.5907,  -14.5563});
        INDArray expectedRow1 = Nd4j.createFromArray(new double[]{   12.8423,   18.8895,   15.0496,   13.9231,  -15.0565,   20.5179,    1.6432,   -6.4921,   -4.9840,   16.1826,   -1.2147,   18.6915,    4.2875,   10.2552,   -5.4427,   -3.4955,    4.6719,   23.4445,   -1.2010,    2.7575,   27.1310,   11.8212,   -6.3938,   -5.0025,   -0.3263,    8.1470,    0.3631,  -13.0426,   23.9079,   13.5081,    7.9130,   36.1028,   -1.1092,    5.5588,    9.8434,    8.8446,  -13.3515,   -0.2960,    8.0915,   22.9880,   -3.6494,  -13.4725,   -2.6862,    7.2024,    4.8478,  -10.0562,   16.4487,    9.0356,   15.1204,   -4.3699,   -5.5829,  -16.9541,   -1.7274,   15.5648,    1.7009});
        INDArray expectedRow1000 = Nd4j.createFromArray(new double[]{  1.1113,   -8.7674,   -4.7539,  -13.7207,   -3.0804,   -1.8222,   17.9112,   20.6517,  -11.4729,   32.3955,   16.4783,  -17.2506,  -15.8604,    5.1881,    8.8596,    3.4119,  -18.1463,   -7.6121,   -1.2889,   -5.3524,   16.1047,   -1.7479,  -21.8792,    8.7193,   15.5989,  -21.3258,   14.8743,  -20.7292,   -9.5401,    6.9035,  -10.3227,   -4.4259,   23.7533,    1.4588,  -21.2481,  -35.4483,   13.3771,  -20.4865,  -20.1713,  -34.3735,    2.6543,  -13.4780,    7.8703,   -9.6158,   10.0727,   13.2017,  -12.6660,   -9.2686,   10.2452,   -0.5687,  -13.0109,   34.6765,   -8.3372,   -2.2234,   -9.3815});
        INDArray expectedRow2498 = Nd4j.createFromArray(new double[]{  -2.7110,  -23.4090,   -8.5700,  -16.7072,   -8.9913,   -9.8487,  -13.7769,   14.7504,   16.8088,    0.1806,   -6.2842,  -25.6749,   26.1366,    3.2840,  -12.8232,   20.6678,    4.5163,  -16.0944,   13.0143,   16.4009,    2.3655,  -25.9959,    3.3085,    4.5080,   -8.4237,  -12.1552,  -12.9152,  -22.7430,   -9.3277,    4.0951,   -7.4336,   17.9464,    4.3807,   -5.2627,  -10.9030,   17.5921,   -3.8680,  -28.6230,   14.4064,    0.3269,  -22.3898,   -7.1473,   -7.3966,   -6.6756,   17.0603,    2.5220,    8.2308,   -8.1403,    3.3613,   -9.0916,  -10.6272,  -40.8022,  -12.3844,  -23.2882,    3.8440});
        INDArray expectedRow2499 = Nd4j.createFromArray(new double[]{  -4.8649,    7.7868,   15.3263,    6.9569,   12.4598,    1.0802,   10.7137,   -7.4892,   15.5141,   -4.5898,   -0.7246,   23.8491,   34.8549,    6.8501,   -8.6615,  -15.2462,   13.3019,  -32.4607,   -5.0742,    3.8444,   25.7100,   12.0716,    1.2711,   -5.2634,   -6.1325,    6.3916,   10.1777,    0.9926,  -20.4353,  -10.4664,  -11.7761,   -5.8192,    4.9823,    1.4449,   -3.1619,    5.7115,    7.5068,    6.4698,  -21.1572,  -30.7836,    9.4256,   -2.5953,   -7.4734,   21.5668,   24.9595,   -1.6078,  -12.0812,    1.9977,   -8.2209,   -6.4152,   -3.9645,   -5.0339,   13.9072,  -23.9383,    7.9930});

        assertArrayEquals(expectedRow0.toDoubleVector(), b.getData().getRow(0).toDoubleVector(), 1e-4);
        assertArrayEquals(expectedRow1.toDoubleVector(), b.getData().getRow(1).toDoubleVector(), 1e-4);
        assertArrayEquals(expectedRow1000.toDoubleVector(), b.getData().getRow(1000).toDoubleVector(), 1e-4);
        assertArrayEquals(expectedRow2498.toDoubleVector(), b.getData().getRow(2498).toDoubleVector(), 1e-4);
        assertArrayEquals(expectedRow2499.toDoubleVector(), b.getData().getRow(2499).toDoubleVector(), 1e-4);
    }

    @Test
    public void testCorrectness1() throws IOException {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(3.0).theta(0.5)
                .useAdaGrad(false).build();

        double[] aData = new double[]{
                0.2999816948164936, 0.26252049735806526, 0.2673853427498767, 0.8604464129156685, 0.4802652829902563, 0.10959096539488711, 0.7950242948008909, 0.5917848948003486,
                0.2738285999345498, 0.9519684328285567, 0.9690024759209738, 0.8585615547624705, 0.8087760944312002, 0.5337951589543348, 0.5960876109129123, 0.7187130179825856,
                0.4629777327445964, 0.08665909175584818, 0.7748005397731237, 0.48020186965468536, 0.24927351841378798, 0.32272599988270445, 0.306414968984427, 0.6980212149215657,
                0.7977183964212472, 0.7673513094629704, 0.1679681724796478, 0.3107359484804584, 0.021701726051792103, 0.13797462786662518, 0.8618953518813538, 0.841333838365635,
                0.5284957375170422, 0.9703367685039823, 0.677388096913733, 0.2624474979832243, 0.43740966353106536, 0.15685545957858893, 0.11072929134449871, 0.06007395961283357,
                0.4093918718557811,  0.9563909195720572, 0.5994144944480242, 0.8278927844215804, 0.38586830957105667, 0.6201844716257464, 0.7603829079070265, 0.07875691596842949,
                0.08651136699915507, 0.7445210640026082, 0.6547649514127559, 0.3384719042666908, 0.05816723105860,0.6248951423054205, 0.7431868493349041};
        INDArray data = Nd4j.createFromArray(aData).reshape(11,5);

        b.fit(data);
        System.out.println(b.getData());
    }

    @Test
    public void testComputePerplexity() {
        double[] input = new double[]{  0.3000,    0.2625,    0.2674,    0.8604,    0.4803,
                0.1096,    0.7950,    0.5918,    0.2738,    0.9520,
                0.9690,    0.8586,    0.8088,    0.5338,    0.5961,
                0.7187,    0.4630,    0.0867,    0.7748,    0.4802,
                0.2493,    0.3227,    0.3064,    0.6980,    0.7977,
                0.7674,    0.1680,    0.3107,    0.0217,    0.1380,
                0.8619,    0.8413,    0.5285,    0.9703,    0.6774,
                0.2624,    0.4374,    0.1569,    0.1107,    0.0601,
                0.4094,    0.9564,    0.5994,    0.8279,    0.3859,
                0.6202,    0.7604,    0.0788,    0.0865,    0.7445,
                0.6548,    0.3385,    0.0582,    0.6249,    0.7432};
        INDArray ndinput = Nd4j.createFromArray(input).reshape(11,5);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(3.0).similarityFunction(Distance.EUCLIDIAN.toString()).theta(0.5)
                .useAdaGrad(false).build();
        b.computeGaussianPerplexity(ndinput, 3.0);

        System.out.println(b.getRows());
        System.out.println(b.getCols());
        System.out.println(b.getVals());
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
