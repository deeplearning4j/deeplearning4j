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

import com.google.common.util.concurrent.AtomicDouble;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.sptree.SpTree;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.nn.gradient.Gradient;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.resources.Resources;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.nd4j.linalg.factory.Nd4j.zeros;

// import org.nd4j.jita.conf.CudaEnvironment;

/**
 * Created by agibsonccc on 10/1/14.
 */
@Slf4j
public class BarnesHutTsneTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Before
    public void setUp() {
        //   CudaEnvironment.getInstance().getConfiguration().enableDebug(true).setVerbose(false);
    }

    @Test
    public void testBarnesHutRun() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
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

        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(250).setMaxIter(200).perplexity(3.0).theta(0.5).numDimension(5).
                invertDistanceMetric(false).similarityFunction(Distance.EUCLIDIAN.toString())
                .setMomentum(0.5).learningRate(200).staticInit(data).setSwitchMomentumIteration(250)
                .useAdaGrad(false).build();

        b.fit(data);
        log.info("Result: {}", b.getData());
        
        val exp = Nd4j.createFromArray(new double[]{-3.5318212819287327, 35.40331834897696, 3.890809489531651, -1.291195609955519, -42.854099388207466, 7.8761368019456635, 28.798057251442877, 7.1456564000935225, 2.9518396278984786, -42.860181054199636, -34.989343304202, -108.99770355680282, 31.78123839126566, -29.322118879730205, 163.87558311206212, 2.9538984612478396, 31.419519824305546, 13.105400907817279, 25.46987139120746, -43.27317406736858, 32.455151773056144, 25.28067703547214, 0.005442008567682552, 21.005029233370358, -61.71390311950051, 5.218417653362599, 47.15762099517554, 8.834739256343404, 17.845790108867153, -54.31654219224107, -18.71285871476804, -16.446982180909007, -71.22568781913213, -12.339975548387091, 70.49096598213703, 25.022454385237456, -14.572652938207126, -5.320080866729078, 1.5874449933639676, -40.60960510287835, -31.98564381157643, -95.40875746933808, 19.196346639002364, -38.80930682421929, 135.00454225923906, 5.277879540549592, 30.79963767087089, -0.007276462027131683, 31.278796123365815, -38.47381680049993, 10.415728497075905, 36.567265019013085, -7.406587944733211, -18.376174615781114, -45.26976962854271}).reshape(-1, 5);
        assertEquals(exp, b.getData());
    }

    @Test
    public void testTsne() throws Exception {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).setMaxIter(10).theta(0.5).learningRate(500)
                        .useAdaGrad(false).build();

        File f = Resources.asFile("/deeplearning4j-core/mnist2500_X.txt");
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(), "   ").get(NDArrayIndex.interval(0, 100),
                        NDArrayIndex.interval(0, 784));

        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream()).subList(0, 100);
        b.fit(data);
        File outDir = testDir.newFolder();
        b.saveAsFile(labelsList, new File(outDir, "out.txt").getAbsolutePath());
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
        BarnesHutTsne b = new BarnesHutTsne.Builder().perplexity(20.0).numDimension(2).learningRate(200).setMaxIter(50)
                .useAdaGrad(false).build();

        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getTempFileFromArchive();
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(), "   ");
        StopWatch watch = new StopWatch();
        watch.start();
        b.fit(data);
        System.out.println(b.getData());
        watch.stop();
        File outDir = testDir.newFolder();
        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream());
        b.saveAsFile(/*labelsList,*/ new File(outDir, "raw.txt").getAbsolutePath());
        System.out.println(b.getData());

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

        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(250).setMaxIter(20).perplexity(3.0).theta(0.5).numDimension(5).
                invertDistanceMetric(false).similarityFunction(Distance.EUCLIDIAN.toString())
                .setMomentum(0.5).learningRate(200).staticInit(data)
                .useAdaGrad(false).build();

        b.fit(data);
        System.out.println(b.getData());

        /*double[] expectedData = new double[]{15.5392794313924, 19.25226403656672, -5.194955746137196, -31.787679714614757, 48.8674725273665,
                24.92775755686273, -22.621939920239065, -29.790772278125395, 19.027362415188914, -16.013800175884274,
                -27.454680593309185, 1.2929960811295493, -40.45000061571038, 61.23261682914338, 5.62278768938746,
                -28.16665244970911, -20.05502814088798, 12.803274346870865, -24.877262522905497, 45.115883138175874,
                21.597495694710616, 18.63254779638783, -4.029728632528419, -0.4596087279592638, -42.35340705500429,
                -69.24727547461491, 40.94332685199673, -24.60866142208024, 17.689874972878723, -3.6779759693605314,
                -30.91803590368529, 10.645452930824145, 36.58583235020565, -64.74975614289316, -39.364099390585956,
                72.54886481127016, -35.30663155696714, 19.37116912936714, -7.790876543092118, 19.6586396288508,
                58.1332709511154, -18.49217368496203, -3.5050200971182424, 5.662891294031322, 39.69533295638775,
                -15.114610550011662, -32.42366951357609, 17.039297537056537, 42.25610885633673, -2.7013781552769904,
                -16.338582630617925, 41.734027526336874, 20.941332646863426, -3.2145240561108244, -45.36033539684912};*/
        double[] expectedData = {40.93810899235225, 50.90183660191448, -14.298857560948981, -86.2012232604988, 129.51281793466023,
                66.29136854264247, -61.650213611972326, -80.42836756633497, 50.28325210727952, -44.29008119040566,
                -74.82748570869279, 2.0170536250746807, -109.21462846594635, 162.3973196127918, 14.000621153511705,
                -76.30892822919527, -54.251704596942275, 33.99763310539589, -67.6307009607032, 119.50868525237786,
                57.17786598853867, 49.1489174572297, -11.25663463504983, -2.38899196609398, -114.27194947404686,
                -185.93832011474473, 108.9022579845252, -66.14099037301474, 47.13683038425694, -10.037893631405792,
                -83.88458799629637, 26.985651418254996, 96.68139337135332, -174.2832443285551, -106.0999118697521,
                193.02622700008175, -94.88003359113081, 51.39502524568139, -20.96021960048648, 52.32291574424741,
                154.33973608321477, -50.90644802585217, -10.345744416395354, 13.721222143380892, 105.2111073677489,
                -41.339268919407345, -87.73042354938127, 45.306865238870046, 112.53877133856602, -8.44454352074299,
                -44.660828600669056, 110.72662022978719, 55.74660833987147, -9.613556053471232, -122.19953914048916};

        INDArray expectedArray = Nd4j.createFromArray(expectedData).reshape(11,5);
        for (int i = 0; i < expectedArray.rows(); ++i)
            assertArrayEquals(expectedArray.getRow(i).toDoubleVector(), b.getData().getRow(i).toDoubleVector(), 1e-2);
    }

    @Test
    public void testComputePerplexity() {
        double[] input = new double[]{0.2999816948164936, 0.26252049735806526, 0.2673853427498767, 0.8604464129156685, 0.4802652829902563, 0.10959096539488711, 0.7950242948008909, 0.5917848948003486,
                0.2738285999345498, 0.9519684328285567, 0.9690024759209738, 0.8585615547624705, 0.8087760944312002, 0.5337951589543348, 0.5960876109129123, 0.7187130179825856,
                0.4629777327445964, 0.08665909175584818, 0.7748005397731237, 0.48020186965468536, 0.24927351841378798, 0.32272599988270445, 0.306414968984427, 0.6980212149215657,
                0.7977183964212472, 0.7673513094629704, 0.1679681724796478, 0.3107359484804584, 0.021701726051792103, 0.13797462786662518, 0.8618953518813538, 0.841333838365635,
                0.5284957375170422, 0.9703367685039823, 0.677388096913733, 0.2624474979832243, 0.43740966353106536, 0.15685545957858893, 0.11072929134449871, 0.06007395961283357,
                0.4093918718557811, 0.9563909195720572, 0.5994144944480242, 0.8278927844215804, 0.38586830957105667, 0.6201844716257464, 0.7603829079070265, 0.07875691596842949,
                0.08651136699915507, 0.7445210640026082, 0.6547649514127559, 0.3384719042666908, 0.05816723105860, 0.6248951423054205, 0.7431868493349041};
        INDArray ndinput = Nd4j.createFromArray(input).reshape(11, 5);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(3.0).similarityFunction(Distance.EUCLIDIAN.toString()).invertDistanceMetric(false).theta(0.5)
                .useAdaGrad(false).build();
        b.computeGaussianPerplexity(ndinput, 3.0);
        INDArray expectedRows = Nd4j.createFromArray(new int[]{0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99});
        INDArray expectedCols = Nd4j.createFromArray(new int[] {4, 3, 10, 8, 6, 7, 1, 5, 9, 4, 9, 8, 10, 2, 0, 6, 7, 3, 6, 8, 3, 9, 10, 1, 4, 0, 5, 10, 0, 4, 6, 8, 9, 2, 5, 7, 0, 10, 3, 1, 8, 9, 6, 7, 2, 7, 9, 3, 10, 0, 4, 2, 8, 1, 2, 8, 3, 10, 0, 4, 9, 1, 5, 5, 9, 0, 3, 10, 4, 8, 1, 2, 6, 2, 0, 3, 4, 1, 10, 9, 7, 10, 1, 3, 7, 4, 5, 2, 8, 6, 3, 4, 0, 9, 6, 5, 8, 7, 1});
        INDArray expectedValues = Nd4j.createFromArray(new double[]{0.6199394088807811, 0.1964597878478939, 0.13826096288374987, 0.019500202354103796, 0.00892011933324624, 0.008390894278481041, 0.00333353509170543, 0.0026231979968002537, 0.0025718913332382506, 0.5877813741023542, 0.2824053513290301, 0.08100641562340703, 0.014863269403258283, 0.01219532549481422, 0.011522812905961816, 0.004243949243254114, 0.0034625890823446427, 0.002518912815575669, 0.6776991917357972, 0.18322100043035286, 0.040180871517768765, 0.02941481903928284, 0.021638322103495665, 0.019899251613183868, 0.011684443899339756, 0.008438621670147969, 0.007823477990631192, 0.6771051692354304, 0.16616561426152007, 0.06038657043891834, 0.04649900136463559, 0.01688479525099354, 0.014596215509122025, 0.006410339053808227, 0.006075759373243866, 0.005876535512328113, 0.6277958923349469, 0.23516301304728018, 0.07022275517450298, 0.030895020584550934, 0.012294459258033335, 0.009236709512467177, 0.00821667460222265, 0.0043013613064171955, 0.0018741141795786528, 0.7122763773574693, 0.07860063708191449, 0.07060648172121314, 0.06721282603559373, 0.028960026354739106, 0.017791245039439314, 0.01482510169996304, 0.005496178688168659, 0.004231126021499254, 0.5266697563046261, 0.33044733058681547, 0.10927281903651001, 0.018510201893239094, 0.006973656012751928, 0.006381768970069082, 0.0010596892780182746, 6.535010081417198E-4, 3.127690982824874E-5, 0.7176189632561156, 0.08740746743997298, 0.059268842313360166, 0.04664131589557433, 0.03288791302822797, 0.029929724912968133, 0.013368915822982491, 0.010616377319500762, 0.0022604800112974647, 0.689185362462809, 0.13977758696450715, 0.05439663822300743, 0.05434167873889952, 0.028687383013327405, 0.02099540802182275, 0.0072154477293594615, 0.0032822412915506907, 0.0021182535547164334, 0.6823844384306867, 0.13452128016104092, 0.08713547969428868, 0.04287399325857787, 0.025452813990877978, 0.016881841237860937, 0.0072200814416566415, 0.0019232561582331975, 0.0016068156267770154, 0.6425943207872832, 0.18472852256294967, 0.1089653923564887, 0.03467849453890959, 0.013282484305873534, 0.005149863792637524, 0.0037974408302766656, 0.003787710699822367, 0.003015770125758626});
        assertArrayEquals(expectedCols.toIntVector(), b.getCols().toIntVector());
        assertArrayEquals(expectedRows.toIntVector(), b.getRows().toIntVector());
        assertArrayEquals(expectedValues.toDoubleVector(), b.getVals().toDoubleVector(), 1e-5);
    }

    @Test
    public void testComputeGradient() {
        double[] input = new double[]{0.3000,    0.2625,    0.2674,    0.8604,    0.4803,
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
        INDArray ndinput = Nd4j.createFromArray(input).reshape(11, 5);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(3.0).similarityFunction(Distance.EUCLIDIAN.toString()).invertDistanceMetric(false).theta(0.5)
                .useAdaGrad(false).staticInit(ndinput).build();
        b.setY(ndinput);
        b.setN(11);

        INDArray rowsP = Nd4j.createFromArray(new int[]{0,         9,        18,        27,        36,        45,        54,        63,        72,        81,        90,        99});
        INDArray colsP = Nd4j.createFromArray(new int[]{4,         3,        10,         8,         6,         7,         1,         5,         9,         4,         9,         8,        10,         2,         0,         6,         7,         3,         6,         8,         3,         9,        10,         1,         4,         0,         5,        10,         0,         4,         6,         8,         9,         2,         5,         7,         0,        10,         3,         1,         8,         9,         6,         7,         2,         7,         9,         3,        10,         0,         4,         2,         8,         1,         2,         8,         3,        10,         0,         4,         9,         1,         5,         5,         9,         0,         3,        10,         4,         8,         1,         2,         6,         2,         0,         3,         4,         1,        10,         9,         7,        10,         1,         3,         7,         4,         5,         2,         8,         6,         3,         4,         0,         9,         6,         5,         8,         7,         1});
        INDArray valsP = Nd4j.createFromArray(new double[]{0.6200,    0.1964,    0.1382,    0.0195,    0.0089,    0.0084,    0.0033,    0.0026,    0.0026,    0.5877,    0.2825,    0.0810,    0.0149,    0.0122,    0.0115,    0.0042,    0.0035,    0.0025,    0.6777,    0.1832,    0.0402,    0.0294,    0.0216,    0.0199,    0.0117,    0.0084,    0.0078,    0.6771,    0.1662,    0.0604,    0.0465,    0.0169,    0.0146,    0.0064,    0.0061,    0.0059,    0.6278,    0.2351,    0.0702,    0.0309,    0.0123,    0.0092,    0.0082,    0.0043,    0.0019,    0.7123,    0.0786,    0.0706,    0.0672,    0.0290,    0.0178,    0.0148,    0.0055,    0.0042,    0.5267,    0.3304,    0.1093,    0.0185,    0.0070,    0.0064,    0.0011,    0.0007, 3.1246e-5,    0.7176,    0.0874,    0.0593,    0.0466,    0.0329,    0.0299,    0.0134,    0.0106,    0.0023,    0.6892,    0.1398,    0.0544,    0.0544,    0.0287,    0.0210,    0.0072,    0.0033,    0.0021,    0.6824,    0.1345,    0.0871,    0.0429,    0.0254,    0.0169,    0.0072,    0.0019,    0.0016,    0.6426,    0.1847,    0.1090,    0.0347,    0.0133,    0.0051,    0.0038,    0.0038,    0.0030});

        b.setRows(rowsP);
        b.setCols(colsP);
        b.setVals(valsP);
        Gradient gradient = b.gradient();

        double[] dC = {-0.0618386320333619, -0.06266654959379839, 0.029998268806149204, 0.10780566335888186, -0.19449543068355346, -0.14763764361792697, 0.17493572758118422, 0.1926109839221966, -0.15176648259935419, 0.10974665709698186, 0.13102419155322598, 0.004941641352409449, 0.19159764518354974, -0.26332838053474944, -0.023631441261541583, 0.09838669432305949, 0.09709129638394683, -0.01605053000727605, 0.06566171635025217, -0.17325078066035252, -0.1090854255505605, 0.023350644966904276, 0.075192354899586, -0.08278373866517603, 0.18431338134579323, 0.2766031655578053, -0.17557907233268688, 0.10616148241800637, -0.09999024423215641, -0.017181932145255287, 0.06711331400576945, -0.01388231800826619, -0.10248189290485302, 0.20786521034824304, 0.11254913977572988, -0.289564646781519, 0.13491805919337516, -0.07504249344962562, 0.004154656287570634, -0.10516715438388784, -0.27984655075804576, 0.09811828071286613, 0.03684521473995052, -0.054645216532387256, -0.18147132772800725, 0.027588750493223044, 0.214734364419479, -0.026729138234415008, -0.28410504978879136, 0.007015481601883835, 0.04427981739424874, -0.059253265830134655, -0.05325479031206952, -0.11319889109674944, 0.1530133971867549};
        INDArray actual = gradient.getGradientFor("yIncs");
        System.out.println(actual);
        assertArrayEquals(dC, actual.reshape(1,55).toDoubleVector(), 1e-05);
    }

    @Test
    public void testApplyGradient() {
        double[] Y = new double[]{0.2999816948164936, 0.26252049735806526, 0.2673853427498767, 0.8604464129156685, 0.4802652829902563, 0.10959096539488711, 0.7950242948008909, 0.5917848948003486,
                0.2738285999345498, 0.9519684328285567, 0.9690024759209738, 0.8585615547624705, 0.8087760944312002, 0.5337951589543348, 0.5960876109129123, 0.7187130179825856,
                0.4629777327445964, 0.08665909175584818, 0.7748005397731237, 0.48020186965468536, 0.24927351841378798, 0.32272599988270445, 0.306414968984427, 0.6980212149215657,
                0.7977183964212472, 0.7673513094629704, 0.1679681724796478, 0.3107359484804584, 0.021701726051792103, 0.13797462786662518, 0.8618953518813538, 0.841333838365635,
                0.5284957375170422, 0.9703367685039823, 0.677388096913733, 0.2624474979832243, 0.43740966353106536, 0.15685545957858893, 0.11072929134449871, 0.06007395961283357,
                0.4093918718557811, 0.9563909195720572, 0.5994144944480242, 0.8278927844215804, 0.38586830957105667, 0.6201844716257464, 0.7603829079070265, 0.07875691596842949,
                0.08651136699915507, 0.7445210640026082, 0.6547649514127559, 0.3384719042666908, 0.05816723105860, 0.6248951423054205, 0.7431868493349041};
        INDArray ndinput = Nd4j.createFromArray(Y).reshape(11,5);

        double[] gradient = {   -0.0635,   -0.0791,    0.0228,    0.1360,   -0.2016,
                   -0.1034,    0.0976,    0.1266,   -0.0781,    0.0707,
                    0.1184,   -0.0018,    0.1719,   -0.2529,   -0.0209,
                    0.1204,    0.0855,   -0.0530,    0.1069,   -0.1860,
                   -0.0890,   -0.0763,    0.0181,    0.0048,    0.1798,
                    0.2917,   -0.1699,    0.1038,   -0.0736,    0.0159,
                    0.1324,   -0.0409,   -0.1502,    0.2738,    0.1668,
                   -0.3012,    0.1489,   -0.0801,    0.0329,   -0.0817,
                   -0.2405,    0.0810,    0.0171,   -0.0201,   -0.1638,
                    0.0656,    0.1383,   -0.0707,   -0.1757,    0.0144,
                    0.0708,   -0.1725,   -0.0870,    0.0160,    0.1921};
        INDArray ndgrad = Nd4j.createFromArray(gradient).reshape(11, 5);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(3.0).similarityFunction(Distance.EUCLIDIAN.toString())
                .invertDistanceMetric(false).theta(0.5).learningRate(200)
                .useAdaGrad(false).staticInit(ndinput).build();
        b.setY(ndinput);
        b.setN(11);
        INDArray yIncs = Nd4j.zeros(DataType.DOUBLE, ndinput.shape());
        b.setYIncs(yIncs);
        INDArray gains = Nd4j.zeros(DataType.DOUBLE, ndinput.shape());
        b.setGains(gains);
        b.update(ndgrad, "yIncs");

        double[] expected = {2.54, 3.164, -0.912, -5.44, 8.064, 4.136, -3.9040000000000004, -5.064, 3.124, -2.828, -4.736000000000001, 0.072, -6.8759999999999994, 10.116, 0.836, -4.816, -3.4200000000000004, 2.12, -4.276, 7.4399999999999995, 3.5599999999999996, 3.0520000000000005, -0.7240000000000001, -0.19199999999999998, -7.191999999999999, -11.668000000000001, 6.795999999999999, -4.152, 2.944, -0.636, -5.295999999999999, 1.636, 6.008, -10.952, -6.672000000000001, 12.048000000000002, -5.956, 3.204, -1.3159999999999998, 3.268, 9.62, -3.24, -0.684, 0.804, 6.552, -2.624, -5.532, 2.828, 7.028, -0.576, -2.832, 6.8999999999999995, 3.4799999999999995, -0.64, -7.683999999999999};
        assertArrayEquals(expected, b.getYIncs().reshape(55).toDoubleVector(), 1e-5);
    }

    @Test
    public void testComputeEdgeForces() {
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
        SpTree tree = new SpTree(ndinput);
        INDArray rows = Nd4j.createFromArray(new int[]{0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99});
        INDArray cols = Nd4j.createFromArray(new int[]{4, 3, 10, 8, 6, 7, 1, 5, 9, 4, 9, 8, 10, 2, 0, 6, 7, 3, 6, 8, 3, 9, 10, 1, 4, 0, 5, 10, 0, 4, 6, 8, 9, 2, 5, 7, 0, 10, 3, 1, 8, 9, 6, 7, 2, 7, 9, 3, 10, 0, 4, 2, 8, 1, 2, 8, 3, 10, 0, 4, 9, 1, 5, 5, 9, 0, 3, 10, 4, 8, 1, 2, 6, 2, 0, 3, 4, 1, 10, 9, 7, 10, 1, 3, 7, 4, 5, 2, 8, 6, 3, 4, 0, 9, 6, 5, 8, 7, 1});
        INDArray vals = Nd4j.createFromArray(new double[]{0.6200, 0.1964, 0.1382, 0.0195, 0.0089, 0.0084, 0.0033, 0.0026, 0.0026, 0.5877, 0.2825, 0.0810, 0.0149, 0.0122, 0.0115, 0.0042, 0.0035, 0.0025, 0.6777, 0.1832, 0.0402, 0.0294, 0.0216, 0.0199, 0.0117, 0.0084, 0.0078, 0.6771, 0.1662, 0.0604, 0.0465, 0.0169, 0.0146, 0.0064, 0.0061, 0.0059, 0.6278, 0.2351, 0.0702, 0.0309, 0.0123, 0.0092, 0.0082, 0.0043, 0.0019, 0.7123, 0.0786, 0.0706, 0.0672, 0.0290, 0.0178, 0.0148, 0.0055, 0.0042, 0.5267, 0.3304, 0.1093, 0.0185, 0.0070, 0.0064, 0.0011, 0.0007, 3.1246e-5, 0.7176, 0.0874, 0.0593, 0.0466, 0.0329, 0.0299, 0.0134, 0.0106, 0.0023, 0.6892, 0.1398, 0.0544, 0.0544, 0.0287, 0.0210, 0.0072, 0.0033, 0.0021, 0.6824, 0.1345, 0.0871, 0.0429, 0.0254, 0.0169, 0.0072, 0.0019, 0.0016, 0.6426, 0.1847, 0.1090, 0.0347, 0.0133, 0.0051, 0.0038, 0.0038, 0.0030});
        int N = 11;
        INDArray posF = Nd4j.create(ndinput.shape());
        tree.computeEdgeForces(rows, cols, vals, N, posF);
        double[] expectedPosF = {-0.08017022778816381, -0.08584612446002386, 0.024041740837932417, 0.13353853518214748, -0.19989209255196486, -0.17059164865362167, 0.18730152809351328, 0.20582835656173232, -0.1652505189678666, 0.13123839113710167, 0.15511476126066306, 0.021425546153174206, 0.21755440369356663, -0.2628756936897519, -0.021079609911707077, 0.11455959658671841, 0.08803186126822704, -0.039212116057989604, 0.08800854045636688, -0.1795568260613919, -0.13265313037184673, 0.0036829788349159154, 0.07205631770917967, -0.06873974602987808, 0.20446419876515043, 0.28724205607738795, -0.19397780156808536, 0.10457369548573531, -0.12340830629973816, -0.03634773269456816, 0.0867775929922852, 0.0029761730963277894, -0.09131897988004745, 0.2348924028566898, 0.12026408931908775, -0.30400848137321873, 0.1282943410872978, -0.08487864823843354, -0.017561758195375168, -0.13082811573092396, -0.2885857462722986, 0.12469730654026252, 0.05408469871148934, -0.03417740859260864, -0.19261929748672968, 0.03318694717819495, 0.22818123908045765, -0.044944593551341956, -0.3141734963080852, 0.020297428845239652, 0.05442118949793863, -0.07890301602838638, -0.07823705950336371, -0.10455483898962027, 0.16980714813230746};
        INDArray indExpectedPositive = Nd4j.createFromArray(expectedPosF).reshape(11, 5);
        assertEquals(indExpectedPositive, posF);

        AtomicDouble sumQ = new AtomicDouble(0.0);
        double theta = 0.5;
        INDArray negF = Nd4j.create(ndinput.shape());

        double[][] neg = {{-1.6243229118532043, -2.0538918185758117, -0.5277950148630416, 2.280133920112387, -0.4781864949257863},
        {-2.033904565482581, 1.0957067439325718, 1.1711627018218371, -1.1947911960637323, 1.904335906364157},
        {2.134613094178481, 1.4606030267537151, 2.299972033488509, 0.040111598796927175, 0.22611223726312565},
        {1.4330457669590706, -0.8027368824700638, -2.052297868677289, 1.9801035811739054, -0.5587649959721402},
        {-2.088283171473531, -1.7427092080895168, -0.27787744880128185, 1.2444077055013942, 1.7855201950031347},
        {0.9426889976629138, -1.6302714638583877, -0.14069035384185855, -2.075023651861262, -1.698239988087389},
        {1.7424090804808496, 1.493794306111751, 0.989121494481274, 2.394820866756112, 0.6836049340540907},
        {-1.279836833417519, -0.5869132848699253, -0.871560326864079, -1.9242443527432451, -2.273762088892443},
        {-0.7743611464510498, 2.3551097898757134, 1.527553257122278, 1.813608037002701, -0.9877974041073948},
        {0.49604405759812625, 1.1914983778171337, -1.6140319597311803, -2.6642997837396654, 1.1768845173097966},
        {0.8986049706740562, -1.7411217160869163, -2.213624650045752, 0.7659306956507013, 1.4880578211349607}};

        double expectedSumQ = 88.60782954084712;

        for (int n = 0; n < N; n++) {
            tree.computeNonEdgeForces(n, theta, negF.slice(n), sumQ);
            assertArrayEquals(neg[n], negF.slice(n).toDoubleVector(), 1e-05);
        }
        assertEquals(expectedSumQ, sumQ.get(), 1e-05);
    }

    /*
    @Test
    public void testSymmetrized() {
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).perplexity(3.0).similarityFunction(Distance.EUCLIDIAN.toString()).invertDistanceMetric(false).theta(0.5)
                .useAdaGrad(false).build();
        INDArray expectedSymmetrized = Nd4j.createFromArray(new double[]{0.6239, 0.1813, 0.12359999999999999, 0.03695, 0.00795, 0.03385, 0.0074, 0.0158, 0.0013, 0.0042, 0.0074, 0.3093, 0.2085, 0.051000000000000004, 0.00895, 0.016050000000000002, 0.00245, 0.00705, 0.00125, 0.0021, 0.016050000000000002, 0.6022, 0.1615, 0.0233, 0.0183, 0.0108, 0.0068000000000000005, 0.0042, 0.011300000000000001, 0.00115, 0.1813, 0.00125, 0.0233, 0.65985, 0.0653, 0.0779, 0.03565, 0.05085, 0.038349999999999995, 0.026250000000000002, 0.6239, 0.3093, 0.0068000000000000005, 0.0653, 0.2099, 0.0205, 0.0173, 0.007300000000000001, 0.0171, 0.0089, 0.0158, 0.011300000000000001, 0.038349999999999995, 0.71495, 0.04775, 0.03615, 0.0089, 0.00275, 0.0021, 1.5623E-5, 0.00795, 0.00245, 0.6022, 0.0779, 0.007300000000000001, 0.5098, 0.015899999999999997, 0.00135, 1.5623E-5, 0.03385, 0.00705, 0.026250000000000002, 0.0171, 0.71495, 0.06515, 0.018349999999999998, 0.00775, 0.00115, 0.03695, 0.051000000000000004, 0.1615, 0.03565, 0.0205, 0.00275, 0.5098, 0.00775, 0.0055, 0.0026, 0.0013, 0.2085, 0.0183, 0.05085, 0.0173, 0.04775, 0.00135, 0.06515, 0.0026, 0.35855, 0.12359999999999999, 0.00895, 0.0108, 0.65985, 0.2099, 0.03615, 0.015899999999999997, 0.018349999999999998, 0.0055, 0.35855});
        INDArray rowsP = Nd4j.createFromArray(new int[]{0,         9,        18,        27,        36,        45,        54,        63,        72,        81,        90,        99});
        INDArray colsP = Nd4j.createFromArray(new int[]{4,         3,        10,         8,         6,         7,         1,         5,         9,         4,         9,         8,        10,         2,         0,         6,         7,         3,         6,         8,         3,         9,        10,         1,         4,         0,         5,        10,         0,         4,         6,         8,         9,         2,         5,         7,         0,        10,         3,         1,         8,         9,         6,         7,         2,         7,         9,         3,        10,         0,         4,         2,         8,         1,         2,         8,         3,        10,         0,         4,         9,         1,         5,         5,         9,         0,         3,        10,         4,         8,         1,         2,         6,         2,         0,         3,         4,         1,        10,         9,         7,        10,         1,         3,         7,         4,         5,         2,         8,         6,         3,         4,         0,         9,         6,         5,         8,         7,         1});
        INDArray valsP = Nd4j.createFromArray(new double[]{0.6200,    0.1964,    0.1382,    0.0195,    0.0089,    0.0084,    0.0033,    0.0026,    0.0026,    0.5877,    0.2825,    0.0810,    0.0149,    0.0122,    0.0115,    0.0042,    0.0035,    0.0025,    0.6777,    0.1832,    0.0402,    0.0294,    0.0216,    0.0199,    0.0117,    0.0084,    0.0078,    0.6771,    0.1662,    0.0604,    0.0465,    0.0169,    0.0146,    0.0064,    0.0061,    0.0059,    0.6278,    0.2351,    0.0702,    0.0309,    0.0123,    0.0092,    0.0082,    0.0043,    0.0019,    0.7123,    0.0786,    0.0706,    0.0672,    0.0290,    0.0178,    0.0148,    0.0055,    0.0042,    0.5267,    0.3304,    0.1093,    0.0185,    0.0070,    0.0064,    0.0011,    0.0007, 3.1246e-5,    0.7176,    0.0874,    0.0593,    0.0466,    0.0329,    0.0299,    0.0134,    0.0106,    0.0023,    0.6892,    0.1398,    0.0544,    0.0544,    0.0287,    0.0210,    0.0072,    0.0033,    0.0021,    0.6824,    0.1345,    0.0871,    0.0429,    0.0254,    0.0169,    0.0072,    0.0019,    0.0016,    0.6426,    0.1847,    0.1090,    0.0347,    0.0133,    0.0051,    0.0038,    0.0038,    0.0030});
        b.setN(11);
        BarnesHutTsne.SymResult actualSymmetrized = b.symmetrized(rowsP, colsP, valsP);
        System.out.println("Symmetrized from Java:" + actualSymmetrized);
        System.out.println(actualSymmetrized.rows);
        System.out.println(actualSymmetrized.cols);
        assertArrayEquals(expectedSymmetrized.toDoubleVector(), actualSymmetrized.vals.toDoubleVector(), 1e-5);


        INDArray rowsFromCpp = Nd4j.create(new int[]{rowsP.rows(),rowsP.columns()}, DataType.INT);
        BarnesHutSymmetrize op = new BarnesHutSymmetrize(rowsP, colsP, valsP, 11, rowsFromCpp);
        Nd4j.getExecutioner().exec(op);
        INDArray valsFromCpp = op.getSymmetrizedValues();
        INDArray colsFromCpp = op.getSymmetrizedCols();
        System.out.println("Symmetrized from C++: " + valsP);
        assertArrayEquals(expectedSymmetrized.toDoubleVector(), valsFromCpp.toDoubleVector(), 1e-5);

        int[] expectedRows = new int[]{0, 10, 20, 30, 40, 50, 60, 69, 78, 88, 98, 108};
        int[] expectedCols = new int[]{4, 3, 10, 8, 6, 7, 1, 5, 9, 2, 0, 4, 9, 8, 10, 2, 6, 7, 3, 5, 1, 6, 8, 3, 9, 10, 4, 0, 5, 7, 0, 1, 2, 10, 4, 6, 8, 9, 5, 7, 0, 1, 2, 3, 10, 8, 9, 6, 7, 5, 0, 2, 3, 7, 9, 10, 4, 8, 1, 6, 0, 1, 2, 3, 4, 8, 10, 9, 5, 0, 1, 3, 4, 5, 9, 10, 8, 2, 0, 1, 2, 3, 4, 5, 6, 7, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        assertArrayEquals(expectedRows, rowsFromCpp.toIntVector());
        assertArrayEquals(expectedCols, colsFromCpp.toIntVector());
    }
     */

    @Test
    public void testVPTree() {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
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
            VPTree tree = new VPTree(Nd4j.createFromArray(d).reshape(11, 5), "euclidean", 1, false);
            INDArray target = Nd4j.createFromArray(new double[]{0.3000, 0.2625, 0.2674, 0.8604, 0.4803});
            List<DataPoint> results = new ArrayList<>();
            List<Double> distances = new ArrayList<>();
            tree.search(target, 11, results, distances);
            System.out.println("Results:" + results);
            System.out.println("Distances:" + distances);
        }
    }

    @Test
    public void testSpTree() {
            double[] input = new double[]{0.2999816948164936, 0.26252049735806526, 0.2673853427498767, 0.8604464129156685, 0.4802652829902563, 0.10959096539488711, 0.7950242948008909, 0.5917848948003486,
                    0.2738285999345498, 0.9519684328285567, 0.9690024759209738, 0.8585615547624705, 0.8087760944312002, 0.5337951589543348, 0.5960876109129123, 0.7187130179825856,
                    0.4629777327445964, 0.08665909175584818, 0.7748005397731237, 0.48020186965468536, 0.24927351841378798, 0.32272599988270445, 0.306414968984427, 0.6980212149215657,
                    0.7977183964212472, 0.7673513094629704, 0.1679681724796478, 0.3107359484804584, 0.021701726051792103, 0.13797462786662518, 0.8618953518813538, 0.841333838365635,
                    0.5284957375170422, 0.9703367685039823, 0.677388096913733, 0.2624474979832243, 0.43740966353106536, 0.15685545957858893, 0.11072929134449871, 0.06007395961283357,
                    0.4093918718557811, 0.9563909195720572, 0.5994144944480242, 0.8278927844215804, 0.38586830957105667, 0.6201844716257464, 0.7603829079070265, 0.07875691596842949,
                    0.08651136699915507, 0.7445210640026082, 0.6547649514127559, 0.3384719042666908, 0.05816723105860, 0.6248951423054205, 0.7431868493349041};
            INDArray ndinput = Nd4j.createFromArray(input).reshape(11, 5);

            double[] rows = {0, 10.0000, 20.0000, 30.0000, 40.0000, 50.0000, 60.0000, 69.0000, 78.0000, 88.0000, 98.0000, 108.0000};
            INDArray indRows = Nd4j.createFromArray(rows);
            double[] cols = {4.0000, 3.0000, 10.0000, 8.0000, 6.0000, 7.0000, 1.0000, 5.0000, 9.0000, 2.0000, 0, 4.0000, 9.0000, 8.0000, 10.0000, 2.0000, 6.0000, 7.0000, 3.0000, 5.0000, 1.0000, 6.0000, 8.0000, 3.0000, 9.0000, 10.0000, 4.0000, 0, 5.0000, 7.0000, 0, 1.0000, 2.0000, 10.0000, 4.0000, 6.0000, 8.0000, 9.0000, 5.0000, 7.0000, 0, 1.0000, 2.0000, 3.0000, 10.0000, 8.0000, 9.0000, 6.0000, 7.0000, 5.0000, 0, 2.0000, 3.0000, 7.0000, 9.0000, 10.0000, 4.0000, 8.0000, 1.0000, 6.0000, 0, 1.0000, 2.0000, 3.0000, 4.0000, 8.0000, 10.0000, 9.0000, 5.0000, 0, 1.0000, 3.0000, 4.0000, 5.0000, 9.0000, 10.0000, 8.0000, 2.0000, 0, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 10.0000, 9.0000, 0, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 10.0000, 0, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000};
            INDArray indCols = Nd4j.createFromArray(cols);
            double[] vals = {0.6806, 0.1978, 0.1349, 0.0403, 0.0087, 0.0369, 0.0081, 0.0172, 0.0014, 0.0046, 0.0081, 0.3375, 0.2274, 0.0556, 0.0098, 0.0175, 0.0027, 0.0077, 0.0014, 0.0023, 0.0175, 0.6569, 0.1762, 0.0254, 0.0200, 0.0118, 0.0074, 0.0046, 0.0124, 0.0012, 0.1978, 0.0014, 0.0254, 0.7198, 0.0712, 0.0850, 0.0389, 0.0555, 0.0418, 0.0286, 0.6806, 0.3375, 0.0074, 0.0712, 0.2290, 0.0224, 0.0189, 0.0080, 0.0187, 0.0097, 0.0172, 0.0124, 0.0418, 0.7799, 0.0521, 0.0395, 0.0097, 0.0030, 0.0023, 1.706e-5, 0.0087, 0.0027, 0.6569, 0.0850, 0.0080, 0.5562, 0.0173, 0.0015, 1.706e-5, 0.0369, 0.0077, 0.0286, 0.0187, 0.7799, 0.0711, 0.0200, 0.0084, 0.0012, 0.0403, 0.0556, 0.1762, 0.0389, 0.0224, 0.0030, 0.5562, 0.0084, 0.0060, 0.0028, 0.0014, 0.2274, 0.0200, 0.0555, 0.0189, 0.0521, 0.0015, 0.0711, 0.0028, 0.3911, 0.1349, 0.0098, 0.0118, 0.7198, 0.2290, 0.0395, 0.0173, 0.0200, 0.0060, 0.3911};
            INDArray indVals = Nd4j.createFromArray(vals);

            final int N = 11;
            INDArray posF = Nd4j.create(DataType.DOUBLE, ndinput.shape());
            SpTree tree = new SpTree(ndinput);
            tree.computeEdgeForces(indRows, indCols, indVals, N, posF);
            double[]expectedPosF = {-0.0818453583761987, -0.10231102631753211, 0.016809473355579547, 0.16176252194290375, -0.20703464777007444, -0.1263832139293613, 0.10996898963389254, 0.13983782727968627, -0.09164547825742625, 0.09219041827159041, 0.14252277104691244, 0.014676985587529433, 0.19786703075718223, -0.25244374832212546, -0.018387062879777892, 0.13652061663449183, 0.07639155593531936, -0.07616591260449279, 0.12919565310762643, -0.19229222179037395, -0.11250575155166542, -0.09598877143033444, 0.014899570740339653, 0.018867923701997365, 0.19996253097190828, 0.30233811684856743, -0.18830455752593392, 0.10223346521208224, -0.09703007177169608, -0.003280966942428477, 0.15213078827243462, -0.02397414389327494, -0.1390550777479942, 0.30088735606726813, 0.17456236098186903, -0.31560012032960044, 0.142309945794784, -0.08988089476622348, 0.011236280978163357, -0.10732740266565795, -0.24928551644245478, 0.10762735102220329, 0.03434270193250408, 2.831838829882295E-4, -0.17494982967210068, 0.07114328804840916, 0.15171552834583996, -0.08888924450773618, -0.20576831397087963, 0.027662749212463134, 0.08096437977846523, -0.19211185715249313, -0.11199893965092741, 0.024654692641180212, 0.20889407228258244};
            assertArrayEquals(expectedPosF, posF.reshape(1,55).toDoubleVector(), 1e-5);

            final double theta = 0.5;
            AtomicDouble sumQ = new AtomicDouble(0.0);
            INDArray negF = Nd4j.create(DataType.DOUBLE, ndinput.shape());
            for (int n = 0; n < N; n++) {
                INDArray prev = ((n == 0) ? negF.slice(n ): negF.slice(n-1));
                tree.computeNonEdgeForces(n, theta, negF.slice(0), sumQ);
            }

            double[] expectedNegF = {-0.15349944039348173, -0.9608688924710804, -1.7099994806905086, 2.6604989787415203, 1.2677709150619332, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            double expectedSum = 88.60715062760883;

            assertArrayEquals(expectedNegF, negF.reshape(1,55).toDoubleVector(), 1e-5);
            assertEquals(expectedSum, sumQ.get(), 1e-5);
    }

    @Test
    public void testZeroMean() {
        double[] aData = new double[]{
                0.2999816948164936, 0.26252049735806526, 0.2673853427498767, 0.8604464129156685, 0.4802652829902563, 0.10959096539488711, 0.7950242948008909, 0.5917848948003486,
                0.2738285999345498, 0.9519684328285567, 0.9690024759209738, 0.8585615547624705, 0.8087760944312002, 0.5337951589543348, 0.5960876109129123, 0.7187130179825856,
                0.4629777327445964, 0.08665909175584818, 0.7748005397731237, 0.48020186965468536, 0.24927351841378798, 0.32272599988270445, 0.306414968984427, 0.6980212149215657,
                0.7977183964212472, 0.7673513094629704, 0.1679681724796478, 0.3107359484804584, 0.021701726051792103, 0.13797462786662518, 0.8618953518813538, 0.841333838365635,
                0.5284957375170422, 0.9703367685039823, 0.677388096913733, 0.2624474979832243, 0.43740966353106536, 0.15685545957858893, 0.11072929134449871, 0.06007395961283357,
                0.4093918718557811,  0.9563909195720572, 0.5994144944480242, 0.8278927844215804, 0.38586830957105667, 0.6201844716257464, 0.7603829079070265, 0.07875691596842949,
                0.08651136699915507, 0.7445210640026082, 0.6547649514127559, 0.3384719042666908, 0.05816723105860,0.6248951423054205, 0.7431868493349041};
        INDArray ndinput = Nd4j.createFromArray(aData).reshape(11,5);
        BarnesHutTsne.zeroMean(ndinput);
        double[] expected = {-0.2384362257971937, -0.3014583649756485, -0.07747340086583643, 0.3347228669042438, -0.07021239883787267, -0.4288269552188002, 0.23104543246717713, 0.24692615118463546, -0.2518949460768749, 0.40149075100042775, 0.43058455530728645, 0.2945826924287568, 0.46391735081548713, 0.008071612942910145, 0.04560992908478334, 0.18029509736889826, -0.10100112958911733, -0.25819965185986493, 0.249076993761699, -0.07027581217344359, -0.28914440219989934, -0.2412528624510093, -0.03844377463128612, 0.17229766891014098, 0.24724071459311825, 0.22893338884928305, -0.39601068985406596, -0.034122795135254735, -0.5040218199596326, -0.4125030539615038, 0.3234774312676665, 0.2773549760319213, 0.18363699390132904, 0.44461322249255764, 0.12691041508560408, -0.275970422630463, -0.12656919880264839, -0.18800328403712419, -0.41499425466692597, -0.4904037222152954, -0.12902604875790624, 0.3924120572383435, 0.2545557508323111, 0.30216923841015575, -0.16460937225707228, 0.0817665510120591, 0.1964040455733127, -0.26610182764728363, -0.4392121790122696, 0.19404338217447925, 0.11634703079906861, -0.22550695806702292, -0.2866915125571131, 0.09917159629399586, 0.19270916750677514};
        assertArrayEquals(expected, ndinput.reshape(55).toDoubleVector(), 1e-5);
    }
}
