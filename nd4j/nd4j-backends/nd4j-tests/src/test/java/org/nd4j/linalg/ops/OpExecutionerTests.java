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

package org.nd4j.linalg.ops;

import lombok.val;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.exception.IllegalOpException;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMax;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldMulOp;
import org.nd4j.linalg.api.ops.random.impl.DropOut;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static org.junit.Assert.*;


/**
 * Created by agibsonccc on 2/22/15.
 */
@RunWith(Parameterized.class)
public class OpExecutionerTests extends BaseNd4jTest {


    public OpExecutionerTests(Nd4jBackend backend) {
        super(backend);
    }



    @Test
    public void testCosineSimilarity() {
        INDArray vec1 = Nd4j.create(new float[] {1, 2, 3, 4, 5});
        INDArray vec2 = Nd4j.create(new float[] {1, 2, 3, 4, 5});
        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(getFailureMessage(), 1, sim, 1e-1);

    }

    @Test
    public void testCosineDistance(){
        INDArray vec1 = Nd4j.create(new float[] {1, 2, 3});
        INDArray vec2 = Nd4j.create(new float[] {3, 5, 7});
        // 1-17*sqrt(2/581)
        double distance = Transforms.cosineDistance(vec1, vec2);
        assertEquals(getFailureMessage(), 0.0025851, distance, 1e-7);
    }

    @Test
    public void testEuclideanDistance() {
        INDArray arr = Nd4j.create(new double[] {55, 55});
        INDArray arr2 = Nd4j.create(new double[] {60, 60});
        double result = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(arr, arr2)).getFinalResult()
                .doubleValue();
        assertEquals(getFailureMessage(), 7.0710678118654755, result, 1e-1);
    }

    @Test
    public void testDimensionalEuclidean() {
        INDArray distanceInputRow = Nd4j.linspace(1, 4, 4);
        INDArray distanceComp = Nd4j.linspace(1, 4, 4).add(1);
        INDArray result = Nd4j.createUninitialized(4);
        Nd4j.getExecutioner().exec(
                new EuclideanDistance(distanceInputRow, distanceComp, result, distanceInputRow.length()), 0);
        INDArray euclideanAssertion = Nd4j.ones(4);
        assertEquals(euclideanAssertion, result);
        System.out.println(result);

    }


    @Test
    @Ignore
    public void testDistance() throws Exception {
        INDArray matrix = Nd4j.rand(new int[] {400,10});
        INDArray rowVector = matrix.getRow(70);
        INDArray resultArr = Nd4j.zeros(400,1);
        Executor executor = Executors.newSingleThreadExecutor();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                Nd4j.getExecutioner().exec(new EuclideanDistance(matrix, rowVector, resultArr, matrix.lengthLong()),
                        -1);
                System.out.println("Ran!");
            }
        });

        Thread.sleep(600000);

    }

    @Test
    public void testScalarMaxOp() {
        INDArray scalarMax = Nd4j.linspace(1, 6, 6).negi();
        INDArray postMax = Nd4j.ones(6);
        Nd4j.getExecutioner().exec(new ScalarMax(scalarMax, 1));
        assertEquals(getFailureMessage(), scalarMax, postMax);
    }

    @Test
    public void testSetRange() {
        INDArray linspace = Nd4j.linspace(1, 4, 4);
        Nd4j.getExecutioner().exec(new SetRange(linspace, 0, 1));
        for (int i = 0; i < linspace.length(); i++) {
            double val = linspace.getDouble(i);
            assertTrue(getFailureMessage(), val >= 0 && val <= 1);
        }

        INDArray linspace2 = Nd4j.linspace(1, 4, 4);
        Nd4j.getExecutioner().exec(new SetRange(linspace2, 2, 4));
        for (int i = 0; i < linspace2.length(); i++) {
            double val = linspace2.getDouble(i);
            assertTrue(getFailureMessage(), val >= 2 && val <= 4);
        }
    }

    @Test
    public void testNormMax() {
        INDArray arr = Nd4j.create(new float[] {1, 2, 3, 4});
        double normMax = Nd4j.getExecutioner().execAndReturn(new NormMax(arr)).getFinalResult().doubleValue();
        assertEquals(getFailureMessage(), 4, normMax, 1e-1);
    }

    @Test
    public void testLog() {
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{0., 1.09861229}, {0.69314718, 1.38629436}});

        INDArray logTest = Transforms.log(arr);
        assertEquals(assertion, logTest);
        arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        assertion = Nd4j.create(new double[][] {{0., 1.09861229, 1.60943791}, {0.69314718, 1.38629436, 1.79175947}});

        logTest = Transforms.log(arr);
        assertEquals(assertion, logTest);
    }


    @Test
    public void testNorm2() {
        INDArray arr = Nd4j.create(new float[] {1, 2, 3, 4});
        double norm2 = Nd4j.getExecutioner().execAndReturn(new Norm2(arr)).getFinalResult().doubleValue();
        assertEquals(getFailureMessage(), 5.4772255750516612, norm2, 1e-1);
    }

    @Test
    public void testAdd() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5, 2.0);
        opExecutioner.exec(new AddOp(new INDArray[]{x, xDup},new INDArray[]{x}));
        assertEquals(getFailureMessage(), solution, x);
    }

    @Test
    public void testMul() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5, 1.0);
        opExecutioner.exec(new OldMulOp(x, xDup, x));
        assertEquals(solution, x);
    }


    @Test
    public void testExecutioner() throws IllegalOpException {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5, 2.0);
        opExecutioner.exec(new AddOp(new INDArray[]{x, xDup},new INDArray[]{x}));
        assertEquals(getFailureMessage(), solution, x);
        Sum acc = new Sum(x.dup());
        opExecutioner.exec(acc);
        assertEquals(getFailureMessage(), 10.0, acc.getFinalResult().doubleValue(), 1e-1);
        Prod prod = new Prod(x.dup());
        opExecutioner.exec(prod);
        assertEquals(getFailureMessage(), 32.0, prod.getFinalResult().doubleValue(), 1e-1);
    }


    @Test
    public void testMaxMin() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);
        Max max = new Max(x);
        opExecutioner.exec(max);
        assertEquals(5, max.getFinalResult().doubleValue(), 1e-1);
        Min min = new Min(x);
        opExecutioner.exec(min);
        assertEquals(1, min.getFinalResult().doubleValue(), 1e-1);
    }

    @Test
    public void testProd() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        Prod prod = new Prod(linspace);
        double prod2 = Nd4j.getExecutioner().execAndReturn(prod).getFinalResult().doubleValue();
        assertEquals(720, prod2, 1e-1);
    }

    @Test
    public void testSum() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        Sum sum = new Sum(linspace);
        double sum2 = Nd4j.getExecutioner().execAndReturn(sum).getFinalResult().doubleValue();
        assertEquals(21, sum2, 1e-1);
    }


    @Test
    public void testDescriptiveStatsDouble() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);

        Mean mean = new Mean(x);
        opExecutioner.exec(mean);
        assertEquals(3.0, mean.getFinalResult().doubleValue(), 1e-1);

        Variance variance = new Variance(x.dup(), true);
        opExecutioner.exec(variance);
        assertEquals(getFailureMessage(), 2.5, variance.getFinalResult().doubleValue(), 1e-1);
    }



    @Test
    public void testIamax() {
        INDArray linspace = Nd4j.linspace(1, 4, 4);
        assertEquals(getFailureMessage(), 3, Nd4j.getBlasWrapper().iamax(linspace));
    }

    @Test
    public void testIamax2() {
        INDArray linspace = Nd4j.linspace(1, 4, 4);
        assertEquals(getFailureMessage(), 3, Nd4j.getBlasWrapper().iamax(linspace));
        val op = new IAMax(linspace);

        int iamax = Nd4j.getExecutioner().execAndReturn(op).getFinalResult();
        assertEquals(3, iamax);
    }


    @Test
    public void testDescriptiveStats() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);

        Mean mean = new Mean(x);
        opExecutioner.exec(mean);
        assertEquals(getFailureMessage(), 3.0, mean.getFinalResult().doubleValue(), 1e-1);

        Variance variance = new Variance(x.dup(), true);
        opExecutioner.exec(variance);
        assertEquals(getFailureMessage(), 2.5, variance.getFinalResult().doubleValue(), 1e-1);
    }

    @Test
    public void testRowSoftmax() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6);
        OldSoftMax softMax = new OldSoftMax(arr);
        opExecutioner.exec(softMax);
        assertEquals(getFailureMessage(), 1.0, softMax.z().sumNumber().doubleValue(), 1e-1);
    }


    @Test
    public void testPow() {
        INDArray oneThroughSix = Nd4j.linspace(1, 6, 6);
        Pow pow = new Pow(oneThroughSix, 2);
        Nd4j.getExecutioner().exec(pow);
        INDArray answer = Nd4j.create(new float[] {1, 4, 9, 16, 25, 36});
        assertEquals(getFailureMessage(), answer, pow.z());
    }


    @Test
    public void testComparisonOps() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        INDArray ones = Nd4j.ones(6);
        INDArray zeros = Nd4j.zeros(6);
        assertEquals(ones, Nd4j.getExecutioner().execAndReturn(new ScalarGreaterThan(linspace, 0)));
        assertEquals(zeros, Nd4j.getExecutioner().execAndReturn(new ScalarGreaterThan(linspace, 7)));
        assertEquals(zeros, Nd4j.getExecutioner().execAndReturn(new ScalarLessThan(linspace, 0)));
        assertEquals(ones, Nd4j.getExecutioner().execAndReturn(new ScalarLessThan(linspace, 7)));
    }

    @Test
    public void testScalarArithmetic() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        INDArray plusOne = Nd4j.linspace(2, 7, 6);
        Nd4j.getExecutioner().exec(new ScalarAdd(linspace, 1));
        assertEquals(plusOne, linspace);
    }


    @Test
    public void testDimensionMax() {
        INDArray linspace = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        int axis = 0;
        INDArray row = linspace.slice(axis);
        Max max = new Max(row);
        double max2 = Nd4j.getExecutioner().execAndReturn(max).getFinalResult().doubleValue();
        assertEquals(5.0, max2, 1e-1);

        Min min = new Min(row);
        double min2 = Nd4j.getExecutioner().execAndReturn(min).getFinalResult().doubleValue();
        assertEquals(1.0, min2, 1e-1);
    }


    @Test
    public void testStridedLog() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray slice = arr.slice(0);
        Log log = new Log(slice);
        opExecutioner.exec(log);
        INDArray assertion = Nd4j.create(Nd4j.createBuffer(new float[] {0.f, 1.09861229f, 1.60943791f}));
        assertEquals(getFailureMessage(), assertion, slice);
    }

    @Test
    public void testSoftmax() {
        INDArray vec = Nd4j.linspace(1, 6, 6);
        INDArray matrix = vec.dup().reshape('f', 2, 3);
        Nd4j.getExecutioner().exec(new OldSoftMax(matrix));
        INDArray matrixAssertion = Nd4j.create(
                new double[] {0.015876241, 0.015876241, 0.11731043, 0.11731043, 0.86681336, 0.86681336},
                new int[] {2, 3}, 'f');
        assertEquals(matrixAssertion, matrix);
    }

    @Test
    public void testOtherSoftmax() {
        INDArray vec = Nd4j.linspace(1, 18, 18);
        INDArray matrix = vec.dup().reshape('f', 3, 6);
        Nd4j.getExecutioner().exec(new OldSoftMax(matrix));
        INDArray assertion = Nd4j.create(new double[] {2.9067235E-7, 2.9067235E-7, 2.9067235E-7, 5.8383102E-6,
                5.8383102E-6, 5.8383102E-6, 1.1726559E-4, 1.1726559E-4, 1.1726559E-4, 0.0023553425,
                0.0023553425, 0.0023553425, 0.047308315, 0.047308315, 0.047308315, 0.95021296, 0.95021296,
                0.95021296}, new int[] {3, 6}, 'f');
        assertEquals(assertion, matrix);

    }

    @Test
    public void testClassificationSoftmax() {
        INDArray input = Nd4j.create(new double[] {-0.11537042, -0.12137824, -0.12023379, -0.121212654, -0.11363918,
                -0.10101747, -0.11571036, -0.11699755, -0.12303393, -0.12222538, -0.111205295, -0.11710347,
                -0.12319956, -0.12442437, -0.10528548, -0.08768979, -0.102969095, -0.11346512, -0.106075466,
                -0.106812954, -0.116048686, -0.107011676, -0.11420248, -0.111682974, -0.1161541, -0.12052244,
                -0.112824574, -0.115148716, -0.116811216, -0.11987898, -0.120540276, -0.11262567, -0.1033787,
                -0.09880979, -0.12222538, -0.11966099, -0.115003735, -0.12222538, -0.12269135, -0.11685945,
                -0.113694824, -0.116669476, -0.12075868, -0.106586955, -0.1025187, -0.119958475, -0.1087377,
                -0.120367825, -0.11125226, -0.11847404, 0.07354959, 0.06268422, 0.08751991, 0.05259514,
                0.07969017, 0.062334877, 0.070893124, -0.0064847446, 0.07025853, 0.036010355, 0.032280773,
                0.051330015, 0.048092365, 0.075383544, 0.0026740413, 0.060346432, 0.0642657, 0.032083385,
                0.0732262, 0.034285806, 0.08459604, 0.040570542, 0.08494349, 0.06835914, 0.05533401, 0.06346914,
                0.082844265, 0.097696416, 0.07128821, 0.0012981347, 0.03325705, 0.02408357, 0.03130123,
                0.0938182, 0.062283132, 0.04927382, 0.07896088, 0.06648662, 0.030163728, 0.047266718,
                0.057046715, 0.06862673, 0.041349716, 0.0029906097, 0.05075717, 0.031863913, 0.04317994,
                0.05359216, -0.026340045, 0.042290315, 0.12401425, 0.10255231, 0.11914651, 0.10838078,
                0.11920456, 0.12058236, 0.07964209, 0.11363033, 0.103594616, 0.124344714, 0.10481718,
                0.10615028, 0.116106674, 0.101516105, 0.11543927, 0.11498181, 0.10836467, 0.12498047,
                0.11773242, 0.080594674, 0.12140609, 0.10168961, 0.11630507, 0.097365394, 0.116597414,
                0.11525783, 0.09534653, 0.09552346, 0.114529714, 0.10820673, 0.1136818, 0.12088456, 0.11661099,
                0.09196414, 0.09367619, 0.12396192, 0.11715829, 0.10781159, 0.09206238, 0.11529949, 0.12193692,
                0.114719115, 0.10255231, 0.12246917, 0.122784376, 0.11647934, 0.0990758, 0.109394, 0.11121255,
                0.0993141, -0.20153984, -0.19392103, -0.19934568, -0.19083072, -0.2002219, -0.18812108,
                -0.19819337, -0.197516, -0.18787667, -0.19108538, -0.1998294, -0.19259658, -0.19106683,
                -0.1962341, -0.20643811, -0.17979848, -0.2008527, -0.2022663, -0.19437766, -0.19513921,
                -0.19446027, -0.19675982, -0.20814218, -0.19372806, -0.18230462, -0.18796727, -0.19594415,
                -0.19937015, -0.20221424, -0.1900377, -0.18905015, -0.20246184, -0.18973505, -0.19170408,
                -0.19108538, -0.20450068, -0.20772256, -0.19108538, -0.19349809, -0.19836158, -0.2043826,
                -0.16650638, -0.19694944, -0.19511233, -0.18056196, -0.19521531, -0.19218433, -0.19556037,
                -0.19890977, -0.19989866, 0.110895246, 0.092092186, 0.13636512, 0.09708373, 0.12663889,
                0.112808585, 0.104376495, 0.008250488, 0.11656463, 0.062448245, 0.07663194, 0.07671328,
                0.09773853, 0.12847707, 0.0019389617, 0.088733524, 0.106456585, 0.06874651, 0.12830634,
                0.06976124, 0.125978, 0.06455773, 0.14945641, 0.12600574, 0.088896096, 0.09622975, 0.13689917,
                0.15111934, 0.11476833, 0.012905663, 0.06886613, 0.056535408, 0.056539863, 0.16477236,
                0.105480224, 0.06795105, 0.12039946, 0.11954279, 0.052694187, 0.08551991, 0.11061126,
                0.11398445, 0.07550914, 0.023510661, 0.09092401, 0.060012117, 0.075267926, 0.08827078,
                -0.0351813, 0.073293045, 0.17944565, 0.16982268, 0.18865392, 0.18693334, 0.18788461, 0.20586023,
                0.13861816, 0.2043775, 0.18895179, 0.1654431, 0.1499911, 0.17463979, 0.17583887, 0.16696453,
                0.16749826, 0.1592366, 0.17954212, 0.18181926, 0.21207902, 0.15266305, 0.17395121, 0.15906093,
                0.21057776, 0.15467101, 0.1741476, 0.19151133, 0.14792839, 0.14762697, 0.18604177, 0.18808068,
                0.19654939, 0.17514956, 0.18510492, 0.16045001, 0.18320353, 0.1866908, 0.16069266, 0.17718756,
                0.14080217, 0.1681495, 0.17300007, 0.15283263, 0.16982268, 0.1817098, 0.16696706, 0.16177532,
                0.16047187, 0.16464046, 0.15210035, 0.16091332, 0.19544482, 0.1334318, 0.16168839, 0.11322637,
                0.19517516, 0.18929672, 0.17545202, 0.16658127, 0.0913124, 0.110042766, 0.20550777, 0.13831234,
                0.10610578, 0.12289246, 0.2714768, 0.20504126, 0.25187582, 0.20981915, 0.2013824, 0.19962603,
                0.15790766, 0.20949605, 0.23528615, 0.18096939, 0.08758451, 0.10911971, 0.18139267, 0.18525597,
                0.19391456, 0.11438081, 0.10939147, 0.22006747, 0.18334162, 0.21811464, 0.110042766, 0.19371074,
                0.2327902, 0.110042766, 0.11990617, 0.17242402, 0.2197558, 0.046736162, 0.14443715, 0.20759603,
                0.13962242, 0.1486803, 0.17288595, 0.14028643, 0.19978581, 0.17370181, -0.03870563,
                -0.038800463, -0.06074495, 0.005578231, -0.026154697, -0.09166621, -0.061155554, 0.008943881,
                -0.047770716, -0.012912758, -0.01086065, -0.019136615, -0.0061139315, -0.09119851, 0.034481727,
                -0.008211095, -0.09062709, -0.04640113, -0.003811527, -0.006515648, -0.06737341, 0.022067834,
                -0.07823941, -0.10467515, -0.012385383, -0.008899722, -0.05071889, -0.06124178, -0.053028326,
                0.036579777, 0.0040080342, 0.0017335843, 0.00966073, -0.13457713, -0.10622793, -0.058109,
                -0.042826377, -0.004804369, -0.05494748, -0.0023090728, -0.08317526, -0.0812492, 0.0012213364,
                0.017189149, -0.041634988, -0.07508251, -0.052436303, -0.028371753, 0.077994466, -0.02655043,
                -0.048018664, -0.113020286, -0.114139564, -0.17401274, -0.114431985, -0.19375473, -0.08697136,
                -0.22462575, -0.18594624, 0.029960819, -0.030721083, -0.10795041, -0.0687456, -0.088536546,
                -0.028004304, -0.0044010356, -0.14119366, -0.057321526, -0.23839925, -0.09940954, -0.03133001,
                -0.07696311, -0.23962286, -0.055784777, -0.07386551, -0.16175163, -0.04683064, -0.0713344,
                -0.12525225, -0.176231, -0.1785344, -0.054819535, -0.10787999, -0.12848954, -0.21946627,
                -0.07054794, -0.004379764, -0.14215486, -0.062456205, -0.038439542, -0.019706637, 0.041873105,
                -0.113020286, -0.06571138, 0.012915805, 0.008474745, -0.05855358, -0.058223557, -0.007257685,
                -0.11702956}, new int[] {150, 3}, 'f');
        INDArray assertion = Nd4j.create(new double[] {0.3046945, 0.31053564, 0.30772904, 0.3127982, 0.3049832,
                0.30736795, 0.30686057, 0.3076439, 0.31483504, 0.3129973, 0.3041549, 0.31072456, 0.31327236,
                0.31140107, 0.29749927, 0.3074947, 0.30005574, 0.30333498, 0.30530176, 0.30543298, 0.30866665,
                0.30427617, 0.3004194, 0.3066349, 0.31625876, 0.31319204, 0.30653065, 0.30584472, 0.30464056,
                0.3128697, 0.31325394, 0.30222437, 0.3077568, 0.30448923, 0.3129973, 0.3042575, 0.30053583,
                0.3129973, 0.31189594, 0.30701208, 0.30219892, 0.31960127, 0.30956632, 0.30446774, 0.31260762,
                0.3090533, 0.30814552, 0.31004447, 0.30479294, 0.30664116, 0.34107947, 0.34090835, 0.34337586,
                0.33338174, 0.3392553, 0.34375596, 0.34360147, 0.32999024, 0.34059194, 0.33567807, 0.3329864,
                0.33810434, 0.3335406, 0.34469903, 0.32986054, 0.33754894, 0.34487507, 0.33762568, 0.33529142,
                0.3337637, 0.34467727, 0.33267412, 0.34292668, 0.34477416, 0.33685294, 0.3374399, 0.34123695,
                0.34388787, 0.3415838, 0.3281285, 0.33251032, 0.33212858, 0.3328727, 0.34851202, 0.34613267,
                0.3428139, 0.3415714, 0.33493212, 0.3399977, 0.334378, 0.341975, 0.3439716, 0.3338435,
                0.32948583, 0.3386694, 0.3416325, 0.33997172, 0.33822724, 0.322459, 0.3372723, 0.34495273,
                0.34774572, 0.34917334, 0.35264698, 0.34931532, 0.35469893, 0.34396452, 0.35612476, 0.35239643,
                0.33876625, 0.34252962, 0.3474446, 0.3456883, 0.34547645, 0.34248832, 0.3409808, 0.35051847,
                0.34580123, 0.3572295, 0.34364316, 0.34337047, 0.34537002, 0.35722163, 0.34282026, 0.34652257,
                0.3524498, 0.34229505, 0.3448508, 0.34953663, 0.35266057, 0.3529821, 0.3454672, 0.3484543,
                0.34806335, 0.35400698, 0.34626326, 0.3412907, 0.35079524, 0.3440239, 0.34343404, 0.34242827,
                0.3368599, 0.34774572, 0.34608114, 0.3400059, 0.3396784, 0.3427608, 0.34453318, 0.34129536,
                0.34855416, 0.27953854, 0.2888062, 0.28432208, 0.2917625, 0.27968782, 0.28172797, 0.28256553,
                0.2838439, 0.2950681, 0.2921697, 0.27835938, 0.28813055, 0.29271683, 0.28982347, 0.26887837,
                0.28043702, 0.27207687, 0.27755985, 0.27949893, 0.27961233, 0.28538817, 0.27815753, 0.2734831,
                0.2824814, 0.29601505, 0.29276544, 0.2820821, 0.28114092, 0.27970335, 0.29167145, 0.29251158,
                0.27625751, 0.28229526, 0.27747792, 0.2921697, 0.27950907, 0.2739233, 0.2921697, 0.29057536,
                0.28298247, 0.2759991, 0.3040637, 0.28685635, 0.2786732, 0.2891384, 0.28664854, 0.28347546,
                0.28758636, 0.27921304, 0.2826625, 0.35405818, 0.35108265, 0.36056453, 0.3485483, 0.35556272,
                0.36155194, 0.35530117, 0.33488873, 0.35673428, 0.34467104, 0.34808716, 0.34679636, 0.35051754,
                0.36349487, 0.32961816, 0.34726825, 0.35973698, 0.35023382, 0.3542774, 0.3458166, 0.35923994,
                0.34075052, 0.3657791, 0.36523324, 0.3483503, 0.3486777, 0.36019015, 0.36275893, 0.3567635,
                0.33195946, 0.34456408, 0.34308356, 0.34138083, 0.3741388, 0.36141226, 0.34927687, 0.35602295,
                0.35318217, 0.34774497, 0.34741682, 0.36079222, 0.3599326, 0.3454444, 0.33631673, 0.35254955,
                0.35138544, 0.35105765, 0.35016224, 0.3196206, 0.3478924, 0.36461383, 0.37194347, 0.37430686,
                0.38146538, 0.37414935, 0.3862741, 0.3648603, 0.3899538, 0.38379708, 0.35297906, 0.35835785,
                0.37207472, 0.3669662, 0.36884367, 0.36079016, 0.35640973, 0.37637684, 0.36602545, 0.39257398,
                0.36932322, 0.36189535, 0.36576378, 0.3925363, 0.3630396, 0.36705002, 0.38037658, 0.36077514,
                0.36329508, 0.3754482, 0.38198447, 0.383479, 0.36473197, 0.37315765, 0.37273598, 0.38716227,
                0.36867967, 0.35647675, 0.37599605, 0.36120692, 0.36207274, 0.36036783, 0.34994662, 0.37194347,
                0.36720264, 0.3553651, 0.35541826, 0.36446443, 0.36410302, 0.3555394, 0.37069988, 0.41576695,
                0.4006582, 0.40794885, 0.3954393, 0.41532904, 0.41090405, 0.41057387, 0.40851218, 0.3900969,
                0.39483297, 0.41748565, 0.40114486, 0.39401078, 0.39877546, 0.4336224, 0.41206822, 0.42786735,
                0.4191052, 0.41519928, 0.4149547, 0.4059452, 0.41756633, 0.42609745, 0.4108837, 0.3877262,
                0.3940425, 0.41138723, 0.4130143, 0.41565615, 0.3954588, 0.39423448, 0.42151815, 0.40994796,
                0.41803288, 0.39483297, 0.4162334, 0.42554083, 0.39483297, 0.3975287, 0.41000548, 0.421802,
                0.37633502, 0.40357736, 0.41685906, 0.39825398, 0.40429813, 0.40837905, 0.40236917, 0.41599396,
                0.41069633, 0.30486232, 0.308009, 0.2960596, 0.31806993, 0.30518195, 0.29469204, 0.30109736,
                0.33512104, 0.30267385, 0.3196509, 0.3189264, 0.3150993, 0.31594187, 0.2918061, 0.3405213,
                0.31518283, 0.29538792, 0.3121405, 0.3104312, 0.32041973, 0.29608276, 0.32657534, 0.2912942,
                0.28999257, 0.31479672, 0.31388244, 0.29857287, 0.2933532, 0.30165273, 0.33991206, 0.3229256,
                0.32478786, 0.32574654, 0.2773492, 0.29245508, 0.3079092, 0.30240566, 0.3118857, 0.31225735,
                0.31820515, 0.2972328, 0.29609585, 0.32071212, 0.33419743, 0.30878097, 0.306982, 0.30897063,
                0.31161052, 0.35792035, 0.31483534, 0.29043347, 0.2803108, 0.2765198, 0.26588768, 0.2765353,
                0.25902697, 0.29117516, 0.25392145, 0.26380652, 0.30825472, 0.29911253, 0.2804806, 0.2873455,
                0.28567985, 0.2967215, 0.30260953, 0.27310467, 0.28817332, 0.25019652, 0.28703368, 0.29473418,
                0.28886622, 0.250242, 0.2941401, 0.28642738, 0.26717362, 0.29692984, 0.29185408, 0.2750152,
                0.2653549, 0.26353893, 0.28980076, 0.27838808, 0.27920067, 0.2588307, 0.28505707, 0.3022325,
                0.27320877, 0.29476917, 0.29449323, 0.29720396, 0.31319344, 0.2803108, 0.28671616, 0.30462897,
                0.3049033, 0.29277474, 0.29136384, 0.30316526, 0.2807459}, new int[] {150, 3}, 'f');

        System.out.println("Data:" + input.data().length());
        OldSoftMax softMax = new OldSoftMax(input);
        Nd4j.getExecutioner().exec(softMax);
        assertEquals(assertion, softMax.z());

    }


    @Test
    public void testAddBroadcast() {
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape('f', 2, 3);
        INDArray arrRow = Nd4j.create(new double[] {1, 2, 3});
        INDArray assertion = Nd4j.create(new double[] {2, 3, 5, 6, 8, 9}, new int[] {2, 3}, 'f');
        INDArray add = arr.addRowVector(arrRow);
        assertEquals(assertion, add);

        INDArray colVec = Nd4j.linspace(1, 2, 2).reshape(2, 1);
        INDArray colAssertion = Nd4j.create(new double[] {2, 4, 4, 6, 6, 8}, new int[] {2, 3}, 'f');
        INDArray colTest = arr.addColumnVector(colVec);
        assertEquals(colAssertion, colTest);
    }


    @Test
    public void testStridedExp() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6).reshape(2, 3);
        INDArray slice = arr.slice(0);
        // FIXME: int cast
        float[] expected = new float[(int) slice.length()];
        for (int i = 0; i < slice.length(); i++)
            expected[i] = (float) Math.exp(slice.getDouble(i));
        Exp exp = new Exp(slice);
        opExecutioner.exec(exp);
        assertEquals(getFailureMessage(), Nd4j.create(Nd4j.createBuffer(expected)), slice);
    }

    @Test
    public void testSoftMax() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1, 6, 6);
        OldSoftMax softMax = new OldSoftMax(arr);
        opExecutioner.exec(softMax);
        assertEquals(getFailureMessage(), 1.0, softMax.z().sumNumber().doubleValue(), 1e-1);
    }

    @Test
    public void testIMax() {
        INDArray arr = Nd4j.linspace(1, 10, 10);
        IMax imax = new IMax(arr);
        assertEquals(9, ((IndexAccumulation) Nd4j.getExecutioner().exec(imax)).getFinalResult());

        arr.muli(-1);
        imax = new IMax(arr);
        int maxIdx = ((IndexAccumulation) Nd4j.getExecutioner().exec(imax)).getFinalResult();
        assertEquals(0, maxIdx);
    }

    @Test
    public void testIMin() {
        INDArray arr = Nd4j.linspace(1, 10, 10);
        IMin imin = new IMin(arr);
        assertEquals(0, ((IndexAccumulation) Nd4j.getExecutioner().exec(imin)).getFinalResult());

        arr.muli(-1);
        imin = new IMin(arr);
        int minIdx = ((IndexAccumulation) Nd4j.getExecutioner().exec(imin)).getFinalResult();
        assertEquals(9, minIdx);
    }


    @Test
    public void testMeanSumSimple() {
        System.out.println("3d");
        INDArray arr = Nd4j.ones(1, 4, 4);
        assertEquals(Nd4j.ones(1), arr.mean(1, 2));
        assertEquals(Nd4j.ones(1).muli(16), arr.sum(1, 2));

        System.out.println("4d");
        INDArray arr4 = Nd4j.ones(1, 1, 4, 4);
        INDArray arr4m = arr4.mean(2, 3);
        INDArray arr4s = arr4.sum(2, 3);
        for (int i = 0; i < arr4m.length(); i++)
            assertEquals(arr4m.getDouble(i), 1, 1e-1);
        for (int i = 0; i < arr4s.length(); i++)
            assertEquals(arr4s.getDouble(i), 16, 1e-1);

        System.out.println("5d");
        INDArray arr5 = Nd4j.ones(1, 1, 4, 4, 4);
        INDArray arr5m = arr5.mean(2, 3);
        INDArray arr5s = arr5.sum(2, 3);
        for (int i = 0; i < arr5m.length(); i++)
            assertEquals(arr5m.getDouble(i), 1, 1e-1);
        for (int i = 0; i < arr5s.length(); i++)
            assertEquals(arr5s.getDouble(i), 16, 1e-1);
        System.out.println("6d");
        INDArray arr6 = Nd4j.ones(1, 1, 4, 4, 4, 4);
        INDArray arr6Tad = arr6.javaTensorAlongDimension(0, 2, 3);
        INDArray arr6s = arr6.sum(2, 3);
        for (int i = 0; i < arr6s.length(); i++)
            assertEquals(arr6s.getDouble(i), 16, 1e-1);

        INDArray arr6m = arr6.mean(2, 3);
        for (int i = 0; i < arr6m.length(); i++)
            assertEquals(arr6m.getDouble(i), 1, 1e-1);

    }

    @Test
    public void tescodtSum6d() {
        INDArray arr6 = Nd4j.ones(1, 1, 4, 4, 4, 4);
        INDArray arr6s = arr6.sum(2, 3);

        System.out.println("Arr6s: " + arr6.length());
        for (int i = 0; i < arr6s.length(); i++)
            assertEquals(16, arr6s.getDouble(i), 1e-1);
    }

    @Test
    public void testSum6d2() throws Exception {
        INDArray arr6 = Nd4j.linspace(1, 256, 256).reshape(1, 1, 4, 4, 4, 4);
        INDArray arr6s = arr6.sum(2, 3);

        assertEquals(136, arr6s.getDouble(0), 1e-1);
        assertEquals(1160, arr6s.getDouble(1), 1e-1);
        assertEquals(2184, arr6s.getDouble(2), 1e-1);
        assertEquals(3208, arr6s.getDouble(3), 1e-1);
        assertEquals(392, arr6s.getDouble(4), 1e-1);
        assertEquals(1416, arr6s.getDouble(5), 1e-1);
        assertEquals(2440, arr6s.getDouble(6), 1e-1);
    }



    @Test
    public void testMean6d() {
        INDArray arr6 = Nd4j.ones(1, 1, 4, 4, 4, 4);

        INDArray arr6m = arr6.mean(2, 3);
        for (int i = 0; i < arr6m.length(); i++)
            assertEquals(1.0, arr6m.getDouble(i), 1e-1);
        /*
        System.out.println("Arr6 shapeInfo: " + arr6.shapeInfoDataBuffer());
        System.out.println("Arr6 length: " + arr6.length());
        System.out.println("Arr6 shapeLlength: " + arr6.shapeInfoDataBuffer().length());
        System.out.println("Arr6s shapeInfo: " + arr6s.shapeInfoDataBuffer());
        System.out.println("Arr6s length: " + arr6s.length());
        System.out.println("Arr6s shapeLength: " + arr6s.shapeInfoDataBuffer().length());
         */
    }


    @Test
    public void testStdev() {
        INDArray arr = Nd4j.create(new float[] {0.9296161f, 0.31637555f, 0.1839188f}, new int[] {1, 3}, ordering());
        double stdev = arr.stdNumber(true).doubleValue();


        val standardDeviation = new org.apache.commons.math3.stat.descriptive.moment.StandardDeviation(true);
        double exp = standardDeviation.evaluate(arr.toDoubleVector());
        assertEquals(exp, stdev, 1e-7f);


        double stdev2 = arr.std(true, 1).getDouble(0);
        assertEquals(stdev, stdev2, 1e-3);
    }

    @Test
    public void testVariance() {
        val f = new double[] {0.9296161, 0.31637555, 0.1839188};
        INDArray arr = Nd4j.create(f, new int[] {1, 3}, ordering());
        double var = arr.varNumber().doubleValue();

        INDArray var1 = arr.var(1);
        double var2 = var1.getDouble(0);
        assertEquals(var, var2, 1e-3);

        val variance = new org.apache.commons.math3.stat.descriptive.moment.Variance(true);
        double exp = variance.evaluate(arr.toDoubleVector());
        assertEquals(exp, var, 1e-7f);
    }

    @Test
    public void testDropout() {
        INDArray array = Nd4j.linspace(1, 100, 100);
        INDArray result = Nd4j.create(100);

        DropOut dropOut = new DropOut(array, result, 0.05);
        Nd4j.getExecutioner().exec(dropOut);

        System.out.println("Src array: " + array);
        System.out.println("Res array: " + result);

        assertNotEquals(array, result);
    }

    @Test
    public void testDropoutInverted() {
        INDArray array = Nd4j.linspace(1, 100, 100);
        INDArray result = Nd4j.create(100);

        DropOutInverted dropOut = new DropOutInverted(array, result, 0.65);
        Nd4j.getExecutioner().exec(dropOut);

        System.out.println("Src array: " + array);
        System.out.println("Res array: " + result);

        assertNotEquals(array, result);
    }

    @Test
    public void testVPull1() {
        int indexes[] = new int[] {0, 2, 4};
        INDArray array = Nd4j.linspace(1, 25, 25).reshape(5, 5);
        INDArray assertion = Nd4j.createUninitialized(new int[] {3, 5}, 'f');
        for (int i = 0; i < 3; i++) {
            assertion.putRow(i, array.getRow(indexes[i]));
        }

        INDArray result = Nd4j.pullRows(array, 1, indexes, 'f');

        assertEquals(3, result.rows());
        assertEquals(5, result.columns());
        assertEquals(assertion, result);
    }

    @Test
    public void testVPull2() {
        int indexes[] = new int[] {0, 2, 4};
        INDArray array = Nd4j.linspace(1, 25, 25).reshape(5, 5);
        INDArray assertion = Nd4j.createUninitialized(new int[] {3, 5}, 'c');
        for (int i = 0; i < 3; i++) {
            assertion.putRow(i, array.getRow(indexes[i]));
        }

        INDArray result = Nd4j.pullRows(array, 1, indexes, 'c');

        assertEquals(3, result.rows());
        assertEquals(5, result.columns());
        assertEquals(assertion, result);

        System.out.println(assertion.toString());
        System.out.println(result.toString());
    }


    @Test
    public void testPile1() throws Exception {
        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(10, 10).assign(i));
        }

        INDArray pile = Nd4j.pile(arrays);

        assertEquals(3, pile.rank());
        for (int i = 0; i < 10; i++) {
            assertEquals((float) i, pile.tensorAlongDimension(i, 1, 2).getDouble(0), 0.01);
        }
    }

    @Test
    public void testPile2() throws Exception {
        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(10, 10, 10).assign(i));
        }

        INDArray pile = Nd4j.pile(arrays);

        assertEquals(4, pile.rank());
        for (int i = 0; i < 10; i++) {
            assertEquals((float) i, pile.tensorAlongDimension(i, 1, 2, 3).getDouble(0), 0.01);
        }
    }

    @Test
    public void testPile3() throws Exception {
        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(1, 10, 10).assign(i));
        }

        INDArray pile = Nd4j.pile(arrays);

        assertEquals(3, pile.rank());
        for (int i = 0; i < 10; i++) {
            assertEquals((float) i, pile.tensorAlongDimension(i, 1, 2).getDouble(0), 0.01);
        }
    }

    @Test
    public void testTear1() {
        List<INDArray> arrays = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            arrays.add(Nd4j.create(10, 10).assign(i));
        }

        INDArray pile = Nd4j.pile(arrays);

        INDArray[] tears = Nd4j.tear(pile, 1, 2);

        for (int i = 0; i < 10; i++) {
            assertEquals((float) i, tears[i].meanNumber().floatValue(), 0.01f);
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
