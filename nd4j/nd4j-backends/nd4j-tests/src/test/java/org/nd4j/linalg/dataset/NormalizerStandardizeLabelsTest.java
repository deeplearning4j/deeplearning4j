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

package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by susaneraly on 7/30/16.
 */
@RunWith(Parameterized.class)
public class NormalizerStandardizeLabelsTest extends BaseNd4jTest {
    public NormalizerStandardizeLabelsTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testBruteForce() {
        /* This test creates a dataset where feature values are multiples of consecutive natural numbers
           The obtained values are compared to the theoretical mean and std dev
         */
        double tolerancePerc = 0.01;
        int nSamples = 5120;
        int x = 1, y = 2, z = 3;

        INDArray featureX = Nd4j.linspace(1, nSamples, nSamples).reshape(nSamples, 1).mul(x);
        INDArray featureY = featureX.mul(y);
        INDArray featureZ = featureX.mul(z);
        INDArray featureSet = Nd4j.concat(1, featureX, featureY, featureZ);
        INDArray labelSet = featureSet.dup().getColumns(new int[] {0});
        DataSet sampleDataSet = new DataSet(featureSet, labelSet);

        double meanNaturalNums = (nSamples + 1) / 2.0;
        INDArray theoreticalMean =
                        Nd4j.create(new double[] {meanNaturalNums * x, meanNaturalNums * y, meanNaturalNums * z});
        INDArray theoreticallabelMean = theoreticalMean.dup().getColumns(new int[] {0});
        double stdNaturalNums = Math.sqrt((nSamples * nSamples - 1) / 12.0);
        INDArray theoreticalStd =
                        Nd4j.create(new double[] {stdNaturalNums * x, stdNaturalNums * y, stdNaturalNums * z});
        INDArray theoreticallabelStd = theoreticalStd.dup().getColumns(new int[] {0});

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        myNormalizer.fitLabel(true);
        myNormalizer.fit(sampleDataSet);

        INDArray meanDelta = Transforms.abs(theoreticalMean.sub(myNormalizer.getMean()));
        INDArray labelDelta = Transforms.abs(theoreticallabelMean.sub(myNormalizer.getLabelMean()));
        INDArray meanDeltaPerc = meanDelta.div(theoreticalMean).mul(100);
        INDArray labelDeltaPerc = labelDelta.div(theoreticallabelMean).mul(100);
        double maxMeanDeltaPerc = meanDeltaPerc.max(1).getDouble(0, 0);
        assertTrue(maxMeanDeltaPerc < tolerancePerc);
        assertTrue(labelDeltaPerc.max(1).getDouble(0, 0) < tolerancePerc);

        INDArray stdDelta = Transforms.abs(theoreticalStd.sub(myNormalizer.getStd()));
        INDArray stdDeltaPerc = stdDelta.div(theoreticalStd).mul(100);
        INDArray stdlabelDeltaPerc =
                        Transforms.abs(theoreticallabelStd.sub(myNormalizer.getLabelStd())).div(theoreticallabelStd);
        double maxStdDeltaPerc = stdDeltaPerc.max(1).mul(100).getDouble(0, 0);
        double maxlabelStdDeltaPerc = stdlabelDeltaPerc.max(1).getDouble(0, 0);
        assertTrue(maxStdDeltaPerc < tolerancePerc);
        assertTrue(maxlabelStdDeltaPerc < tolerancePerc);


        // SAME TEST WITH THE ITERATOR
        int bSize = 10;
        tolerancePerc = 0.1; // 1% of correct value
        DataSetIterator sampleIter = new TestDataSetIterator(sampleDataSet, bSize);
        myNormalizer.fit(sampleIter);

        meanDelta = Transforms.abs(theoreticalMean.sub(myNormalizer.getMean()));
        meanDeltaPerc = meanDelta.div(theoreticalMean).mul(100);
        maxMeanDeltaPerc = meanDeltaPerc.max(1).getDouble(0, 0);
        assertTrue(maxMeanDeltaPerc < tolerancePerc);

        stdDelta = Transforms.abs(theoreticalMean.sub(myNormalizer.getMean()));
        stdDeltaPerc = stdDelta.div(theoreticalStd).mul(100);
        maxStdDeltaPerc = stdDeltaPerc.max(1).getDouble(0, 0);
        assertTrue(maxStdDeltaPerc < tolerancePerc);
    }

    @Test
    public void testTransform() {
        /*Random dataset is generated such that
            AX + B where X is from a normal distribution with mean 0 and std 1
            The mean of above will be B and std A
            Obtained mean and std dev are compared to theoretical
            Transformed values should be the same as X with the same seed.
         */
        long randSeed = 2227724;

        int nFeatures = 2;
        int nSamples = 6400;
        int bsize = 8;
        int a = 5;
        int b = 100;
        INDArray sampleMean, sampleStd, sampleMeanDelta, sampleStdDelta, delta, deltaPerc;
        double maxDeltaPerc, sampleMeanSEM;

        genRandomDataSet normData = new genRandomDataSet(nSamples, nFeatures, a, b, randSeed);
        genRandomDataSet expectedData = new genRandomDataSet(nSamples, nFeatures, 1, 0, randSeed);
        genRandomDataSet beforeTransformData = new genRandomDataSet(nSamples, nFeatures, a, b, randSeed);

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        myNormalizer.fitLabel(true);
        DataSetIterator normIterator = normData.getIter(bsize);
        DataSetIterator expectedIterator = expectedData.getIter(bsize);
        DataSetIterator beforeTransformIterator = beforeTransformData.getIter(bsize);

        myNormalizer.fit(normIterator);

        double tolerancePerc = 0.5; //within 0.5%
        sampleMean = myNormalizer.getMean();
        sampleMeanDelta = Transforms.abs(sampleMean.sub(normData.theoreticalMean));
        assertTrue(sampleMeanDelta.mul(100).div(normData.theoreticalMean).max(1).getDouble(0, 0) < tolerancePerc);
        //sanity check to see if it's within the theoretical standard error of mean
        sampleMeanSEM = sampleMeanDelta.div(normData.theoreticalSEM).max(1).getDouble(0, 0);
        assertTrue(sampleMeanSEM < 2.6); //99% of the time it should be within this many SEMs

        tolerancePerc = 5; //within 5%
        sampleStd = myNormalizer.getStd();
        sampleStdDelta = Transforms.abs(sampleStd.sub(normData.theoreticalStd));
        assertTrue(sampleStdDelta.div(normData.theoreticalStd).max(1).mul(100).getDouble(0, 0) < tolerancePerc);

        tolerancePerc = 1; //within 1%
        normIterator.setPreProcessor(myNormalizer);
        while (normIterator.hasNext()) {
            INDArray before = beforeTransformIterator.next().getFeatures();
            DataSet here = normIterator.next();
            assertEquals(here.getFeatures(), here.getLabels()); //bootstrapping existing test on features
            INDArray after = here.getFeatures();
            INDArray expected = expectedIterator.next().getFeatures();
            delta = Transforms.abs(after.sub(expected));
            deltaPerc = delta.div(before.sub(expected));
            deltaPerc.muli(100);
            maxDeltaPerc = deltaPerc.max(0, 1).getDouble(0, 0);
            //System.out.println("=== BEFORE ===");
            //System.out.println(before);
            //System.out.println("=== AFTER ===");
            //System.out.println(after);
            //System.out.println("=== SHOULD BE ===");
            //System.out.println(expected);
            assertTrue(maxDeltaPerc < tolerancePerc);
        }
    }


    public class genRandomDataSet {
        /* generate random dataset from normally distributed mean 0, std 1
        based on given seed and scaling constants
         */
        DataSet sampleDataSet;
        INDArray theoreticalMean;
        INDArray theoreticalStd;
        INDArray theoreticalSEM;

        public genRandomDataSet(int nSamples, int nFeatures, int a, int b, long randSeed) {
            /* if a =1 and b = 0,normal distribution
                otherwise with some random mean and some random distribution
             */
            int i = 0;
            // Randomly generate scaling constants and add offsets
            // to get aA and bB
            INDArray aA = a == 1 ? Nd4j.ones(1, nFeatures) : Nd4j.rand(1, nFeatures, randSeed).mul(a); //a = 1, don't scale
            INDArray bB = Nd4j.rand(1, nFeatures, randSeed).mul(b); //b = 0 this zeros out
            // transform ndarray as X = aA + bB * X
            INDArray randomFeatures = Nd4j.zeros(nSamples, nFeatures);
            while (i < nFeatures) {
                INDArray randomSlice = Nd4j.randn(nSamples, 1, randSeed);
                randomSlice.muli(aA.getScalar(0, i));
                randomSlice.addi(bB.getScalar(0, i));
                randomFeatures.putColumn(i, randomSlice);
                i++;
            }
            INDArray randomLabels = randomFeatures.dup();
            this.sampleDataSet = new DataSet(randomFeatures, randomLabels);
            this.theoreticalMean = bB.dup();
            this.theoreticalStd = aA.dup();
            this.theoreticalSEM = this.theoreticalStd.div(Math.sqrt(nSamples));
        }

        public DataSetIterator getIter(int bsize) {
            return new TestDataSetIterator(sampleDataSet, bsize);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}

