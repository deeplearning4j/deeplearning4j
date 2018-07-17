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

package org.deeplearning4j.eval;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMin;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 19/06/2017.
 */
public class EvalCustomThreshold extends BaseDL4JTest {

    @Test
    public void testEvaluationCustomBinaryThreshold() {
        Nd4j.getRandom().setSeed(12345);

        //Sanity checks: 0.5 threshold for 1-output and 2-output binary cases
        Evaluation e = new Evaluation();
        Evaluation e05 = new Evaluation(0.5);
        Evaluation e05v2 = new Evaluation(0.5);

        int nExamples = 20;
        int nOut = 2;
        INDArray probs = Nd4j.rand(nExamples, nOut);
        probs.diviColumnVector(probs.sum(1));
        INDArray labels = Nd4j.create(nExamples, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < nExamples; i++) {
            labels.putScalar(i, r.nextInt(2), 1.0);
        }

        e.eval(labels, probs);
        e05.eval(labels, probs);
        e05v2.eval(labels.getColumn(1), probs.getColumn(1)); //"single output binary" case

        for (Evaluation e2 : new Evaluation[] {e05, e05v2}) {
            assertEquals(e.accuracy(), e2.accuracy(), 1e-6);
            assertEquals(e.f1(), e2.f1(), 1e-6);
            assertEquals(e.precision(), e2.precision(), 1e-6);
            assertEquals(e.recall(), e2.recall(), 1e-6);
            assertEquals(e.confusion, e2.confusion);
        }

        //Check with decision threshold of 0.25
        //In this test, we'll cheat a bit: multiply class 1 probabilities by 2 (max of 1.0); this should give an
        // identical result to a threshold of 0.5 vs. no multiplication and threshold of 0.25

        INDArray p2 = probs.dup();
        INDArray p2c = p2.getColumn(1);
        p2c.muli(2.0);
        Nd4j.getExecutioner().exec(new ScalarMin(p2c, null, p2c, p2c.length(), 1.0));
        p2.getColumn(0).assign(p2.getColumn(1).rsub(1.0));

        Evaluation e025 = new Evaluation(0.25);
        e025.eval(labels, probs);

        Evaluation ex2 = new Evaluation();
        ex2.eval(labels, p2);

        assertEquals(ex2.accuracy(), e025.accuracy(), 1e-6);
        assertEquals(ex2.f1(), e025.f1(), 1e-6);
        assertEquals(ex2.precision(), e025.precision(), 1e-6);
        assertEquals(ex2.recall(), e025.recall(), 1e-6);
        assertEquals(ex2.confusion, e025.confusion);


        //Check the same thing, but the single binary output case:

        Evaluation e025v2 = new Evaluation(0.25);
        e025v2.eval(labels.getColumn(1), probs.getColumn(1));

        assertEquals(ex2.accuracy(), e025v2.accuracy(), 1e-6);
        assertEquals(ex2.f1(), e025v2.f1(), 1e-6);
        assertEquals(ex2.precision(), e025v2.precision(), 1e-6);
        assertEquals(ex2.recall(), e025v2.recall(), 1e-6);
        assertEquals(ex2.confusion, e025v2.confusion);
    }

    @Test
    public void testEvaluationCostArray() {


        int nExamples = 20;
        int nOut = 3;
        Nd4j.getRandom().setSeed(12345);
        INDArray probs = Nd4j.rand(nExamples, nOut);
        probs.diviColumnVector(probs.sum(1));
        INDArray labels = Nd4j.create(nExamples, nOut);
        Random r = new Random(12345);
        for (int j = 0; j < nExamples; j++) {
            labels.putScalar(j, r.nextInt(2), 1.0);
        }

        Evaluation e = new Evaluation();
        e.eval(labels, probs);

        //Sanity check: "all equal" cost array - equal to no cost array
        for (int i = 1; i <= 3; i++) {
            Evaluation e2 = new Evaluation(Nd4j.valueArrayOf(new int[] {1, nOut}, i));
            e2.eval(labels, probs);

            assertEquals(e.accuracy(), e2.accuracy(), 1e-6);
            assertEquals(e.f1(), e2.f1(), 1e-6);
            assertEquals(e.precision(), e2.precision(), 1e-6);
            assertEquals(e.recall(), e2.recall(), 1e-6);
            assertEquals(e.confusion, e2.confusion);
        }

        //Manual checks:
        INDArray costArray = Nd4j.create(new double[] {5, 2, 1});
        labels = Nd4j.create(new double[][] {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        probs = Nd4j.create(new double[][] {{0.2, 0.3, 0.5}, //1.0, 0.6, 0.5
                        {0.1, 0.4, 0.5}, //0.5, 0.8, 0.5
                        {0.1, 0.1, 0.8}}); //0.5, 0.2, 0.8

        //With no cost array: only last example is predicted correctly
        e = new Evaluation();
        e.eval(labels, probs);
        assertEquals(1.0 / 3, e.accuracy(), 1e-6);

        //With cost array: all examples predicted correctly
        Evaluation e2 = new Evaluation(costArray);
        e2.eval(labels, probs);
        assertEquals(1.0, e2.accuracy(), 1e-6);
    }

    @Test
    public void testEvaluationBinaryCustomThreshold() {

        //Sanity check: same results for 0.5 threshold vs. default (no threshold)
        int nExamples = 20;
        int nOut = 2;
        INDArray probs = Nd4j.rand(nExamples, nOut);
        INDArray labels = Nd4j.getExecutioner()
                        .exec(new BernoulliDistribution(Nd4j.createUninitialized(nExamples, nOut), 0.5));

        EvaluationBinary eStd = new EvaluationBinary();
        eStd.eval(labels, probs);

        EvaluationBinary eb05 = new EvaluationBinary(Nd4j.create(new double[] {0.5, 0.5}));
        eb05.eval(labels, probs);

        EvaluationBinary eb05v2 = new EvaluationBinary(Nd4j.create(new double[] {0.5, 0.5}));
        for (int i = 0; i < nExamples; i++) {
            eb05v2.eval(labels.getRow(i), probs.getRow(i));
        }

        for (EvaluationBinary eb2 : new EvaluationBinary[] {eb05, eb05v2}) {
            assertArrayEquals(eStd.getCountTruePositive(), eb2.getCountTruePositive());
            assertArrayEquals(eStd.getCountFalsePositive(), eb2.getCountFalsePositive());
            assertArrayEquals(eStd.getCountTrueNegative(), eb2.getCountTrueNegative());
            assertArrayEquals(eStd.getCountFalseNegative(), eb2.getCountFalseNegative());

            for (int j = 0; j < nOut; j++) {
                assertEquals(eStd.accuracy(j), eb2.accuracy(j), 1e-6);
                assertEquals(eStd.f1(j), eb2.f1(j), 1e-6);
            }
        }


        //Check with decision threshold of 0.25 and 0.125 (for different outputs)
        //In this test, we'll cheat a bit: multiply probabilities by 2 (max of 1.0) and threshold of 0.25 should give
        // an identical result to a threshold of 0.5
        //Ditto for 4x and 0.125 threshold

        INDArray probs2 = probs.mul(2);
        probs2 = Transforms.min(probs2, 1.0);

        INDArray probs4 = probs.mul(4);
        probs4 = Transforms.min(probs4, 1.0);

        EvaluationBinary ebThreshold = new EvaluationBinary(Nd4j.create(new double[] {0.25, 0.125}));
        ebThreshold.eval(labels, probs);

        EvaluationBinary ebStd2 = new EvaluationBinary();
        ebStd2.eval(labels, probs2);

        EvaluationBinary ebStd4 = new EvaluationBinary();
        ebStd4.eval(labels, probs4);

        assertEquals(ebThreshold.truePositives(0), ebStd2.truePositives(0));
        assertEquals(ebThreshold.trueNegatives(0), ebStd2.trueNegatives(0));
        assertEquals(ebThreshold.falsePositives(0), ebStd2.falsePositives(0));
        assertEquals(ebThreshold.falseNegatives(0), ebStd2.falseNegatives(0));

        assertEquals(ebThreshold.truePositives(1), ebStd4.truePositives(1));
        assertEquals(ebThreshold.trueNegatives(1), ebStd4.trueNegatives(1));
        assertEquals(ebThreshold.falsePositives(1), ebStd4.falsePositives(1));
        assertEquals(ebThreshold.falseNegatives(1), ebStd4.falseNegatives(1));
    }

}
