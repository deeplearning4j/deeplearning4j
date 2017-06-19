package org.deeplearning4j.eval;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.GreaterThan;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by Alex on 19/06/2017.
 */
public class EvalCustomThreshold {

    @Test
    public void testEvaluationCustomThreshold(){

        fail();
    }

    @Test
    public void testEvaluationBinaryCustomThreshold(){

        //Sanity check: same results for 0.5 threshold vs.

        int nExamples = 20;
        int nOut = 2;

        INDArray probs = Nd4j.rand(nExamples, nOut);
        INDArray labels = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(nExamples,nOut),0.5));

        EvaluationBinary eStd = new EvaluationBinary();
        eStd.eval(labels, probs);

        EvaluationBinary eb05 = new EvaluationBinary(Nd4j.create(new double[]{0.5, 0.5}));
        eb05.eval(labels, probs);

        EvaluationBinary eb05v2 = new EvaluationBinary(Nd4j.create(new double[]{0.5, 0.5}));
        for( int i=0; i<nExamples; i++ ){
            eb05v2.eval(labels.getRow(i), probs.getRow(i));
        }

        for(EvaluationBinary eb2 : new EvaluationBinary[]{eb05, eb05v2}){
            assertArrayEquals(eStd.getCountTruePositive(), eb2.getCountTruePositive());
            assertArrayEquals(eStd.getCountFalsePositive(), eb2.getCountFalsePositive());
            assertArrayEquals(eStd.getCountTrueNegative(), eb2.getCountTrueNegative());
            assertArrayEquals(eStd.getCountFalseNegative(), eb2.getCountFalseNegative());

            assertEquals(eStd.stats(), eb2.stats());
        }


        //Check with decision threshold of 0.25 and 0.125 (for different outputs)
        //In this test, we'll cheat a bit: multiply probabilities by 2 (max of 1.0) and threshold of 0.25 should give
        // an identical result to a threshold of 0.5
        //Ditto for 4x and 0.125 threshold

        INDArray probs2 = probs.mul(2);
        probs2 = Transforms.min(probs2, 1.0);

        INDArray probs4 = probs.mul(4);
        probs4 = Transforms.min(probs4, 1.0);

        EvaluationBinary ebThreshold = new EvaluationBinary(Nd4j.create(new double[]{0.25, 0.125}));
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
