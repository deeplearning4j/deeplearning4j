package org.deeplearning4j.eval;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.FeatureUtil;

/**
 * Created by agibsonccc on 12/22/14.
 */
public class EvalTest {

    @Test
    public void testEval() {
        Evaluation eval = new Evaluation();
        INDArray trueOutcome = FeatureUtil.toOutcomeVector(0,5);
        INDArray test = FeatureUtil.toOutcomeVector(0,5);
        eval.eval(trueOutcome,test);
        assertEquals(1.0,eval.f1(),1e-1);
        assertEquals(1,eval.classCount(0));

        INDArray one = FeatureUtil.toOutcomeVector(1,5);
        INDArray miss = FeatureUtil.toOutcomeVector(0,5);
        eval.eval(one,miss);
        assertTrue(eval.f1() < 1);
        assertEquals(1,eval.classCount(0));
        assertEquals(1,eval.classCount(1));

        assertEquals(2.0,eval.positive(),1e-1);
        assertEquals(1.0,eval.negative(),1e-1);



    }


}
