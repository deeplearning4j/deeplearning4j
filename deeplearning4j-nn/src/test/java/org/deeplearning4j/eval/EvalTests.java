package org.deeplearning4j.eval;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 5/16/17.
 */
public class EvalTests {

    @Test
    public void testSerde() {
        Evaluation evaluation =  new Evaluation();
        String json = evaluation.toJson();
        assertEquals(evaluation,BaseEvaluation.fromJson(json,Evaluation.class));

        ROC roc =  new ROC(2);
        json = roc.toJson();
        assertEquals(roc,BaseEvaluation.fromJson(json,ROC.class));

        ROCBinary roc2 =  new ROCBinary(2);
        json = roc2.toJson();
        assertEquals(roc2,BaseEvaluation.fromJson(json,ROCBinary.class));

        ROCMultiClass roc3 =  new ROCMultiClass(2);
        json = roc3.toJson();
        assertEquals(roc3,BaseEvaluation.fromJson(json,ROCMultiClass.class));


        RegressionEvaluation regressionEvaluation =  new RegressionEvaluation();
        json = regressionEvaluation.toJson();
        assertEquals(regressionEvaluation,BaseEvaluation.fromJson(json,RegressionEvaluation.class));

        EvaluationBinary evaluationBinary = new EvaluationBinary();
        json = evaluationBinary.toJson();
        assertEquals(evaluationBinary,BaseEvaluation.fromJson(json,EvaluationBinary.class));

        evaluationBinary.averageAccuracy();
        evaluationBinary.averageF1();
        evaluationBinary.averagePrecision();
        evaluationBinary.averageRecall();


    }

}
