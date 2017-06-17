package org.deeplearning4j.eval;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by agibsonccc on 5/16/17.
 */
public class EvalJsonTest {

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


        RegressionEvaluation regressionEvaluation =  new RegressionEvaluation(1);
        regressionEvaluation.eval(Nd4j.create(10,1), Nd4j.ones(10,1));
        json = regressionEvaluation.toJson();
        System.out.println(json);
        assertEquals(regressionEvaluation,BaseEvaluation.fromJson(json,RegressionEvaluation.class));

        EvaluationBinary evaluationBinary = new EvaluationBinary();
        json = evaluationBinary.toJson();
        assertEquals(evaluationBinary,BaseEvaluation.fromJson(json,EvaluationBinary.class));

        evaluationBinary.averageAccuracy();
        evaluationBinary.averageF1();
        evaluationBinary.averagePrecision();
        evaluationBinary.averageRecall();
        roc2.calculateAverageAuc();
        regressionEvaluation.averagecorrelationR2();
        regressionEvaluation.averageMeanAbsoluteError();
        regressionEvaluation.averagerelativeSquaredError();
        regressionEvaluation.averagerootMeanSquaredError();


        fail("Need to test exact ROC implemnetations + write better tests before merging");
    }

}
