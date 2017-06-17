package org.deeplearning4j.eval;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by agibsonccc on 5/16/17.
 */
public class EvalJsonTest {

    @Test
    public void testSerdeEmpty() {
        boolean print = false;

        IEvaluation[] arr = new IEvaluation[]{
                new Evaluation(),
                new EvaluationBinary(),
                new ROCBinary(10),
                new ROCMultiClass(10),
                new RegressionEvaluation(3),
                new RegressionEvaluation()
        };

        for (IEvaluation e : arr) {
            String json = e.toJson();
            String stats = e.stats();
            if (print) {
                System.out.println(e.getClass() + "\n" + json + "\n\n");
            }
            IEvaluation fromJson = BaseEvaluation.fromJson(json, BaseEvaluation.class);
            assertEquals(e, fromJson);
            assertEquals(stats, fromJson.stats());
        }
    }

    @Test
    public void testSerde(){
        boolean print = true;

        Nd4j.getRandom().setSeed(12345);
        Evaluation evaluation =  new Evaluation();
        EvaluationBinary evaluationBinary = new EvaluationBinary();
        ROC roc =  new ROC(2);
        ROCBinary roc2 =  new ROCBinary(2);
        ROCMultiClass roc3 =  new ROCMultiClass(2);
        RegressionEvaluation regressionEvaluation =  new RegressionEvaluation();


        IEvaluation[] arr = new IEvaluation[]{
                evaluation,
                evaluationBinary,
                roc,
                roc2,
                roc3,
                regressionEvaluation
        };

        INDArray evalLabel = Nd4j.create(10,3);
        for( int i=0; i<10; i++ ){
            evalLabel.putScalar(i, i%3, 1.0);
        }
        INDArray evalProb = Nd4j.rand(10, 3);
        evalProb.diviColumnVector(evalProb.sum(1));
        evaluation.eval(evalLabel, evalProb);
        roc3.eval(evalLabel, evalProb);

        evalLabel = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(10,3), 0.5));
        evalProb = Nd4j.rand(10, 3);
        evaluationBinary.eval(evalLabel, evalProb);
        roc2.eval(evalLabel, evalProb);

        evalLabel = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(10,1), 0.5));
        evalProb = Nd4j.rand(10, 1);
        roc.eval(evalLabel, evalProb);

        regressionEvaluation.eval(Nd4j.rand(10,3), Nd4j.rand(10,3));


        for (IEvaluation e : arr) {
            String json = e.toJson();
            String stats = e.stats();
            if (print) {
                System.out.println(e.getClass() + "\n" + json + "\n\n");
            }
            IEvaluation fromJson = BaseEvaluation.fromJson(json, BaseEvaluation.class);
            assertEquals(e, fromJson);
            assertEquals(stats, fromJson.stats());
        }
    }

    public void testSerdeExactRoc(){


    }

}
