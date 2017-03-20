package org.deeplearning4j.eval;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarLessThan;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by Alex on 20/03/2017.
 */
public class EvaluationBinaryTest {

    @Test
    public void testEvaluationBinary(){
        //Compare EvaluationBinary to Evaluation class

        Nd4j.getRandom().setSeed(12345);

        int nExamples = 50;
        int nOut = 4;
        int[] shape = {nExamples,nOut};

        INDArray labels = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape), 0.5));

        INDArray predicted = Nd4j.rand(shape);
        INDArray binaryPredicted = predicted.gt(0.5);

        EvaluationBinary eb = new EvaluationBinary();
        eb.eval(labels, predicted);

        System.out.println(eb.stats());

        double eps = 1e-6;
        for( int i=0; i<nOut; i++ ){
            INDArray lCol = labels.getColumn(i);
            INDArray pCol = predicted.getColumn(i);
            INDArray bpCol = binaryPredicted.getColumn(i);

            int countCorrect = 0;
            int tpCount = 0;
            int tnCount = 0;
            for( int j=0; j<lCol.length(); j++ ){
                if(lCol.getDouble(j) == bpCol.getDouble(j)){
                    countCorrect++;
                    if(lCol.getDouble(j) == 1){
                        tpCount++;
                    } else {
                        tnCount++;
                    }
                }
            }
            double acc = countCorrect / (double)lCol.length();

            Evaluation e = new Evaluation();
            e.eval(lCol, pCol);

            assertEquals(acc, eb.accuracy(i), eps);
            assertEquals(e.accuracy(), eb.accuracy(i), eps);
            assertEquals(e.precision(1), eb.precision(i), eps);
            assertEquals(e.recall(1), eb.recall(i), eps);
            assertEquals(e.f1(1), eb.f1(i), eps);

            assertEquals(tpCount, eb.truePositives(i));
            assertEquals(tnCount, eb.trueNegatives(i));

            assertEquals((int)e.truePositives().get(1), eb.truePositives(i));
            assertEquals((int)e.trueNegatives().get(1), eb.trueNegatives(i));
            assertEquals((int)e.falsePositives().get(1), eb.falsePositives(i));
            assertEquals((int)e.falseNegatives().get(1), eb.falseNegatives(i));

            assertEquals(nExamples, eb.totalCount(i));
        }
    }


    @Test
    public void testEvaluationBinaryPerOutputMasking(){

        fail();
    }


    @Test
    public void testEBMerging(){

        fail();
    }
}
