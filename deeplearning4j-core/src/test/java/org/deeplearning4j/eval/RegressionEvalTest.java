package org.deeplearning4j.eval;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author Alex Black
 */
public class RegressionEvalTest {

    @Test
    public void testPerfectPredictions(){

        int nCols = 5;
        int nTestArrays = 100;
        int valuesPerTestArray = 3;
        RegressionEvaluation eval = new RegressionEvaluation(nCols);

        for( int i=0; i<nTestArrays; i++ ){
            INDArray rand = Nd4j.rand(valuesPerTestArray,nCols);
            eval.eval(rand,rand);
        }

        System.out.println(eval.stats());

        for( int i=0; i<nCols; i++ ){
            assertEquals(0.0,eval.meanSquaredError(i),1e-6);
            assertEquals(0.0,eval.meanAbsoluteError(i),1e-6);
            assertEquals(0.0,eval.rootMeanSquaredError(i),1e-6);
            assertEquals(0.0,eval.relativeSquaredError(i),1e-6);
            assertEquals(1.0,eval.correlationR2(i),1e-6);
        }
    }

    @Test
    public void testKnownValues(){
        double[][] labelsD = new double[][]{
                {1,2,3},
                {0.1,0.2,0.3},
                {6,5,4}
        };

        double[][] predictedD = new double[][]{
                {2.5,3.2,3.8},
                {2.15,1.3,-1.2},
                {7,4.5,3}
        };

        double[] expMSE = {2.484166667,0.966666667,1.296666667};
        double[] expMAE = {1.516666667,0.933333333,1.1};
        double[] expRSE = {0.368813923,0.246598639,0.530937216};
        double[] expCorrs = {0.997013483, 0.968619605, 0.915603032};

        INDArray labels = Nd4j.create(labelsD);
        INDArray predicted = Nd4j.create(predictedD);

        RegressionEvaluation eval = new RegressionEvaluation(3);

        eval.eval(labels,predicted);

        for( int i=0; i<3; i++ ){
            assertEquals(expMSE[i],eval.meanSquaredError(i),1e-5);
            assertEquals(expMAE[i],eval.meanAbsoluteError(i),1e-5);
            assertEquals(Math.sqrt(expMSE[i]),eval.rootMeanSquaredError(i),1e-5);
            assertEquals(expRSE[i],eval.relativeSquaredError(i),1e-5);
            assertEquals(expCorrs[i],eval.correlationR2(i),1e-5);

        }
    }


    @Test
    public void testRegressionEvaluationMerging(){
        Nd4j.getRandom().setSeed(12345);

        int nRows = 20;
        int nCols = 3;

        int numMinibatches = 5;
        int nEvalInstances = 4;

        List<RegressionEvaluation> list = new ArrayList<>();

        RegressionEvaluation single = new RegressionEvaluation(nCols);

        for( int i=0; i<nEvalInstances; i++ ){
            list.add(new RegressionEvaluation(nCols));
            for( int j=0; j<numMinibatches; j++ ){
                INDArray p = Nd4j.rand(nRows, nCols);
                INDArray act = Nd4j.rand(nRows, nCols);

                single.eval(act, p);

                list.get(i).eval(act,p);
            }
        }

        RegressionEvaluation merged = list.get(0);
        for( int i=1; i<nEvalInstances; i++ ){
            merged.merge(list.get(i));
        }

        double prec = 1e-6;
        for( int i=0; i<nCols; i++ ){
            assertEquals(single.correlationR2(i), merged.correlationR2(i), prec);
            assertEquals(single.meanAbsoluteError(i), merged.meanAbsoluteError(i), prec);
            assertEquals(single.meanSquaredError(i), merged.meanSquaredError(i), prec);
            assertEquals(single.relativeSquaredError(i), merged.relativeSquaredError(i), prec);
            assertEquals(single.rootMeanSquaredError(i), merged.rootMeanSquaredError(i), prec);
        }
    }

}
