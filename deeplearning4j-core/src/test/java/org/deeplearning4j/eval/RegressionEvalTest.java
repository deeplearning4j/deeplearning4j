package org.deeplearning4j.eval;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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



}
