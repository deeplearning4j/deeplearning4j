package org.deeplearning4j.eval;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 04/11/2016.
 */
public class ROCTest {

    private static Map<Double,Double> expTPR;
    private static Map<Double,Double> expFPR;

    static {
        expTPR = new HashMap<>();
        double totalPositives = 5.0;
        expTPR.put(0/10.0, 5.0/totalPositives);  //All 10 predicted as class 1, of which 5 of 5 are correct
        expTPR.put(1/10.0, 5.0/totalPositives);
        expTPR.put(2/10.0, 5.0/totalPositives);
        expTPR.put(3/10.0, 5.0/totalPositives);
        expTPR.put(4/10.0, 5.0/totalPositives);
        expTPR.put(5/10.0, 5.0/totalPositives);
        expTPR.put(6/10.0, 4.0/totalPositives);    //Threshold: 0.4 -> last 4 predicted; last 5 actual
        expTPR.put(7/10.0, 3.0/totalPositives);
        expTPR.put(8/10.0, 2.0/totalPositives);
        expTPR.put(9/10.0, 1.0/totalPositives);
        expTPR.put(10/10.0, 0.0/totalPositives);

        expFPR = new HashMap<>();
        double totalNegatives = 5.0;
        expFPR.put(0/10.0, 5.0/totalNegatives);  //All 10 predicted as class 1, but all 5 true negatives are predicted positive
        expFPR.put(1/10.0, 4.0/totalNegatives);  //1 true negative is predicted as negative; 4 false positives
        expFPR.put(2/10.0, 3.0/totalNegatives);  //2 true negatives are predicted as negative; 3 false positives
        expFPR.put(3/10.0, 2.0/totalNegatives);
        expFPR.put(4/10.0, 1.0/totalNegatives);
        expFPR.put(5/10.0, 0.0/totalNegatives);
        expFPR.put(6/10.0, 0.0/totalNegatives);
        expFPR.put(7/10.0, 0.0/totalNegatives);
        expFPR.put(8/10.0, 0.0/totalNegatives);
        expFPR.put(9/10.0, 0.0/totalNegatives);
        expFPR.put(10/10.0, 0.0/totalNegatives);
    }

    @Test
    public void testRocBasic(){
        //2 outputs here - probability distribution over classes (softmax)
        INDArray predictions = Nd4j.create(new double[][]{
                {1.0, 0.001},   //add 0.001 to avoid numerical/rounding issues (float vs. double, etc)
                {0.899, 0.101},
                {0.799, 0.201},
                {0.699, 0.301},
                {0.599, 0.401},
                {0.499, 0.501},
                {0.399, 0.601},
                {0.299, 0.701},
                {0.199, 0.801},
                {0.099, 0.901}});

        INDArray actual = Nd4j.create(new double[][]{
                {1, 0},
                {1, 0},
                {1, 0},
                {1, 0},
                {1, 0},
                {0, 1},
                {0, 1},
                {0, 1},
                {0, 1},
                {0, 1}});

        ROC roc = new ROC(10);
        roc.eval(actual, predictions);

        List<ROC.ROCValue> list = roc.getResults();

        assertEquals(11,list.size());   //0 + 10 steps
        for( int i=0; i<11; i++ ){
            ROC.ROCValue v = list.get(i);
            double expThreshold = i / 10.0;
            assertEquals(expThreshold, v.getThreshold(), 1e-5);

//            System.out.println("t=" + expThreshold + "\t" + v.getFalsePositiveRate() + "\t" + v.getTruePositiveRate());

            double efpr = expFPR.get(expThreshold);
            double afpr = v.getFalsePositiveRate();
            assertEquals(efpr, afpr, 1e-5);

            double etpr = expTPR.get(expThreshold);
            double atpr = v.getTruePositiveRate();
            assertEquals(etpr, atpr, 1e-5);
        }
    }

    @Test
    public void testRocBasicSingleClass(){
        //1 output here - single probability value (sigmoid)

        //add 0.001 to avoid numerical/rounding issues (float vs. double, etc)
        INDArray predictions = Nd4j.create(new double[]{0.001, 0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901}, new int[]{10,1});

        INDArray actual = Nd4j.create(new double[]{ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1}, new int[]{10,1});

        ROC roc = new ROC(10);
        roc.eval(actual, predictions);

        List<ROC.ROCValue> list = roc.getResults();

        assertEquals(11,list.size());   //0 + 10 steps
        for( int i=0; i<11; i++ ){
            ROC.ROCValue v = list.get(i);
            double expThreshold = i / 10.0;
            assertEquals(expThreshold, v.getThreshold(), 1e-5);

//            System.out.println("t=" + expThreshold + "\t" + v.getFalsePositiveRate() + "\t" + v.getTruePositiveRate());

            double efpr = expFPR.get(expThreshold);
            double afpr = v.getFalsePositiveRate();
            assertEquals(efpr, afpr, 1e-5);

            double etpr = expTPR.get(expThreshold);
            double atpr = v.getTruePositiveRate();
            assertEquals(etpr, atpr, 1e-5);
        }
    }


    @Test
    public void testRoc(){
        //Previous tests allowed for a perfect classifier with right threshold...

        INDArray labels = Nd4j.create(new double[][]{
                {0,1},
                {0,1},
                {1,0},
                {1,0},
                {1,0}});

        INDArray prediction = Nd4j.create(new double[][]{
                {0.199, 0.801},
                {0.499, 0.501},
                {0.399, 0.601},
                {0.799, 0.201},
                {0.899, 0.101}});

        Map<Double,Double> expTPR = new HashMap<>();
        double totalPositives = 2.0;
        expTPR.put(0.0, 2.0/totalPositives);    //All predicted class 1 -> 2 true positives / 2 total positives
        expTPR.put(0.1, 2.0/totalPositives);
        expTPR.put(0.2, 2.0/totalPositives);
        expTPR.put(0.3, 2.0/totalPositives);
        expTPR.put(0.4, 2.0/totalPositives);
        expTPR.put(0.5, 2.0/totalPositives);
        expTPR.put(0.6, 1.0/totalPositives);    //At threshold of 0.6, only 1 of 2 positives are predicted positive
        expTPR.put(0.7, 1.0/totalPositives);
        expTPR.put(0.8, 1.0/totalPositives);
        expTPR.put(0.9, 0.0/totalPositives);    //At threshold of 0.9, 0 of 2 positives are predicted positive
        expTPR.put(1.0, 0.0/totalPositives);

        Map<Double,Double> expFPR = new HashMap<>();
        double totalNegatives = 3.0;
        expFPR.put(0.0, 3.0/totalNegatives);    //All predicted class 1 -> 3 false positives / 3 total negatives
        expFPR.put(0.1, 3.0/totalNegatives);
        expFPR.put(0.2, 2.0/totalNegatives);    //At threshold of 0.2: 1 true negative, 2 false positives
        expFPR.put(0.3, 1.0/totalNegatives);    //At threshold of 0.3: 2 true negative, 1 false positive
        expFPR.put(0.4, 1.0/totalNegatives);
        expFPR.put(0.5, 1.0/totalNegatives);
        expFPR.put(0.6, 1.0/totalNegatives);
        expFPR.put(0.7, 0.0/totalNegatives);    //At threshold of 0.7: 3 true negatives, 0 false positives
        expFPR.put(0.8, 0.0/totalNegatives);
        expFPR.put(0.9, 0.0/totalNegatives);
        expFPR.put(1.0, 0.0/totalNegatives);


        ROC roc = new ROC(10);
        roc.eval(labels, prediction);

        List<ROC.ROCValue> list = roc.getResults();

        assertEquals(11,list.size());   //0 + 10 steps
        for( int i=0; i<11; i++ ){
            ROC.ROCValue v = list.get(i);
            double expThreshold = i / 10.0;
            assertEquals(expThreshold, v.getThreshold(), 1e-5);

            double efpr = expFPR.get(expThreshold);
            double afpr = v.getFalsePositiveRate();
            assertEquals(efpr, afpr, 1e-5);

            double etpr = expTPR.get(expThreshold);
            double atpr = v.getTruePositiveRate();
            assertEquals(etpr, atpr, 1e-5);

            System.out.println(v.getFalsePositiveRate() + "\t" + v.getTruePositiveRate());
        }
    }
}
