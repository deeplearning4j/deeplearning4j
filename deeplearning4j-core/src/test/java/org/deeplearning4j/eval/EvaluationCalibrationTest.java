package org.deeplearning4j.eval;

import org.deeplearning4j.eval.curves.ReliabilityDiagram;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by Alex on 05/07/2017.
 */
public class EvaluationCalibrationTest {

    @Test
    public void testReliabilityDiagram() {


        //Test using 5 bins - format: binary softmax-style output
        //Note: no values fall in fourth bin

        //[0, 0.2)
        INDArray bin0Probs = Nd4j.create(new double[][]{{1.0, 0.0}, {0.9, 0.1}, {0.85, 0.15}});
        INDArray bin0Labels = Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});

        //[0.2, 0.4)
        INDArray bin1Probs = Nd4j.create(new double[][]{{0.8, 0.2}, {0.7, 0.3}, {0.65, 0.35}});
        INDArray bin1Labels = Nd4j.create(new double[][]{{1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}});

        //[0.4, 0.6)
        INDArray bin2Probs = Nd4j.create(new double[][]{{0.59, 0.41}, {0.5, 0.5}, {0.45, 0.55}});
        INDArray bin2Labels = Nd4j.create(new double[][]{{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}});

        //[0.6, 0.8)
        //Empty

        //[0.8, 1.0]
        INDArray bin4Probs = Nd4j.create(new double[][]{{0.0, 1.0}, {0.1, 0.9}});
        INDArray bin4Labels = Nd4j.create(new double[][]{{0.0, 1.0}, {0.0, 1.0}});


        INDArray probs = Nd4j.vstack(bin0Probs, bin1Probs, bin2Probs, bin4Probs);
        INDArray labels = Nd4j.vstack(bin0Labels, bin1Labels, bin2Labels, bin4Labels);

        EvaluationCalibration ec = new EvaluationCalibration(5);
        ec.eval(labels, probs);

        for (int i = 0; i < 1; i++) {
            double[] avgBinProbsClass;
            double[] fracPos;
            if (i == 0) {
                //Class 0: needs to be handled a little differently, due to threshold/edge cases (0.8, etc)
                avgBinProbsClass =  new double[]{0.05, (0.59+0.5+0.45)/3, (0.65+0.7)/2.0, (0.8+0.85+0.9+1.0)/4 };
                fracPos =           new double[]{0.0/2.0, 1.0/3, 1.0/2, 3.0/4 };
            } else {
                avgBinProbsClass = new double[]{
                        bin0Probs.getColumn(i).meanNumber().doubleValue(),
                        bin1Probs.getColumn(i).meanNumber().doubleValue(),
                        bin2Probs.getColumn(i).meanNumber().doubleValue(),
                        bin4Probs.getColumn(i).meanNumber().doubleValue()};

                fracPos = new double[]{
                        bin0Labels.getColumn(i).sumNumber().doubleValue() / bin0Labels.size(0),
                        bin1Labels.getColumn(i).sumNumber().doubleValue() / bin1Labels.size(0),
                        bin2Labels.getColumn(i).sumNumber().doubleValue() / bin2Labels.size(0),
                        bin4Labels.getColumn(i).sumNumber().doubleValue() / bin4Labels.size(0)};
            }

            ReliabilityDiagram rd = ec.getReliabilityDiagram(i);

            double[] x = rd.getMeanPredictedValueX();
            double[] y = rd.getFractionPositivesY();

            assertArrayEquals(avgBinProbsClass, x, 1e-6);
            assertArrayEquals(fracPos, y, 1e-6);
        }
    }

}
