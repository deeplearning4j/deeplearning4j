package org.deeplearning4j.eval;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.eval.curves.Histogram;
import org.deeplearning4j.eval.curves.ReliabilityDiagram;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

/**
 * Created by Alex on 05/07/2017.
 */
public class EvaluationCalibrationTest extends BaseDL4JTest {

    @Test
    public void testReliabilityDiagram() {


        //Test using 5 bins - format: binary softmax-style output
        //Note: no values fall in fourth bin

        //[0, 0.2)
        INDArray bin0Probs = Nd4j.create(new double[][] {{1.0, 0.0}, {0.9, 0.1}, {0.85, 0.15}});
        INDArray bin0Labels = Nd4j.create(new double[][] {{1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});

        //[0.2, 0.4)
        INDArray bin1Probs = Nd4j.create(new double[][] {{0.8, 0.2}, {0.7, 0.3}, {0.65, 0.35}});
        INDArray bin1Labels = Nd4j.create(new double[][] {{1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}});

        //[0.4, 0.6)
        INDArray bin2Probs = Nd4j.create(new double[][] {{0.59, 0.41}, {0.5, 0.5}, {0.45, 0.55}});
        INDArray bin2Labels = Nd4j.create(new double[][] {{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}});

        //[0.6, 0.8)
        //Empty

        //[0.8, 1.0]
        INDArray bin4Probs = Nd4j.create(new double[][] {{0.0, 1.0}, {0.1, 0.9}});
        INDArray bin4Labels = Nd4j.create(new double[][] {{0.0, 1.0}, {0.0, 1.0}});


        INDArray probs = Nd4j.vstack(bin0Probs, bin1Probs, bin2Probs, bin4Probs);
        INDArray labels = Nd4j.vstack(bin0Labels, bin1Labels, bin2Labels, bin4Labels);

        EvaluationCalibration ec = new EvaluationCalibration(5, 5);
        ec.eval(labels, probs);

        for (int i = 0; i < 1; i++) {
            double[] avgBinProbsClass;
            double[] fracPos;
            if (i == 0) {
                //Class 0: needs to be handled a little differently, due to threshold/edge cases (0.8, etc)
                avgBinProbsClass = new double[] {0.05, (0.59 + 0.5 + 0.45) / 3, (0.65 + 0.7) / 2.0,
                                (0.8 + 0.85 + 0.9 + 1.0) / 4};
                fracPos = new double[] {0.0 / 2.0, 1.0 / 3, 1.0 / 2, 3.0 / 4};
            } else {
                avgBinProbsClass = new double[] {bin0Probs.getColumn(i).meanNumber().doubleValue(),
                                bin1Probs.getColumn(i).meanNumber().doubleValue(),
                                bin2Probs.getColumn(i).meanNumber().doubleValue(),
                                bin4Probs.getColumn(i).meanNumber().doubleValue()};

                fracPos = new double[] {bin0Labels.getColumn(i).sumNumber().doubleValue() / bin0Labels.size(0),
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

    @Test
    public void testLabelAndPredictionCounts() {

        int minibatch = 50;
        int nClasses = 3;

        INDArray arr = Nd4j.rand(minibatch, nClasses);
        arr.diviColumnVector(arr.sum(1));
        INDArray labels = Nd4j.zeros(minibatch, nClasses);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nClasses), 1.0);
        }

        EvaluationCalibration ec = new EvaluationCalibration(5, 5);
        ec.eval(labels, arr);

        int[] expLabelCounts = labels.sum(0).data().asInt();
        // FIXME: int cast
        int[] expPredictionCount = new int[(int) labels.size(1)];
        INDArray argmax = Nd4j.argMax(arr, 1);
        for (int i = 0; i < argmax.length(); i++) {
            expPredictionCount[argmax.getInt(i, 0)]++;
        }

        assertArrayEquals(expLabelCounts, ec.getLabelCountsEachClass());
        assertArrayEquals(expPredictionCount, ec.getPredictionCountsEachClass());
    }

    @Test
    public void testResidualPlots() {

        int minibatch = 50;
        int nClasses = 3;

        INDArray arr = Nd4j.rand(minibatch, nClasses);
        arr.diviColumnVector(arr.sum(1));
        INDArray labels = Nd4j.zeros(minibatch, nClasses);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nClasses), 1.0);
        }

        int numBins = 5;
        EvaluationCalibration ec = new EvaluationCalibration(numBins, numBins);
        ec.eval(labels, arr);

        INDArray absLabelSubProb = Transforms.abs(labels.sub(arr));
        INDArray argmaxLabels = Nd4j.argMax(labels, 1);

        int[] countsAllClasses = new int[numBins];
        int[][] countsByClass = new int[nClasses][numBins]; //Histogram count of |label[x] - p(x)|; rows x are over classes
        double binSize = 1.0 / numBins;

        for (int i = 0; i < minibatch; i++) {
            int actualClassIdx = argmaxLabels.getInt(i, 0);
            for (int j = 0; j < nClasses; j++) {
                double labelSubProb = absLabelSubProb.getDouble(i, j);
                for (int k = 0; k < numBins; k++) {
                    double binLower = k * binSize;
                    double binUpper = (k + 1) * binSize;
                    if (k == numBins - 1)
                        binUpper = 1.0;

                    if (labelSubProb >= binLower && labelSubProb < binUpper) {
                        countsAllClasses[k]++;
                        if (j == actualClassIdx) {
                            countsByClass[j][k]++;
                        }
                    }
                }
            }
        }

        //Check residual plot - all classes/predictions
        Histogram rpAllClasses = ec.getResidualPlotAllClasses();
        int[] rpAllClassesBinCounts = rpAllClasses.getBinCounts();
        assertArrayEquals(countsAllClasses, rpAllClassesBinCounts);

        //Check residual plot - split by labels for each class
        // i.e., histogram of |label[x] - p(x)| only for those examples where label[x] == 1
        for (int i = 0; i < nClasses; i++) {
            Histogram rpCurrClass = ec.getResidualPlot(i);
            int[] rpCurrClassCounts = rpCurrClass.getBinCounts();

            //            System.out.println(Arrays.toString(countsByClass[i]));
            //            System.out.println(Arrays.toString(rpCurrClassCounts));

            assertArrayEquals("Class: " + i, countsByClass[i], rpCurrClassCounts);
        }



        //Check overall probability distribution
        int[] probCountsAllClasses = new int[numBins];
        int[][] probCountsByClass = new int[nClasses][numBins]; //Histogram count of |label[x] - p(x)|; rows x are over classes
        for (int i = 0; i < minibatch; i++) {
            int actualClassIdx = argmaxLabels.getInt(i, 0);
            for (int j = 0; j < nClasses; j++) {
                double prob = arr.getDouble(i, j);
                for (int k = 0; k < numBins; k++) {
                    double binLower = k * binSize;
                    double binUpper = (k + 1) * binSize;
                    if (k == numBins - 1)
                        binUpper = 1.0;

                    if (prob >= binLower && prob < binUpper) {
                        probCountsAllClasses[k]++;
                        if (j == actualClassIdx) {
                            probCountsByClass[j][k]++;
                        }
                    }
                }
            }
        }

        Histogram allProb = ec.getProbabilityHistogramAllClasses();
        int[] actProbCountsAllClasses = allProb.getBinCounts();

        assertArrayEquals(probCountsAllClasses, actProbCountsAllClasses);

        //Check probability distribution - for each label class
        for (int i = 0; i < nClasses; i++) {
            Histogram probCurrClass = ec.getProbabilityHistogram(i);
            int[] actProbCurrClass = probCurrClass.getBinCounts();

            assertArrayEquals(probCountsByClass[i], actProbCurrClass);
        }
    }
}
