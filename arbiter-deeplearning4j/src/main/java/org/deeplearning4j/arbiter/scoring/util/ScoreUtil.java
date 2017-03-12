package org.deeplearning4j.arbiter.scoring.util;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Map;


/**
 * Created by agibsonccc on 3/11/17.
 */
public class ScoreUtil {


    /**
     * Get the evaluation
     * for the given model and test dataset
     * @param model the model to get the evaluation from
     * @param testData the test data to do the evaluation on
     * @return the evaluation object with accumulated statistics
     * for the current test data
     */
    public static  Evaluation getEvaluation(ComputationGraph model, MultiDataSetIterator testData) {
        if (model.getNumOutputArrays() != 1)
            throw new IllegalStateException("GraphSetSetAccuracyScoreFunction cannot be " +
                    "applied to ComputationGraphs with more than one output. NumOutputs = " + model.getNumOutputArrays());

        Evaluation evaluation = new Evaluation();

        while (testData.hasNext()) {
            MultiDataSet next = testData.next();
            if (next.hasMaskArrays()) {
                INDArray[] fMask = next.getFeaturesMaskArrays();
                INDArray[] lMask = next.getLabelsMaskArrays();

                model.setLayerMaskArrays(fMask, lMask);

                INDArray out = model.output(next.getFeatures())[0];

                //Assume this is time series data. Not much point having a mask array for non TS data
                if (lMask != null) {
                    evaluation.evalTimeSeries(next.getLabels(0), out, lMask[0]);
                } else {
                    evaluation.evalTimeSeries(next.getLabels(0), out);
                }

                model.clearLayerMaskArrays();
            } else {
                INDArray out = model.output(false, next.getFeatures())[0];
                if (next.getLabels(0).rank() == 3) evaluation.evalTimeSeries(next.getLabels(0), out);
                else evaluation.eval(next.getLabels(0), out);
            }
        }

        return evaluation;
    }


    /**
     * Get the evaluation
     * for the given model and test dataset
     * @param model the model to get the evaluation from
     * @param testData the test data to do the evaluation on
     * @return the evaluation object with accumulated statistics
     * for the current test data
     */
    public static  Evaluation getEvaluation(ComputationGraph model, DataSetIterator testData) {
        if (model.getNumOutputArrays() != 1)
            throw new IllegalStateException("GraphSetSetAccuracyScoreFunctionDataSet cannot be " +
                    "applied to ComputationGraphs with more than one output. NumOutputs = " + model.getNumOutputArrays());

        Evaluation evaluation = new Evaluation();

        while (testData.hasNext()) {
            DataSet next = testData.next();
            if (next.hasMaskArrays()) {
                INDArray fMask = next.getFeaturesMaskArray();
                INDArray lMask = next.getLabelsMaskArray();

                INDArray[] fMasks = (fMask == null ? null : new INDArray[]{fMask});
                INDArray[] lMasks = (lMask == null ? null : new INDArray[]{lMask});

                model.setLayerMaskArrays(fMasks, lMasks);

                INDArray out = model.output(next.getFeatures())[0];

                //Assume this is time series data. Not much point having a mask array for non TS data
                if (lMask != null) {
                    evaluation.evalTimeSeries(next.getLabels(), out, lMask);
                } else {
                    evaluation.evalTimeSeries(next.getLabels(), out);
                }

                model.clearLayerMaskArrays();
            } else {
                INDArray out = model.output(false, next.getFeatures())[0];
                if (next.getLabels().rank() == 3) evaluation.evalTimeSeries(next.getLabels(), out);
                else evaluation.eval(next.getLabels(), out);
            }
        }

        return evaluation;
    }



    /**
     * Score based on the loss function
     * @param model the model to score with
     * @param testData the test data to score
     * @param average whether to average the score
     *                for the whole batch or not
     * @return the score for the given test set
     */
    public static double score(ComputationGraph model,MultiDataSetIterator testData,boolean average) {
        //TODO: do this properly taking into account division by N, L1/L2 etc
        double sumScore = 0.0;
        int totalExamples = 0;
        while (testData.hasNext()) {
            MultiDataSet ds = testData.next();
            int numExamples = ds.getFeatures(0).size(0);
            sumScore += numExamples * model.score(ds);
            totalExamples += numExamples;
        }

        if (!average) return sumScore;
        return sumScore / totalExamples;
    }

    /**
     * Score based on the loss function
     * @param model the model to score with
     * @param testData the test data to score
     * @param average whether to average the score
     *                for the whole batch or not
     * @return the score for the given test set
     */
    public static double score(ComputationGraph model,DataSetIterator testData,boolean average) {
        //TODO: do this properly taking into account division by N, L1/L2 etc
        double sumScore = 0.0;
        int totalExamples = 0;
        while (testData.hasNext()) {
            DataSet ds = testData.next();
            int numExamples = testData.numExamples();

            sumScore += numExamples * model.score(ds);
            totalExamples += numExamples;
        }

        if (!average) return sumScore;
        return sumScore / totalExamples;
    }


    /**
     *
     * @param model
     * @param testSet
     * @param regressionValue
     * @return
     */
    public static double score(ComputationGraph model, MultiDataSetIterator testSet,RegressionValue regressionValue) {
        int nOutputs = model.getNumOutputArrays();

        RegressionEvaluation[] evaluations = new RegressionEvaluation[nOutputs];
        for (int i = 0; i < evaluations.length; i++) evaluations[i] = new RegressionEvaluation();

        while (testSet.hasNext()) {
            MultiDataSet next = testSet.next();
            INDArray[] labels = next.getLabels();

            if (next.hasMaskArrays()) {
                INDArray[] fMasks = next.getFeaturesMaskArrays();
                INDArray[] lMasks = next.getLabelsMaskArrays();

                model.setLayerMaskArrays(fMasks, lMasks);

                INDArray[] outputs = model.output(false, next.getFeatures());
                for (int i = 0; i < evaluations.length; i++) {
                    if (lMasks != null && lMasks[i] != null) {
                        evaluations[i].evalTimeSeries(labels[i], outputs[i], lMasks[i]);
                    } else {
                        evaluations[i].evalTimeSeries(labels[i], outputs[i]);
                    }
                }

                model.clearLayerMaskArrays();
            } else {
                INDArray[] outputs = model.output(false, next.getFeatures());
                for (int i = 0; i < evaluations.length; i++) {
                    if (labels[i].rank() == 3) {
                        evaluations[i].evalTimeSeries(labels[i], outputs[i]);
                    } else {
                        evaluations[i].eval(labels[i], outputs[i]);
                    }
                }
            }
        }

        double sum = 0.0;
        int totalColumns = 0;
        for (int i = 0; i < evaluations.length; i++) {
            int nColumns = evaluations[i].numColumns();
            totalColumns += nColumns;
            switch (regressionValue) {
                case MSE:
                    for (int j = 0; j < nColumns; j++) sum += evaluations[i].meanSquaredError(j);
                    break;
                case MAE:
                    for (int j = 0; j < nColumns; j++) sum += evaluations[i].meanAbsoluteError(j);
                    break;
                case RMSE:
                    for (int j = 0; j < nColumns; j++) sum += evaluations[i].rootMeanSquaredError(j);
                    break;
                case RSE:
                    for (int j = 0; j < nColumns; j++) sum += evaluations[i].relativeSquaredError(j);
                    break;
                case CorrCoeff:
                    for (int j = 0; j < nColumns; j++) sum += evaluations[i].correlationR2(j);
                    break;
            }
        }
        if (regressionValue == RegressionValue.CorrCoeff) sum /= totalColumns;

        return sum;
    }


    /**
     * Run a {@link RegressionEvaluation}
     * over a {@link DataSetIterator}
     * @param model the model to use
     * @param testSet the test set iterator
     * @param regressionValue  the regression type to use
     * @return
     */
    public static double score(ComputationGraph model,DataSetIterator testSet,RegressionValue regressionValue) {
        RegressionEvaluation evaluation = new RegressionEvaluation();
        while (testSet.hasNext()) {
            DataSet next = testSet.next();
            INDArray labels = next.getLabels();

            if (next.hasMaskArrays()) {
                INDArray fMask = next.getFeaturesMaskArray();
                INDArray lMask = next.getLabelsMaskArray();

                INDArray[] fMasks = (fMask == null ? null : new INDArray[]{fMask});
                INDArray[] lMasks = (lMask == null ? null : new INDArray[]{lMask});

                model.setLayerMaskArrays(fMasks, lMasks);

                INDArray[] outputs = model.output(false, next.getFeatures());
                if (lMasks != null && lMasks[0] != null) {
                    evaluation.evalTimeSeries(labels, outputs[0], lMasks[0]);
                } else {
                    evaluation.evalTimeSeries(labels, outputs[0]);
                }

                model.clearLayerMaskArrays();
            } else {
                INDArray[] outputs = model.output(false, next.getFeatures());
                if (labels.rank() == 3) {
                    evaluation.evalTimeSeries(labels, outputs[0]);
                } else {
                    evaluation.eval(labels, outputs[0]);
                }
            }
        }

        double sum = 0.0;
        int nColumns = evaluation.numColumns();
        switch (regressionValue) {
            case MSE:
                for (int j = 0; j < nColumns; j++) sum += evaluation.meanSquaredError(j);
                break;
            case MAE:
                for (int j = 0; j < nColumns; j++) sum += evaluation.meanAbsoluteError(j);
                break;
            case RMSE:
                for (int j = 0; j < nColumns; j++) sum += evaluation.rootMeanSquaredError(j);
                break;
            case RSE:
                for (int j = 0; j < nColumns; j++) sum += evaluation.relativeSquaredError(j);
                break;
            case CorrCoeff:
                for (int j = 0; j < nColumns; j++) sum += evaluation.correlationR2(j);
                sum /= nColumns;
                break;
        }

        return sum;
    }


    /**
     * Score the given test data
     * with the given multi layer network
     * @param model model to use
     * @param testData the test data to test with
     * @param average whether to average the score or not
     * @return the score for the given test data given the model
     */
    public static double score(MultiLayerNetwork model, DataSetIterator testData,boolean average) {
        //TODO: do this properly taking into account division by N, L1/L2 etc
        double sumScore = 0.0;
        int totalExamples = 0;
        while (testData.hasNext()) {
            DataSet ds = testData.next();
            int numExamples = ds.numExamples();

            sumScore += numExamples * model.score(ds);
            totalExamples += numExamples;
        }

        if (!average) return sumScore;
        return sumScore / totalExamples;
    }


    /**
     * Score the given multi layer network
     * @param model the model to score
     * @param testSet the test set
     * @param regressionValue the regression function to use
     * @return the score from the given test set
     */
    public static double score(MultiLayerNetwork model, DataSetIterator testSet,RegressionValue regressionValue) {
        RegressionEvaluation eval = null;
        while (testSet.hasNext()) {
            DataSet next = testSet.next();

            if (eval == null) {
                eval = new RegressionEvaluation(next.getLabels().size(1));
            }

            INDArray out;
            if (next.hasMaskArrays()) {
                out = model.output(next.getFeatures(), false, next.getFeaturesMaskArray(), next.getLabelsMaskArray());

            } else {
                out = model.output(next.getFeatures(), false);
            }

            if (out.rank() == 3) {
                if (next.getLabelsMaskArray() != null) {
                    eval.evalTimeSeries(next.getLabels(), out, next.getLabelsMaskArray());
                } else {
                    eval.evalTimeSeries(next.getLabels(), out);
                }
            } else {
                eval.eval(next.getLabels(), out);
            }
        }

        if (eval == null) {
            throw new IllegalStateException("test iterator is empty");
        }

        double sum = 0.0;
        int nColumns = eval.numColumns();
        switch (regressionValue) {
            case MSE:
                for (int i = 0; i < nColumns; i++) sum += eval.meanSquaredError(i);
                break;
            case MAE:
                for (int i = 0; i < nColumns; i++) sum += eval.meanAbsoluteError(i);
                break;
            case RMSE:
                for (int i = 0; i < nColumns; i++) sum += eval.rootMeanSquaredError(i);
                break;
            case RSE:
                for (int i = 0; i < nColumns; i++) sum += eval.relativeSquaredError(i);
                break;
            case CorrCoeff:
                for (int i = 0; i < nColumns; i++) sum += eval.correlationR2(i);
                sum /= nColumns;
                break;
        }

        return sum;
    }


}
