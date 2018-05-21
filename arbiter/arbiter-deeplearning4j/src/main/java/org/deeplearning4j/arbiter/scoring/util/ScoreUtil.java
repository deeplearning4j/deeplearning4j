package org.deeplearning4j.arbiter.scoring.util;

import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIteratorFactory;



/**
 * Various utilities for functions used in arbiter.
 *
 * @author Adam Gibson
 */
public class ScoreUtil {



    /**
     * Get a {@link DataSetIterator}
     * from the given object whether it's a {@link DataSetIterator}
     * or {@link DataSetIteratorFactory}, any other type will throw
     * an {@link IllegalArgumentException}
     * @param o the object to get the iterator from
     * @return the datasetiterator from the given objects
     */
    public static MultiDataSetIterator getMultiIterator(Object o) {
        if (o instanceof MultiDataSetIterator) {
            return (MultiDataSetIterator) o;
        } else if (o instanceof MultiDataSetIteratorFactory) {
            MultiDataSetIteratorFactory factory = (MultiDataSetIteratorFactory) o;
            return factory.create();
        } else if( o instanceof DataSetIterator ){
            return new MultiDataSetIteratorAdapter((DataSetIterator)o);
        } else if( o instanceof DataSetIteratorFactory ){
            return new MultiDataSetIteratorAdapter(((DataSetIteratorFactory)o).create());
        }

        throw new IllegalArgumentException("Type must either be DataSetIterator or DataSetIteratorFactory");
    }


    /**
     * Get a {@link DataSetIterator}
     * from the given object whether it's a {@link DataSetIterator}
     * or {@link DataSetIteratorFactory}, any other type will throw
     * an {@link IllegalArgumentException}
     * @param o the object to get the iterator from
     * @return the datasetiterator from the given objects
     */
    public static DataSetIterator getIterator(Object o) {
        if (o instanceof DataSetIterator)
            return (DataSetIterator) o;
        else if (o instanceof DataSetIteratorFactory) {
            DataSetIteratorFactory factory = (DataSetIteratorFactory) o;
            return factory.create();
        }

        throw new IllegalArgumentException("Type must either be DataSetIterator or DataSetIteratorFactory");
    }

    /**
     *
     * @param model
     * @param testData
     * @return
     */
    public static Evaluation getEvaluation(MultiLayerNetwork model, DataSetIterator testData) {
        return model.evaluate(testData);
    }

    /**
     * Get the evaluation
     * for the given model and test dataset
     * @param model the model to get the evaluation from
     * @param testData the test data to do the evaluation on
     * @return the evaluation object with accumulated statistics
     * for the current test data
     */
    public static Evaluation getEvaluation(ComputationGraph model, MultiDataSetIterator testData) {
        if (model.getNumOutputArrays() != 1)
            throw new IllegalStateException("GraphSetSetAccuracyScoreFunction cannot be "
                            + "applied to ComputationGraphs with more than one output. NumOutputs = "
                            + model.getNumOutputArrays());

        return model.evaluate(testData);
    }


    /**
     * Get the evaluation
     * for the given model and test dataset
     * @param model the model to get the evaluation from
     * @param testData the test data to do the evaluation on
     * @return the evaluation object with accumulated statistics
     * for the current test data
     */
    public static Evaluation getEvaluation(ComputationGraph model, DataSetIterator testData) {
        if (model.getNumOutputArrays() != 1)
            throw new IllegalStateException("GraphSetSetAccuracyScoreFunctionDataSet cannot be "
                            + "applied to ComputationGraphs with more than one output. NumOutputs = "
                            + model.getNumOutputArrays());

        return model.evaluate(testData);
    }



    /**
     * Score based on the loss function
     * @param model the model to score with
     * @param testData the test data to score
     * @param average whether to average the score
     *                for the whole batch or not
     * @return the score for the given test set
     */
    public static double score(ComputationGraph model, MultiDataSetIterator testData, boolean average) {
        //TODO: do this properly taking into account division by N, L1/L2 etc
        double sumScore = 0.0;
        int totalExamples = 0;
        while (testData.hasNext()) {
            MultiDataSet ds = testData.next();
            long numExamples = ds.getFeatures(0).size(0);
            sumScore += numExamples * model.score(ds);
            totalExamples += numExamples;
        }

        if (!average)
            return sumScore;
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
    public static double score(ComputationGraph model, DataSetIterator testData, boolean average) {
        //TODO: do this properly taking into account division by N, L1/L2 etc
        double sumScore = 0.0;
        int totalExamples = 0;
        while (testData.hasNext()) {
            DataSet ds = testData.next();
            int numExamples = testData.numExamples();

            sumScore += numExamples * model.score(ds);
            totalExamples += numExamples;
        }

        if (!average)
            return sumScore;
        return sumScore / totalExamples;
    }


    /**
     *
     * @param model
     * @param testSet
     * @param regressionValue
     * @return
     */
    public static double score(ComputationGraph model, MultiDataSetIterator testSet, RegressionValue regressionValue) {
        int nOutputs = model.getNumOutputArrays();

        RegressionEvaluation[] evaluations = new RegressionEvaluation[nOutputs];
        for (int i = 0; i < evaluations.length; i++)
            evaluations[i] = new RegressionEvaluation();

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
            sum += getScoreFromRegressionEval(evaluations[i], regressionValue);
        }
        if (regressionValue == RegressionValue.CorrCoeff)
            sum /= totalColumns;

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
    public static double score(ComputationGraph model, DataSetIterator testSet, RegressionValue regressionValue) {
        RegressionEvaluation evaluation = model.evaluateRegression(testSet);
        return getScoreFromRegressionEval(evaluation, regressionValue);
    }


    /**
     * Score the given test data
     * with the given multi layer network
     * @param model model to use
     * @param testData the test data to test with
     * @param average whether to average the score or not
     * @return the score for the given test data given the model
     */
    public static double score(MultiLayerNetwork model, DataSetIterator testData, boolean average) {
        //TODO: do this properly taking into account division by N, L1/L2 etc
        double sumScore = 0.0;
        int totalExamples = 0;
        while (testData.hasNext()) {
            DataSet ds = testData.next();
            int numExamples = ds.numExamples();

            sumScore += numExamples * model.score(ds);
            totalExamples += numExamples;
        }

        if (!average)
            return sumScore;
        return sumScore / totalExamples;
    }


    /**
     * Score the given multi layer network
     * @param model the model to score
     * @param testSet the test set
     * @param regressionValue the regression function to use
     * @return the score from the given test set
     */
    public static double score(MultiLayerNetwork model, DataSetIterator testSet, RegressionValue regressionValue) {
        RegressionEvaluation eval = model.evaluateRegression(testSet);
        return getScoreFromRegressionEval(eval, regressionValue);
    }


    @Deprecated
    public static double getScoreFromRegressionEval(RegressionEvaluation eval, RegressionValue regressionValue) {
        double sum = 0.0;
        int nColumns = eval.numColumns();
        switch (regressionValue) {
            case MSE:
                for (int i = 0; i < nColumns; i++)
                    sum += eval.meanSquaredError(i);
                break;
            case MAE:
                for (int i = 0; i < nColumns; i++)
                    sum += eval.meanAbsoluteError(i);
                break;
            case RMSE:
                for (int i = 0; i < nColumns; i++)
                    sum += eval.rootMeanSquaredError(i);
                break;
            case RSE:
                for (int i = 0; i < nColumns; i++)
                    sum += eval.relativeSquaredError(i);
                break;
            case CorrCoeff:
                for (int i = 0; i < nColumns; i++)
                    sum += eval.correlationR2(i);
                sum /= nColumns;
                break;
        }

        return sum;
    }

}
