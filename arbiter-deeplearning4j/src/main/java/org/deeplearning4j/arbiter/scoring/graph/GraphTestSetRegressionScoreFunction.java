package org.deeplearning4j.arbiter.scoring.graph;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Map;

/**
 * Score function for regression (including multi-label regression, and multiple output arrays) for a ComputationGraph
 * on a MultiDataSetIterator
 *
 * @author Alex Black
 */
public class GraphTestSetRegressionScoreFunction implements ScoreFunction<ComputationGraph, MultiDataSetIterator> {

    private final RegressionValue regressionValue;

    /**
     * @param regressionValue The type of evaluation to do: MSE, MAE, RMSE, etc
     */
    public GraphTestSetRegressionScoreFunction(RegressionValue regressionValue) {
        this.regressionValue = regressionValue;
    }

    @Override
    public double score(ComputationGraph model, DataProvider<MultiDataSetIterator> dataProvider, Map<String, Object> dataParameters) {

        MultiDataSetIterator testSet = dataProvider.testData(dataParameters);

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

    @Override
    public boolean minimize() {
        return regressionValue != RegressionValue.CorrCoeff;    //Maximize correlation coefficient, minimize the remaining ones
    }

    @Override
    public String toString() {
        return "GraphTestSetRegressionScoreFunction(type=" + regressionValue + ")";
    }
}
