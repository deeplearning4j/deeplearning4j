package org.deeplearning4j.arbiter.scoring.graph;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Map;

/**
 * Score function for regression (including multi-label regression) for a ComputationGraph on a DataSetIterator
 *
 * @author Alex Black
 */
public class GraphTestSetRegressionScoreFunctionDataSet implements ScoreFunction<ComputationGraph, DataSetIterator> {

    private final RegressionValue regressionValue;

    /**
     * @param regressionValue The type of evaluation to do: MSE, MAE, RMSE, etc
     */
    public GraphTestSetRegressionScoreFunctionDataSet(RegressionValue regressionValue) {
        this.regressionValue = regressionValue;
    }

    @Override
    public double score(ComputationGraph model, DataProvider<DataSetIterator> dataProvider, Map<String, Object> dataParameters) {

        DataSetIterator testSet = dataProvider.testData(dataParameters);

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

    @Override
    public boolean minimize() {
        return regressionValue != RegressionValue.CorrCoeff;    //Maximize correlation coefficient, minimize the remaining ones
    }

    @Override
    public String toString() {
        return "GraphTestSetRegressionScoreFunctionDataSet(type=" + regressionValue + ")";
    }
}
