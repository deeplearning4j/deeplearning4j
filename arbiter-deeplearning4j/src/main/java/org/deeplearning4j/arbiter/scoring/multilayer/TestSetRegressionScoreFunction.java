package org.deeplearning4j.arbiter.scoring.multilayer;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.scoring.RegressionValue;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 * Score function for regression (including multi-label regression) for a MultiLayerNetwork on a test set
 *
 * @author Alex Black
 */
public class TestSetRegressionScoreFunction implements ScoreFunction<MultiLayerNetwork, DataSetIterator> {
    private final RegressionValue regressionValue;

    /**
     * @param regressionValue The type of evaluation to do: MSE, MAE, RMSE, etc
     */
    public TestSetRegressionScoreFunction(RegressionValue regressionValue) {
        this.regressionValue = regressionValue;
    }

    @Override
    public double score(MultiLayerNetwork model, DataProvider<DataSetIterator> dataProvider, Map<String, Object> dataParameters) {

        DataSetIterator testSet = dataProvider.testData(dataParameters);

        RegressionEvaluation eval = null;
        while (testSet.hasNext()) {
            DataSet next = testSet.next();

            if (eval == null) {
                eval = new RegressionEvaluation(next.getLabels().columns());
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

    @Override
    public boolean minimize() {
        return regressionValue != RegressionValue.CorrCoeff;    //Maximize correlation coefficient, minimize the remaining ones
    }

    @Override
    public String toString() {
        return "TestSetRegressionScoreFunction(type=" + regressionValue + ")";
    }
}
