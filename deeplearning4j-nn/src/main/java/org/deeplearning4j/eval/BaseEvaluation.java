package org.deeplearning4j.eval;

import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.List;

/**
 * BaseEvaluation implement common evaluation functionality (for time series, etc) for {@link Evaluation},
 * {@link RegressionEvaluation}, {@link ROC}, {@link ROCMultiClass} etc.
 *
 * @author Alex Black
 */
public abstract class BaseEvaluation<T extends BaseEvaluation> implements IEvaluation<T> {

    @Override
    public void evalTimeSeries(INDArray labels, INDArray predicted) {
        evalTimeSeries(labels, predicted, null);
    }

    @Override
    public void evalTimeSeries(INDArray labels, INDArray predictions, INDArray labelsMask) {
        Pair<INDArray, INDArray> pair = EvaluationUtils.extractNonMaskedTimeSteps(labels, predictions, labelsMask);
        INDArray labels2d = pair.getFirst();
        INDArray predicted2d = pair.getSecond();

        eval(labels2d, predicted2d);
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, List<? extends Serializable> recordMetaData) {
        eval(labels, networkPredictions);
    }
}
